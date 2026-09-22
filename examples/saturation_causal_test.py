"""Causal test: is saturation in one layer the whole of the per-channel INT8 accuracy loss?

`anneal saturation` predicts that full-range per-channel static INT8 on ResNet-18 saturates
16-bit pair sums in exactly one layer, the stem convolution, on x86 CPUs without VNNI. If
that is the cause of the ~4pp loss, then changing *only the stem* should remove it, on the
machine that saturates, with every other layer still full-range per-channel.

Variants, all calibrated on the same 64 Imagenette train images:

* full-range per-channel everywhere          (the broken recipe)
* same, stem convolution left in FP32         (remove the saturating layer)
* same, reduce_range everywhere               (the known blanket fix)

    python examples/saturation_causal_test.py --eval-limit 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from anneal.core.artifact import ModelArtifact, sample_shape
from anneal.core.audit import mcnemar_exact, paired_delta_ci, predict
from anneal.core.dataset import load_calibset, load_evalset
from anneal.core.measure import Benchmarker
from anneal.core.saturation import analyse, summarise
from anneal.core.targets import get_target
from anneal.core.transforms import _EvalSetCalibrationReader, _input_name

HERE = Path(__file__).parent
MODEL = HERE / "resnet18-cpu1t" / "models" / "resnet18-fp32.onnx"
WORK = HERE / "saturation_candidates"
STEM = "/conv1/Conv"


def quantize(name: str, calibset, *, reduce_range: bool, exclude: list[str]) -> Path:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    WORK.mkdir(exist_ok=True)
    pre = WORK / "resnet18-preproc.onnx"
    if not pre.exists():
        quant_pre_process(str(MODEL), str(pre))
    out = WORK / f"{name}.onnx"
    quantize_static(
        str(pre), str(out),
        _EvalSetCalibrationReader(calibset, _input_name(pre), 64),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=reduce_range,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=exclude,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-limit", type=int, default=1024)
    parser.add_argument("--target", default="cpu-1t")
    parser.add_argument("--out", default=str(HERE / "saturation" / "causal_test.json"))
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    from anneal.core.environment import snapshot, warnings_for
    from anneal.core.environment import cpu_features

    cache = Path.home() / ".anneal_cache"
    shape = sample_shape(MODEL)
    evalset = load_evalset("imagenette", cache_dir=cache, batch_size=32, limit=args.eval_limit,
                           sample_shape=shape)
    calibset = load_calibset("imagenette", cache_dir=cache, batch_size=32, limit=64, sample_shape=shape)
    probe = [next(iter(load_evalset("imagenette", cache_dir=cache, batch_size=8, limit=8,
                                    sample_shape=shape).batches()))[0]]
    target = get_target(args.target)
    bench = Benchmarker(target, warmup=10, runs=50)

    base = ModelArtifact(path=MODEL)
    base_pred, labels = predict(base, target, evalset)
    base_right = base_pred == labels
    base_lat = bench.measure(base).latency_ms_p50

    variants = {
        "per-channel, full range (broken)": dict(reduce_range=False, exclude=[]),
        "per-channel, full range, stem in FP32": dict(reduce_range=False, exclude=[STEM]),
        "per-channel, reduce_range everywhere": dict(reduce_range=True, exclude=[]),
    }
    rows = []
    for i, (name, kw) in enumerate(variants.items()):
        path = quantize(f"v{i}", calibset, **kw)
        sat = summarise(analyse(path, probe, n_positions=128))
        pred, _ = predict(ModelArtifact(path=path), target, evalset)
        right = pred == labels
        b, c = int(np.sum(base_right & ~right)), int(np.sum(~base_right & right))
        delta, lo, hi = paired_delta_ci(b, c, len(labels))
        lat = bench.measure(ModelArtifact(path=path)).latency_ms_p50
        row = {"variant": name, "delta_pp": delta, "ci95_pp": [lo, hi], "mcnemar_p": mcnemar_exact(b, c),
               "changed_fraction": float(np.mean(pred != base_pred)), "speedup": base_lat / lat,
               "predicted_saturation": sat}
        rows.append(row)
        print(f"{name:40s} {delta:+6.2f}pp [{lo:+.2f},{hi:+.2f}] p={row['mcnemar_p']:.2g}  "
              f"changed {row['changed_fraction'] * 100:5.1f}%  {row['speedup']:.2f}x  "
              f"predicted: {sat['layers_saturating']} layer(s), worst {sat['worst_layer']} "
              f"{sat['worst_layer_accumulator_rate'] * 100:.1f}%")

    base_lat_end = bench.measure(base).latency_ms_p50
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "cpu": cpu_features(), "target": target.name, "n_eval": len(labels),
        "fp32_accuracy": float(base_right.mean()), "fp32_p50_ms": base_lat,
        "fp32_p50_end_ms": base_lat_end, "environment_warnings": warnings_for(snapshot()),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
