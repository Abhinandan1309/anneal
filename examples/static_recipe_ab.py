"""Why did Anneal's static INT8 lose 4.3pp on ResNet-18 when Olive's lost none?

The two recipes differ in several ways at once. This isolates them: every variant is
calibrated on the same 64 Imagenette *train* images and differs only in

* activation type — uint8 activations with int8 weights (U8S8), or int8 for both (S8S8)
* per-channel vs per-tensor weight scales
* reduce_range (7-bit weights)

First run (Zen 2 Ryzen 7 4800H, AVX2, no VNNI): the original hypothesis — that unsigned
activations were to blame — was wrong. S8S8 per-channel broke exactly as badly as U8S8.
Full-range per-channel *weights* were the problem, and reduce_range (7-bit weights) fixed it,
consistent with the 16-bit intermediate saturation onnxruntime documents for x86 CPUs
without VNNI. examples/hardware_lab runs this same study on other CPUs to test that
explanation where VNNI is present.

    python examples/static_recipe_ab.py --eval-limit 1024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anneal.core.artifact import ModelArtifact, sample_shape
from anneal.core.audit import mcnemar_exact, paired_delta_ci, predict
from anneal.core.dataset import load_calibset, load_evalset
from anneal.core.measure import Benchmarker
from anneal.core.targets import get_target
from anneal.core.transforms import TransformContext, apply_transform

HERE = Path(__file__).parent


def model_path(name: str) -> Path:
    """Where a torchvision model's FP32 export lives (exported on first use)."""
    if name == "resnet18":  # kept at its original location, which other examples use
        return HERE / "resnet18-cpu1t" / "models" / "resnet18-fp32.onnx"
    return HERE / "models" / f"{name}-fp32.onnx"


def candidates_dir(name: str) -> Path:
    return HERE / "static_ab_candidates" if name == "resnet18" else HERE / "static_ab_candidates" / name

VARIANTS = {
    "U8S8 per-channel (Anneal's old default)": {"per_channel": True, "activation_type": "uint8"},
    "U8S8 per-channel + reduce_range": {
        "per_channel": True, "activation_type": "uint8", "reduce_range": True,
    },
    "S8S8 per-channel": {"per_channel": True, "activation_type": "int8"},
    "U8S8 per-tensor": {"per_channel": False, "activation_type": "uint8"},
    "S8S8 per-tensor (Olive's default)": {"per_channel": False, "activation_type": "int8"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-limit", type=int, default=1024)
    parser.add_argument("--target", default="cpu-4t")
    parser.add_argument("--model", default="resnet18", help="torchvision model name")
    parser.add_argument("--out", default=str(HERE / "olive_resnet18" / "static_recipe_ab.json"))
    args = parser.parse_args()

    model = model_path(args.model)
    if not model.exists():
        from anneal.models import export_torchvision

        export_torchvision(args.model, model)
    cache = Path.home() / ".anneal_cache"
    shape = sample_shape(model)
    evalset = load_evalset("imagenette", cache_dir=cache, batch_size=32,
                           limit=args.eval_limit, sample_shape=shape)
    calibset = load_calibset("imagenette", cache_dir=cache, batch_size=32, limit=64,
                             sample_shape=shape)
    target = get_target(args.target)
    bench = Benchmarker(target, warmup=10, runs=50)

    base = ModelArtifact(path=model)
    from anneal.core.saturation import analyse, summarise

    probe = [next(iter(load_evalset("imagenette", cache_dir=cache, batch_size=8, limit=8,
                                    sample_shape=shape).batches()))[0]]
    base_pred, labels = predict(base, target, evalset)
    base_right = base_pred == labels
    base_lat = bench.measure(base).latency_ms_p50
    n = len(labels)
    print(f"FP32: top-1 {base_right.mean() * 100:.2f}%  p50 {base_lat:.2f}ms  (n={n}, {target.name})")

    rows = []
    for name, params in VARIANTS.items():
        ctx = TransformContext(workdir=candidates_dir(args.model), calibset=calibset,
                               calib_samples=64)
        params = {"calibrate_method": "minmax", "reduce_range": False, **params}
        cand = apply_transform("quantize_static_int8", params, base, ctx)
        pred, _ = predict(cand, target, evalset)
        right = pred == labels
        b = int(np.sum(base_right & ~right))
        c = int(np.sum(~base_right & right))
        delta, lo, hi = paired_delta_ci(b, c, n)
        lat = bench.measure(cand).latency_ms_p50
        # The emulation is hardware-independent: every machine predicts the same thing, and
        # only the non-VNNI x86 ones should see it happen.
        predicted = summarise(analyse(cand.path, probe, n_positions=128))
        row = {
            "predicted_saturation": predicted,
            "variant": name,
            "params": params,
            "accuracy": float(right.mean()),
            "delta_pp": delta,
            "ci95_pp": [lo, hi],
            "mcnemar_p": mcnemar_exact(b, c),
            "regressions": b,
            "fixes": c,
            "changed_fraction": float(np.mean(pred != base_pred)),
            "p50_ms": lat,
            "speedup": base_lat / lat,
        }
        rows.append(row)
        print(f"{name:42s} top-1 {row['accuracy'] * 100:6.2f}%  "
              f"{delta:+6.2f}pp [{lo:+.2f},{hi:+.2f}] p={row['mcnemar_p']:.2g}  "
              f"changed {row['changed_fraction'] * 100:5.1f}%  {row['speedup']:.2f}x")

    # Time FP32 again: on a shared or battery-powered machine the performance state can move
    # during the study, and then no speedup above is comparable.
    from anneal.core.environment import drift, snapshot, warnings_for

    base_lat_end = bench.measure(base).latency_ms_p50
    moved = drift(base_lat, base_lat_end)
    print(f"FP32 re-timed: {base_lat:.2f}ms -> {base_lat_end:.2f}ms ({moved * 100:.1f}% drift)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "target": target.name, "n_eval": n,
        "calibration": "64 Imagenette train images", "fp32_accuracy": float(base_right.mean()),
        "fp32_p50_ms": base_lat, "fp32_p50_end_ms": base_lat_end, "latency_drift": moved,
        "environment_warnings": warnings_for(snapshot()), "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
