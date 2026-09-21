"""Why did Anneal's static INT8 lose 4.3pp on ResNet-18 when Olive's lost none?

The two recipes differ in several ways at once. This isolates them: every variant is
calibrated on the same 64 Imagenette *train* images and differs only in

* activation type — uint8 activations with int8 weights (U8S8), or int8 for both (S8S8)
* per-channel vs per-tensor weight scales
* reduce_range (7-bit weights)

Hypothesis under test: on x86 CPUs without VNNI (this machine is a Zen 2 Ryzen 7 4800H),
U8S8 kernels multiply u8 x s8 pairs and add them into 16-bit intermediates that can
saturate, silently corrupting results. S8S8 and reduce_range both avoid it.

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
MODEL = HERE / "resnet18-cpu1t" / "models" / "resnet18-fp32.onnx"

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
    parser.add_argument("--out", default=str(HERE / "olive_resnet18" / "static_recipe_ab.json"))
    args = parser.parse_args()

    cache = Path.home() / ".anneal_cache"
    shape = sample_shape(MODEL)
    evalset = load_evalset("imagenette", cache_dir=cache, batch_size=32,
                           limit=args.eval_limit, sample_shape=shape)
    calibset = load_calibset("imagenette", cache_dir=cache, batch_size=32, limit=64,
                             sample_shape=shape)
    target = get_target(args.target)
    bench = Benchmarker(target, warmup=10, runs=50)

    base = ModelArtifact(path=MODEL)
    base_pred, labels = predict(base, target, evalset)
    base_right = base_pred == labels
    base_lat = bench.measure(base).latency_ms_p50
    n = len(labels)
    print(f"FP32: top-1 {base_right.mean() * 100:.2f}%  p50 {base_lat:.2f}ms  (n={n}, {target.name})")

    rows = []
    for name, params in VARIANTS.items():
        ctx = TransformContext(workdir=HERE / "static_ab_candidates", calibset=calibset,
                               calib_samples=64)
        params = {"calibrate_method": "minmax", "reduce_range": False, **params}
        cand = apply_transform("quantize_static_int8", params, base, ctx)
        pred, _ = predict(cand, target, evalset)
        right = pred == labels
        b = int(np.sum(base_right & ~right))
        c = int(np.sum(~base_right & right))
        delta, lo, hi = paired_delta_ci(b, c, n)
        lat = bench.measure(cand).latency_ms_p50
        row = {
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

    Path(args.out).write_text(json.dumps({
        "model": MODEL.name, "target": target.name, "n_eval": n,
        "calibration": "64 Imagenette train images", "fp32_accuracy": float(base_right.mean()),
        "fp32_p50_ms": base_lat, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
