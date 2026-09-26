"""Which part of the gated-depthwise recipe does the work, on ImageNet?

The advised recipe for EfficientNet/MobileNetV3 is equalise + asymmetric 99.99 percentile + float
stem. On ResNet-50 the 99.99 percentile over-clipped (resnet50_ablation.json), so this asks the
same of the gated-depthwise recipe, one part at a time, on the first 10,000 scored images of the
ImageNet run, paired against its stored predictions (FP32 must match exactly). Emulated
(32-bit accumulation), as on ARM and NPUs.

    python recipe_ablation.py --model efficientnet_b0
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples" / "imagenet"))

from run_imagenet import model_path, session, stats  # noqa: E402

from anneal.core.artifact import ModelArtifact, sample_shape  # noqa: E402
from anneal.core.audit import mcnemar_exact, paired_delta_ci  # noqa: E402
from anneal.core.dataset import load_calibset, load_evalset  # noqa: E402
from anneal.core.transforms import TransformContext, apply_transform  # noqa: E402

BASE = {"per_channel": True, "activation_type": "uint8", "calib_samples": 64}
EQ, STEM = {"equalize": True}, {"float_stem": True}
A9999 = {"calibrate_method": "percentile_asym", "calib_percentile": 99.99}
A99999 = {"calibrate_method": "percentile_asym", "calib_percentile": 99.999}
S99999 = {"calibrate_method": "percentile", "calib_percentile": 99.999}
RECIPES = {
    "eq + asym 99.999 + stem": {**BASE, **EQ, **A99999, **STEM},
    "eq + sym 99.999 + stem": {**BASE, **EQ, **S99999, **STEM},
    "eq + asym 99.99": {**BASE, **EQ, **A9999},
    "eq + minmax + stem": {**BASE, **EQ, "calibrate_method": "minmax", **STEM},
    "asym 99.99 + stem (no eq)": {**BASE, **A9999, **STEM},
    "eq + minmax": {**BASE, **EQ, "calibrate_method": "minmax"},
    "advised + stem int16": {**BASE, **EQ, **A9999, **STEM, "stem_int16": True},
}
ADVISED = "anneal (32-bit): equalise + percentile + float stem"  # eq + asym 99.99 + stem, stored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--only", default=None, help="comma-separated recipe labels to run (default: all)")
    ap.add_argument("--calib-stride", type=int, default=None,
                    help="calibrate in chunks of this many 8-image batches (bounds RAM; e.g. 1 for EfficientNet-B1)")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    name, path = args.model, model_path(args.model)
    shape = sample_shape(path)
    cache = Path.home() / ".anneal_cache"
    calib = load_calibset("imagenet", cache_dir=cache, batch_size=8, limit=64, sample_shape=shape)
    ev = load_evalset("imagenet", cache_dir=cache, batch_size=32, limit=args.limit, sample_shape=shape)
    ctx = TransformContext(workdir=ROOT / "scratch" / "imagenet" / f"{name}-recipe-ablation", calibset=calib)

    stored_path = ROOT / "examples" / "imagenet" / "results" / f"{name}-predictions.npz"
    recipes = dict(RECIPES)
    if not stored_path.exists():  # no ImageNet run to pair with: score the advised recipe and the default here
        recipes = {"advised: eq + asym 99.99 + stem": {**BASE, **EQ, **A9999, **STEM},
                   "minmax (default)": {**BASE, "calibrate_method": "minmax"}, **recipes}
    sens = HERE / f"{name}_tensor_sensitivity.json"
    if sens.exists():  # mixed precision: the k tensors noise injection ranks most sensitive at 16 bits
        order = json.loads(sens.read_text(encoding="utf-8"))["order"]
        for k in (3, 5):
            recipes[f"advised + top{k} int16"] = {**BASE, **EQ, **A9999, **STEM, "int16_tensors": order[:k]}
    if args.only:
        keep = set(args.only.split(","))
        recipes = {k: v for k, v in recipes.items() if k in keep}
    if args.calib_stride:
        recipes = {k: {**v, "calib_stride": args.calib_stride} for k, v in recipes.items()}
    built = {}
    for label, params in recipes.items():
        built[label] = apply_transform("quantize_static_int8", dict(params), ModelArtifact(path=path), ctx).path
        gc.collect()
        print(f"  built {label}", flush=True)
    def lean(model: Path, fused: bool):
        # No memory arena: nine sessions of a 240-px model otherwise exhaust 16 GB between them.
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.enable_cpu_mem_arena = False
        if not fused:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])

    sessions = {"fp32": lean(path, True), **{k: lean(p, False) for k, p in built.items()}}
    inp = sessions["fp32"].get_inputs()[0].name
    preds = {k: [] for k in sessions}
    labels, t0, n = [], time.time(), 0
    for x, y in ev.batches():
        for k, s in sessions.items():
            preds[k].append(np.asarray(ev.decode(s.run(None, {inp: x})[0])))
        labels.append(y)
        n += len(y)
        if n % 1024 < len(y):
            print(f"  {n}/{len(ev)} images ({time.time() - t0:.0f}s)", flush=True)
    y = np.concatenate(labels)
    p = {k: np.concatenate(v) for k, v in preds.items()}
    m = len(y)
    if stored_path.exists():
        stored = np.load(stored_path)
        if not (np.array_equal(stored["labels"][:m], y) and np.array_equal(stored["fp32"][:m], p["fp32"])):
            raise RuntimeError("does not reproduce the stored ImageNet FP32 predictions on these images")
        p["advised: eq + asym 99.99 + stem"] = stored[ADVISED][:m]
        p["minmax (default)"] = stored["minmax"][:m]

    fp_right = p["fp32"] == y
    ref = p["advised: eq + asym 99.99 + stem"] == y
    rows = {}
    for k in dict.fromkeys(k for k in ["advised: eq + asym 99.99 + stem", "minmax (default)", *recipes] if k in p):
        right = p[k] == y
        row = stats(fp_right, right)
        b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right))
        d, lo, hi = paired_delta_ci(b, c, m)
        row["vs_advised"] = {"delta_pp": d, "ci95_pp": [lo, hi], "mcnemar_p": mcnemar_exact(b, c)}
        row["params"] = recipes.get(k)
        rows[k] = row
        print(f"  {k:36s} {row['delta_pp']:+6.2f}pp vs FP32 | {d:+6.2f}pp vs advised [{lo:+.2f},{hi:+.2f}] "
              f"p={row['vs_advised']['mcnemar_p']:.2g}", flush=True)
    out = HERE / (f"{name}_recipe_ablation.json" if not args.only else f"{name}_recipe_ablation_subset.json")
    out.write_text(json.dumps({"model": name, "n": m, "fp32_accuracy": float(fp_right.mean()), "mode": "emulated",
                               "rows": rows}, indent=1), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
