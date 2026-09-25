"""Anneal's recipes against published post-training baselines on the full ImageNet val set.

Every recipe is built by the same pipeline (onnxruntime static INT8, QDQ, U8S8, per-channel
weights) from the same 64 held-out calibration images, and scored on the same 49,000
validation images the calibration never touched. Baselines:

* ``minmax``            onnxruntime's default ("max" calibration; NVIDIA, Wu et al. 2020)
* ``entropy``           KL/entropy calibration, NVIDIA's best PTQ result on EfficientNet-B0
* ``percentile``        onnxruntime's percentile (symmetric, 99.999), NVIDIA's other calibrator
* ``anneal (32-bit)``   `anneal advise` for a 32-bit-accumulating CPU (ARM, VNNI)

Scoring:

* *emulated*: the QDQ graph run in float, i.e. what a 32-bit-accumulating CPU computes. This
  is also how papers simulate quantization, so it is the column comparable to them.
* *fused, this laptop*: the advised recipe *for this CPU* (x86 without VNNI) on its real
  kernels, the deployment number here.

One data pass per model: each batch is decoded once and fed to FP32 and every recipe, so all
see identical inputs. Per-image predictions are saved per model, so a crash loses one model
and every statistic can be recomputed.

    python examples/imagenet/run_imagenet.py --models efficientnet_b0,resnet50 --limit 49000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from anneal.core.advise import advise
from anneal.core.artifact import ModelArtifact, sample_shape
from anneal.core.audit import mcnemar_exact, paired_delta_ci
from anneal.core.dataset import load_calibset, load_evalset
from anneal.core.environment import cpu_features, snapshot, warnings_for
from anneal.core.transforms import TransformContext, apply_transform

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = {"per_channel": True, "activation_type": "uint8", "calib_samples": 64}
BASELINES = {
    "minmax": {**BASE, "calibrate_method": "minmax"},
    "entropy": {**BASE, "calibrate_method": "entropy"},
    "percentile": {**BASE, "calibrate_method": "percentile", "calib_percentile": 99.999},
}
#: Histogram calibrators hold every activation of every calibration image in memory; on a
#: transformer that exceeds this laptop. MinMax only there.
HISTOGRAM_OK = {"efficientnet_b0", "mobilenet_v3_large", "resnet50", "resnet18", "efficientnet_b1"}
#: Transformers and ConvNeXt are scored on a prefix: the emulated passes are too slow on a
#: laptop for all 49,000 (10,000 still resolves about +-0.6pp).
SUBSET = {"vit_b_16": 10_000, "resnet50": 10_000, "convnext_tiny": 10_000}


def model_path(name: str) -> Path:
    if name == "resnet18":
        return ROOT / "examples" / "resnet18-cpu1t" / "models" / "resnet18-fp32.onnx"
    return ROOT / "examples" / "models" / f"{name}-fp32.onnx"


def session(path: Path, fused: bool, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    if not fused:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def stats(ref_right: np.ndarray, right: np.ndarray) -> dict:
    b = int(np.sum(ref_right & ~right))
    c = int(np.sum(~ref_right & right))
    delta, lo, hi = paired_delta_ci(b, c, len(right))
    return {"accuracy": float(right.mean()), "delta_pp": delta, "ci95_pp": [lo, hi],
            "mcnemar_p": mcnemar_exact(b, c), "regressions": b, "fixes": c}


def run_model(name: str, limit: int, threads: int, out: Path, redo: set[str] | None = None) -> dict:
    path = model_path(name)
    shape = sample_shape(path)
    cache = Path.home() / ".anneal_cache"
    calib = load_calibset("imagenet", cache_dir=cache, batch_size=8, limit=64, sample_shape=shape)
    limit = min(limit or 49_000, SUBSET.get(name, 49_000))
    ev = load_evalset("imagenet", cache_dir=cache, batch_size=32, limit=limit, sample_shape=shape)
    ctx = TransformContext(workdir=ROOT / "scratch" / "imagenet" / name, calibset=calib)

    local = cpu_features()["int8_path"]
    recipes: dict[str, tuple[dict, bool]] = {}  # label -> (params, fused)
    for label, params in BASELINES.items():
        if params["calibrate_method"] != "minmax" and name not in HISTOGRAM_OK:
            continue
        recipes[label] = (params, False)
    adv32 = advise(path, "arm-dotprod")
    recipes["anneal (32-bit): " + adv32.recommended.label] = (adv32.recommended.params, False)
    adv_here = advise(path, local)
    recipes[f"anneal ({local}, fused): " + adv_here.recommended.label] = (adv_here.recommended.params, True)
    if adv32.profile.family in ("gated-depthwise", "convnext"):
        # Experimental: dense-consumer equalisation on top of the advice (Imagenette: within noise).
        recipes["anneal (32-bit) + dense equalisation"] = ({**adv32.recommended.params, "equalize_dense": True}, False)
    recipes["minmax (fused, this laptop)"] = (BASELINES["minmax"], True)
    if redo:
        recipes = {k: v for k, v in recipes.items() if k in redo}

    sessions = {"fp32": session(path, True, threads)}
    built = {}
    for label, (params, fused) in recipes.items():
        t = time.time()
        art = apply_transform("quantize_static_int8", dict(params), ModelArtifact(path=path), ctx)
        built[label] = str(art.path)
        sessions[label] = session(art.path, fused, threads)
        print(f"  built {label} ({time.time() - t:.0f}s)", flush=True)

    inp = sessions["fp32"].get_inputs()[0].name
    preds: dict[str, list[np.ndarray]] = {k: [] for k in sessions}
    labels: list[np.ndarray] = []
    t0, n = time.time(), 0
    for x, y in ev.batches():
        for k, s in sessions.items():
            preds[k].append(np.asarray(ev.decode(s.run(None, {inp: x})[0])))
        labels.append(y)
        n += len(y)
        if n % 1024 < len(y):
            print(f"  {n}/{len(ev)} images ({time.time() - t0:.0f}s)", flush=True)
    y = np.concatenate(labels)
    p = {k: np.concatenate(v) for k, v in preds.items()}
    previous = None
    if redo:
        # Merge into the finished run: same images in the same order, so the stored FP32
        # predictions must match exactly, or the merge would mix two different samples.
        old = dict(np.load(out / f"{name}-predictions.npz"))
        if not (np.array_equal(old["labels"], y) and np.array_equal(old["fp32"], p["fp32"])):
            raise RuntimeError(f"{name}: redo does not reproduce the stored FP32 predictions")
        previous = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
        p = {**{k: v for k, v in old.items() if k != "labels"}, **{k.replace("/", "_"): v for k, v in p.items()}}
    np.savez_compressed(out / f"{name}-predictions.npz", labels=y, **{k.replace("/", "_"): v for k, v in p.items()})

    fp_right = p["fp32"] == y
    result = {
        "model": name, "n": int(len(y)), "fp32_accuracy": float(fp_right.mean()),
        "calibration": "64 held-out ImageNet val images (every 50th), disjoint from the scored images",
        "local_int8_path": local, "environment_warnings": warnings_for(snapshot()),
        "recipes": {k: {**stats(fp_right, p[k] == y), "params": recipes[k][0],
                        "mode": "fused" if recipes[k][1] else "emulated", "model_path": built[k]}
                    for k in recipes},
    }
    if previous is not None:
        result["recipes"] = {**previous["recipes"], **result["recipes"]}
        result["redone"] = sorted(set(previous.get("redone", [])) | set(recipes))
    (out / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{name}: FP32 {result['fp32_accuracy'] * 100:.2f}% on {len(y)}", flush=True)
    for k, r in result["recipes"].items():
        print(f"  {k:70s} {r['delta_pp']:+6.2f}pp [{r['ci95_pp'][0]:+.2f},{r['ci95_pp'][1]:+.2f}] "
              f"p={r['mcnemar_p']:.2g}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="efficientnet_b0,mobilenet_v3_large,resnet50,convnext_tiny,vit_b_16")
    ap.add_argument("--limit", type=int, default=None, help="scored images (default: all 49,000)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--redo", default="", help="comma-separated recipe labels to re-score on finished models")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    redo = {r.strip() for r in args.redo.split(",") if r.strip()}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if redo:
            print(f"\n== {name} (redo:{', '.join(sorted(redo))})", flush=True)
            run_model(name, args.limit, args.threads, out, redo)
            continue
        if (out / f"{name}.json").exists():
            print(f"{name}: already done, skipping", flush=True)
            continue
        print(f"\n== {name}", flush=True)
        run_model(name, args.limit, args.threads, out)


if __name__ == "__main__":
    main()
