"""Static INT8 across architectures: which ones break, why, and does equalisation help?

For each torchvision classifier:

1. what its activations and convolutions look like (op histogram, depthwise count),
2. what `anneal imbalance` predicts from the float model, before and after equalisation,
3. top-1 on Imagenette validation images for four static INT8 recipes, each scored
   *fused* (this machine's INT8 kernels) and *emulated* (QDQ graph run in float, standing in
   for CPUs that accumulate in 32 bits: ARM dot-product, x86 VNNI).

Accuracy only; no latency (that is the hardware lab's job).

    python examples/zoo/run_zoo.py --models mobilenet_v2,regnet_y_400mf --eval-limit 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from anneal.core.artifact import ModelArtifact, sample_shape
from anneal.core.dataset import load_calibset, load_evalset
from anneal.core.equalize import equalise, find_sites
from anneal.core.imbalance import analyse, summarise
from anneal.core.transforms import TransformContext, apply_transform

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P = {"calibrate_method": "percentile_asym"}
TRANSFORMER_RECIPES = {
    "minmax (default)": {"per_channel": True},
    "minmax, compute ops only": {"per_channel": True, "quantize_ops": "compute"},
}
_EQ = {"per_channel": True, "equalize": True, **P, "float_stem": True, "reduce_range": True}
DENSE_RECIPES = {
    "minmax (default)": {"per_channel": True},
    "percentile + float stem + reduce_range": {"per_channel": True, **P, "float_stem": True, "reduce_range": True},
    "... + dense equalisation": {"per_channel": True, **P, "float_stem": True, "reduce_range": True,
                                 "equalize_dense": True},
    "equalize + percentile + float stem + reduce_range": _EQ,
    "... + dense equalisation ": {**_EQ, "equalize_dense": True},
}
RECIPES = {
    "minmax (default)": {"per_channel": True},
    "percentile + float stem": {"per_channel": True, **P, "float_stem": True},
    "equalize + percentile + float stem": {"per_channel": True, "equalize": True, **P, "float_stem": True},
    "equalize + percentile + float stem + reduce_range": {
        "per_channel": True, "equalize": True, **P, "float_stem": True, "reduce_range": True,
    },
}


def model_path(name: str) -> Path:
    if name == "resnet18":
        return ROOT / "examples" / "resnet18-cpu1t" / "models" / "resnet18-fp32.onnx"
    return ROOT / "examples" / "models" / f"{name}-fp32.onnx"


def predictions(path: Path, ev, fused: bool, threads: int) -> np.ndarray:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    if not fused:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0].name
    return np.concatenate([np.asarray(ev.decode(s.run(None, {i: x})[0])) for x, _ in ev.batches()])


def describe(path: Path) -> dict:
    m = onnx.load(str(path))
    ops = Counter(n.op_type for n in m.graph.node)
    inits = {i.name: i for i in m.graph.initializer}
    dw = 0
    for n in m.graph.node:
        if n.op_type == "Conv" and n.input[1] in inits:
            g = next((a.i for a in n.attribute if a.name == "group"), 1)
            if g > 1 and inits[n.input[1]].dims[1] == 1:
                dw += 1
    return {"ops": dict(ops.most_common()), "depthwise_convs": dw,
            "params_m": round(sum(int(np.prod(i.dims)) for i in m.graph.initializer) / 1e6, 2)}


def run_one(name: str, eval_limit: int, threads: int, out_dir: Path, recipes: str = "cnn") -> dict:
    path = model_path(name)
    if not path.exists():
        from anneal.models import export_torchvision

        export_torchvision(name, path)
    shape = sample_shape(path)
    cache = Path.home() / ".anneal_cache"
    ev = load_evalset("imagenette", cache_dir=cache, batch_size=32, limit=eval_limit, sample_shape=shape)
    calib = load_calibset("imagenette", cache_dir=cache, batch_size=8, limit=64, sample_shape=shape)  # small batches: same statistics, less memory
    work = ROOT / "scratch" / "zoo" / name
    ctx = TransformContext(workdir=work, calibset=calib)
    result: dict = {"model": name, **describe(path)}
    result["equalisable_sites"] = len(find_sites(onnx.load(str(path))))

    before = summarise(analyse(path, calib.calibration_batches(64)))
    eq_path = work / "equalised-fp32.onnx"
    work.mkdir(parents=True, exist_ok=True)
    eq = equalise(path, eq_path, calib.calibration_batches(64))
    after = summarise(analyse(eq_path, calib.calibration_batches(64))) if eq.sites else before
    result["imbalance"] = {"before": before, "after": after, "equalisation": eq.summary()}

    labels = np.concatenate([np.asarray(y) for _, y in ev.batches()])
    fp = predictions(path, ev, True, threads)
    result["fp32"] = float((fp == labels).mean())
    result["n_eval"] = int(len(labels))
    print(f"\n{name}: FP32 {result['fp32'] * 100:.1f}% on {len(labels)} | {result['depthwise_convs']} dw convs, "
          f"{result['equalisable_sites']} equalisable sites | imbalance flagged {before['flagged']} -> {after['flagged']}",
          flush=True)
    rows = {}
    sets = {"transformer": TRANSFORMER_RECIPES, "dense": DENSE_RECIPES, "cnn": RECIPES}
    for label, params in sets[recipes].items():
        t = time.time()
        try:
            art = apply_transform("quantize_static_int8", {"activation_type": "uint8", **params},
                                  ModelArtifact(path=path), ctx)
        except Exception as exc:  # a recipe that cannot be built is a result too
            rows[label] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"  {label:48s} ERROR {rows[label]['error'][:120]}", flush=True)
            continue
        row = {}
        for fused in (True, False):
            p = predictions(art.path, ev, fused, threads)
            key = "fused" if fused else "emulated"
            b = int(((fp == labels) & (p != labels)).sum())
            c = int(((fp != labels) & (p == labels)).sum())
            row[key] = {"acc": float((p == labels).mean()), "agree": float((p == fp).mean()),
                        "regressions": b, "fixes": c}
        rows[label] = row
        print(f"  {label:48s} fused {row['fused']['acc'] * 100:5.1f}% ({(row['fused']['acc'] - result['fp32']) * 100:+5.1f})"
              f"  emulated {row['emulated']['acc'] * 100:5.1f}% ({(row['emulated']['acc'] - result['fp32']) * 100:+5.1f})"
              f"  [{time.time() - t:.0f}s]", flush=True)
    result["recipes"] = rows
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--eval-limit", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--recipes", default="cnn", choices=["cnn", "transformer", "dense"])
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            run_one(name, args.eval_limit, args.threads, Path(args.out), args.recipes)
        except Exception as exc:
            print(f"\n{name}: FAILED {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
