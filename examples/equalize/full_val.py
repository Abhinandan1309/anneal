"""Reproduce the prototype numbers through the real transform (anneal.core.equalize).

Each variant is scored twice: fused (this laptop's x86 kernels, no VNNI) and with graph
optimisations off (the QDQ graph in float, standing in for 32-bit-accumulating CPUs).

    python via_transform.py MODEL
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from anneal.core.artifact import ModelArtifact, sample_shape
from anneal.core.dataset import load_calibset, load_evalset
from anneal.core.transforms import TransformContext, apply_transform

name = sys.argv[1]
ROOT = Path(__file__).resolve().parents[2]
model = ROOT / "examples" / "models" / f"{name}-fp32.onnx"
cache = Path.home() / ".anneal_cache"
shape = sample_shape(model)
calib = load_calibset("imagenette", cache_dir=cache, batch_size=32, limit=64, sample_shape=shape)
ev = load_evalset("imagenette", cache_dir=cache, batch_size=32, limit=None, sample_shape=shape)
work = ROOT / "scratch" / "eq" / f"transform-{name}"
ctx = TransformContext(workdir=work, calibset=calib)


def run(path: Path, fused: bool) -> np.ndarray:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    if not fused:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0].name
    return np.concatenate([np.asarray(ev.decode(s.run(None, {i: x})[0])) for x, _ in ev.batches()])


labels = np.concatenate([np.asarray(y) for _, y in ev.batches()])
fp = run(model, True)
out = {"fp32": float((fp == labels).mean())}
print(f"{name}: fp32 {out['fp32'] * 100:.1f}%")
P = {"calibrate_method": "percentile_asym"}
variants = {
    "baseline (minmax)": {"per_channel": True},
    "P + stem": {"per_channel": True, **P, "float_stem": True},
    "EQ": {"per_channel": True, "equalize": True},
    "EQ + P": {"per_channel": True, "equalize": True, **P},
    "EQ + P + stem": {"per_channel": True, "equalize": True, **P, "float_stem": True},
    "EQ + P + stem + reduce_range": {"per_channel": True, "equalize": True, **P, "float_stem": True, "reduce_range": True},
}
for label, params in variants.items():
    art = apply_transform("quantize_static_int8", {"activation_type": "uint8", **params},
                          ModelArtifact(path=model), ctx)
    row = {"meta": art.meta.get("equalisation")}
    for fused in (True, False):
        p = run(art.path, fused)
        key = "fused" if fused else "emulated"
        row[key] = float((p == labels).mean())
        row[key + "_agree"] = float((p == fp).mean())
    row["n"] = int(len(labels))
    b = int(((fp == labels) & (p != labels)).sum()); c = int(((fp != labels) & (p == labels)).sum())
    out[label] = row
    print(f"  {label:30s} fused {row['fused'] * 100:5.1f}%  emulated {row['emulated'] * 100:5.1f}%  "
          f"(agree {row['fused_agree'] * 100:.1f} / {row['emulated_agree'] * 100:.1f})  ")
(work / "result_full.json").write_text(json.dumps(out, indent=1))
