"""EfficientViT loses ~72pp under every per-tensor INT8 recipe tested, equalisation included.

Its linear attention (ReLU kernels, K^T V and a division by a normaliser) produces tensors with a
range no single 8-bit scale can hold. This keeps op groups in float (Imagenette, emulated 32-bit,
paired against FP32): all ops quantized; Conv/MatMul/Gemm only; Conv/Gemm only (attention float);
and the last two with equalisation.

    python efficientvit_fix.py --model efficientvit_b0 --images 512
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = Path.home() / ".anneal_cache"
BASE = {"per_channel": True, "activation_type": "uint8", "calib_samples": 64, "calib_stride": 1,
        "calibrate_method": "percentile_asym", "calib_percentile": 99.99}
VARIANTS = {
    "all ops (percentile)": BASE,
    "compute ops only": {**BASE, "quantize_ops": "compute"},
    "conv only (attention float)": {**BASE, "quantize_ops": "conv"},
    "compute ops only + eq": {**BASE, "quantize_ops": "compute", "equalize": True},
    "conv only + eq": {**BASE, "quantize_ops": "conv", "equalize": True},
    "conv only + eq + int16 top4 (auto)": {**BASE, "quantize_ops": "conv", "equalize": True, "int16_top_k": 4},
    "conv only + eq + int16 top12 (auto)": {**BASE, "quantize_ops": "conv", "equalize": True, "int16_top_k": 12},
    # tensor ranking: depthwise -> Hardswish -> dense 1x1 conv inputs dominate after equalisation
    "conv only + eq + dense eq": {**BASE, "quantize_ops": "conv", "equalize": True, "equalize_dense": True},
    "conv only + eq + dense eq + int16 top4": {**BASE, "quantize_ops": "conv", "equalize": True, "equalize_dense": True,
                                               "int16_top_k": 4},
    # every op quantized, for nets without attention (MobileNetV3-Large's stem block is residual too)
    "eq": {**BASE, "equalize": True},
    "eq (residual)": {**BASE, "equalize": True, "equalize_residual": True},
    # the stem's output also feeds a residual Add, which plain equalisation skips
    "conv only + eq (residual) + dense eq": {**BASE, "quantize_ops": "conv", "equalize": True,
                                             "equalize_residual": True, "equalize_dense": True},
    "conv only + eq (residual) + dense eq + int16 top4": {**BASE, "quantize_ops": "conv", "equalize": True,
                                                          "equalize_residual": True, "equalize_dense": True,
                                                          "int16_top_k": 4},
}


def main() -> None:
    import onnxruntime as ort

    from anneal.core.artifact import ModelArtifact, sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientvit_b0")
    ap.add_argument("--images", type=int, default=512)
    ap.add_argument("--src", help="an ONNX model to use instead of examples/models/<model>-fp32.onnx")
    ap.add_argument("--only", help="comma-separated variant labels")
    ap.add_argument("--tag", default="", help="suffix of the output JSON")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    src = Path(args.src) if args.src else ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    variants = {k: v for k, v in VARIANTS.items() if not args.only or k in args.only.split(",")}
    shape = sample_shape(src)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64, sample_shape=shape)
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=32, limit=args.images, sample_shape=shape)
    ctx = TransformContext(workdir=ROOT / "scratch" / "efficientvit_fix" / (args.model + args.tag), calibset=calib)
    built = {}
    for k, params in variants.items():
        built[k] = apply_transform("quantize_static_int8", dict(params), ModelArtifact(path=src), ctx).path
        gc.collect()
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    emu = ort.SessionOptions()
    emu.enable_cpu_mem_arena = False
    emu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sessions = {"fp32": ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"]),
                **{k: ort.InferenceSession(str(p), emu, providers=["CPUExecutionProvider"]) for k, p in built.items()}}
    inp = sessions["fp32"].get_inputs()[0].name
    preds, ys = {k: [] for k in sessions}, []
    for x, y in ev.batches():
        for k, s in sessions.items():
            preds[k].append(np.asarray(ev.decode(s.run(None, {inp: x})[0])))
        ys.append(y)
    y = np.concatenate(ys)
    fp = np.concatenate(preds["fp32"]) == y
    rows = {}
    for k in variants:
        right = np.concatenate(preds[k]) == y
        b, c = int(np.sum(fp & ~right)), int(np.sum(~fp & right))
        d, lo, hi = paired_delta_ci(b, c, len(y))
        rows[k] = {"accuracy": float(right.mean()), "delta_pp": d, "ci95_pp": [lo, hi], "mcnemar_p": mcnemar_exact(b, c),
                   "params": variants[k], "correct": "".join("1" if r else "0" for r in right)}
        print(f"  {k:32s} {d:+6.2f}pp vs FP32 [{lo:+.2f},{hi:+.2f}]", flush=True)
    out = HERE / f"{args.model}_fix{args.tag}.json"
    out.write_text(json.dumps({"model": args.model, "n": int(len(y)), "fp32_accuracy": float(fp.mean()), "rows": rows,
                               "fp32_correct": "".join("1" if r else "0" for r in fp)},
                              indent=1), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
