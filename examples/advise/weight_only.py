"""Is EfficientNet-B1's residual INT8 loss in the weights? Weight-only fake quantization.

Every Conv/Gemm weight is rounded to per-output-channel symmetric int8 (as onnxruntime's
per-channel QDQ does), activations stay float. Done on the plain and the equalised model and scored
on Imagenette against FP32, paired.

    python weight_only.py --model efficientnet_b1 --images 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = Path.home() / ".anneal_cache"


def fake_quant_weights(src: Path, dst: Path) -> dict:
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(src))
    weights = {n.input[1] for n in m.graph.node if n.op_type in ("Conv", "Gemm") and len(n.input) > 1}
    worst = []
    for init in m.graph.initializer:
        if init.name not in weights:
            continue
        w = numpy_helper.to_array(init).astype(np.float64)
        flat = w.reshape(w.shape[0], -1)
        amax = np.abs(flat).max(1, keepdims=True)
        scale = np.where(amax > 0, amax / 127.0, 1.0)
        q = np.clip(np.round(flat / scale), -127, 127) * scale
        err = np.linalg.norm(q - flat, axis=1) / np.maximum(np.linalg.norm(flat, axis=1), 1e-12)
        worst.append((init.name, float(err.max())))
        init.CopyFrom(numpy_helper.from_array(q.reshape(w.shape).astype(np.float32), init.name))
    onnx.save(m, str(dst))
    worst.sort(key=lambda t: -t[1])
    return {"weights": len(worst), "worst_relative_error": worst[:8]}


def main() -> None:
    import onnxruntime as ort

    from anneal.core.artifact import sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import equalise

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b1")
    ap.add_argument("--images", type=int, default=2048)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    work = ROOT / "scratch" / "weight_only" / args.model
    work.mkdir(parents=True, exist_ok=True)
    shape = sample_shape(src)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64, sample_shape=shape)
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=32, limit=args.images, sample_shape=shape)
    eq = work / "equalised.onnx"
    equalise(src, eq, list(calib.calibration_batches(64)))
    info = {"plain": fake_quant_weights(src, work / "plain-w8.onnx"), "equalised": fake_quant_weights(eq, work / "eq-w8.onnx")}
    models = {"fp32": src, "plain, int8 weights": work / "plain-w8.onnx", "equalised, int8 weights": work / "eq-w8.onnx"}
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    sessions = {k: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"]) for k, p in models.items()}
    inp = sessions["fp32"].get_inputs()[0].name
    preds, ys = {k: [] for k in sessions}, []
    for x, y in ev.batches():
        for k, s in sessions.items():
            preds[k].append(np.asarray(ev.decode(s.run(None, {inp: x})[0])))
        ys.append(y)
    y = np.concatenate(ys)
    fp = np.concatenate(preds["fp32"]) == y
    rows = {}
    for k in models:
        if k == "fp32":
            continue
        right = np.concatenate(preds[k]) == y
        b, c = int(np.sum(fp & ~right)), int(np.sum(~fp & right))
        d, lo, hi = paired_delta_ci(b, c, len(y))
        rows[k] = {"accuracy": float(right.mean()), "delta_pp": d, "ci95_pp": [lo, hi], "mcnemar_p": mcnemar_exact(b, c)}
        print(f"  {k:28s} {d:+6.2f}pp vs FP32 [{lo:+.2f},{hi:+.2f}]", flush=True)
    for k, v in info.items():
        print(f"  {k}: worst per-channel relative weight error {v['worst_relative_error'][:3]}", flush=True)
    path = HERE / f"{args.model}_weight_only.json"
    path.write_text(json.dumps({"model": args.model, "n": int(len(y)), "fp32_accuracy": float(fp.mean()),
                                "rows": rows, "weights": info}, indent=1), encoding="utf-8")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
