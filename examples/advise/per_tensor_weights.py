"""Equalisation when weights are quantized per tensor (AMD XINT8, TI TIDL's default, older NPUs).

Anneal's equalisation fills the shared activation scale; with per-channel weights that is free.
With per-tensor weights the same rescale moves precision between the channels of A's and B's
weights too. The scales tried are ``sign(s) |s|^alpha s_cle^beta`` (``equalise(mix=...)``):
(1, 0) is Anneal's default, (0, 1) plain cross-layer equalisation (Nagel et al. 2019), and the
rest mix the two. onnxruntime QDQ with per-tensor weights, emulated in float, paired against FP32.

    python per_tensor_weights.py --model efficientnet_b0 --limit 2000
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
BASE = {"activation_type": "uint8", "calib_samples": 64, "calibrate_method": "percentile_asym",
        "calib_percentile": 99.99}
# label -> (mix for gated depthwise sites, mix for dense sites or None = not rewritten)
MIXES = {"plain": (None, None), "eq (1,0)": ((1.0, 0.0), None), "cle (0,1)": ((0.0, 1.0), None),
         "mix (0.5,0.5)": ((0.5, 0.5), None), "mix (1,0.5)": ((1.0, 0.5), None), "mix (1,1)": ((1.0, 1.0), None),
         "mix (0.5,0.5) + dense (0,1)": ((0.5, 0.5), (0.0, 1.0)),
         "mix (0.5,0.5) + dense (0.5,0.5)": ((0.5, 0.5), (0.5, 0.5)),
         "mix (0.5,0.5) + dense (1,0)": ((0.5, 0.5), (1.0, 0.0)),
         "eq (1,0) + dense (0,1)": ((1.0, 0.0), (0.0, 1.0))}


def main() -> None:
    import onnxruntime as ort

    from anneal.core.artifact import ModelArtifact, sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import equalise
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--dataset", default="imagenet")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    shape = sample_shape(src)
    calib = load_calibset(args.dataset, cache_dir=CACHE, batch_size=8, limit=64, sample_shape=shape)
    work = ROOT / "scratch" / "per_tensor_weights" / args.model
    work.mkdir(parents=True, exist_ok=True)
    ctx = TransformContext(workdir=work, calibset=calib)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from anneal.core.equalize_dense import equalise_dense

    pre = work / "preprocessed.onnx"  # decomposes HardSwish, so the dense finder sees the gate
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    batches = list(calib.calibration_batches(64))
    built = {}
    for label, (mix, dense) in MIXES.items():
        fp32 = pre
        if mix is not None:
            fp32 = work / f"{label.replace(' ', '_').replace(',', '-').replace('(', '').replace(')', '')}.onnx"
            equalise(pre, fp32, batches, mix=mix)
            if dense is not None:
                equalise_dense(fp32, fp32, batches, mix=dense)
        built[f"per-tensor w, {label}"] = apply_transform(
            "quantize_static_int8", {**BASE, "per_channel": False}, ModelArtifact(path=fp32), ctx).path
        gc.collect()
    # the per-channel reference: what Anneal does when the target allows it
    built["per-channel w, eq (1,0)"] = apply_transform(
        "quantize_static_int8", {**BASE, "per_channel": True, "equalize": True}, ModelArtifact(path=src), ctx).path

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    emu = ort.SessionOptions()
    emu.enable_cpu_mem_arena = False
    emu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sessions = {"fp32": ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"]),
                **{k: ort.InferenceSession(str(p), emu, providers=["CPUExecutionProvider"]) for k, p in built.items()}}
    ev = load_evalset(args.dataset, cache_dir=CACHE, batch_size=32, limit=args.limit, sample_shape=shape)
    inp = sessions["fp32"].get_inputs()[0].name
    preds, ys = {k: [] for k in sessions}, []
    for x, y in ev.batches():
        for k, s in sessions.items():
            preds[k].append(np.asarray(ev.decode(s.run(None, {inp: x})[0])))
        ys.append(y)
    y = np.concatenate(ys)
    fp = np.concatenate(preds["fp32"]) == y
    rows = {}
    for k in built:
        right = np.concatenate(preds[k]) == y
        b, c = int(np.sum(fp & ~right)), int(np.sum(~fp & right))
        d, lo, hi = paired_delta_ci(b, c, len(y))
        rows[k] = {"accuracy": float(right.mean()), "delta_pp": d, "ci95_pp": [lo, hi], "mcnemar_p": mcnemar_exact(b, c),
                   "correct": "".join("1" if r else "0" for r in right)}
        print(f"  {k:44s} {d:+7.2f}pp vs FP32 [{lo:+.2f},{hi:+.2f}]", flush=True)
    out = HERE / f"{args.model}_per_tensor_weights.json"
    out.write_text(json.dumps({"model": args.model, "dataset": args.dataset, "n": int(len(y)),
                               "fp32_accuracy": float(fp.mean()), "rows": rows,
                               "fp32_correct": "".join("1" if r else "0" for r in fp)}, indent=1), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
