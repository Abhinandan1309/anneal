"""Which activation tensors carry EfficientNet-B1's residual INT8 loss after equalisation?

Weights cost only ~0.5pp (weight_only.py) and no conv input is flagged as starved, yet the advised
recipe still loses ~3.5pp on ImageNet. Here every conv input of the equalised model is
fake-quantized alone (per-tensor uint8 min/max, as in site_sensitivity.py) and ranked by the share of
Imagenette top-1 predictions it flips; then the cumulative damage of quantizing the top-k tensors
together is measured, to see whether a few tensors carry it.

    python tensor_sensitivity.py --model efficientnet_b1 --images 512
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
sys.path.insert(0, str(HERE))
from site_sensitivity import fake_quant, predict  # noqa: E402


def main() -> None:
    import onnx

    from anneal.core.artifact import sample_shape
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import channel_ranges, equalise

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b1")
    ap.add_argument("--images", type=int, default=512)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    work = ROOT / "scratch" / "tensor_sensitivity" / args.model
    work.mkdir(parents=True, exist_ok=True)
    shape = sample_shape(src)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64, sample_shape=shape)
    batches = list(calib.calibration_batches(64))
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=32, limit=args.images, sample_shape=shape)
    xs = [x for x, _ in ev.batches()]
    eq = work / "equalised.onnx"
    equalise(src, eq, batches)
    m = onnx.load(str(eq))
    inits = {i.name for i in m.graph.initializer}
    kind = {}
    for n in m.graph.node:
        if n.op_type == "Conv" and n.input[0] not in inits:
            g = next((a.i for a in n.attribute if a.name == "group"), 1)
            kind[n.input[0]] = "depthwise" if g > 1 else "dense"
    tensors = sorted(kind)
    r = {t: (float(lo.min()), float(hi.max())) for t, (lo, hi) in channel_ranges(m, tensors, batches).items()}
    ref = predict(eq, xs)
    rows = {}
    for i, t in enumerate(tensors):
        flips = float(np.mean(predict(fake_quant(eq, work / "one.onnx", {t: r[t]}), xs) != ref))
        rows[t] = {"kind": kind[t], "flips": flips}
        if i % 10 == 0:
            print(f"  {i}/{len(tensors)} tensors", flush=True)
    order = sorted(tensors, key=lambda t: -rows[t]["flips"])
    all_flips = float(np.mean(predict(fake_quant(eq, work / "all.onnx", {t: r[t] for t in tensors}), xs) != ref))
    cumulative = {}
    for k in (1, 3, 5, 10, 20, 40):
        rest = order[k:]  # everything except the top k quantized
        cumulative[k] = float(np.mean(predict(fake_quant(eq, work / "rest.onnx", {t: r[t] for t in rest}), xs) != ref))
    print("top tensors (share of top-1 flipped when quantized alone):", flush=True)
    for t in order[:12]:
        print(f"  {rows[t]['flips'] * 100:5.2f}%  {rows[t]['kind']:9s} {t}", flush=True)
    print(f"all conv inputs quantized: {100 * all_flips:.2f}% flipped; leaving the top k in float:", flush=True)
    for k, v in cumulative.items():
        print(f"  k={k:3d}: {100 * v:.2f}%", flush=True)
    path = HERE / f"{args.model}_tensor_sensitivity.json"
    path.write_text(json.dumps({"model": args.model, "n": len(ref), "all_conv_inputs_flips": all_flips,
                                "float_top_k_flips": cumulative, "order": order, "tensors": rows}, indent=1), encoding="utf-8")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
