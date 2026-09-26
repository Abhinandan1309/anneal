"""Predict each equalisation site's value by noise injection, and check it against the device.

``rank_sites`` scores a site by the quantization signal it rescues at the site itself; on the
Galaxy S24 that ranked the most valuable site (the stem, -6.2pp if left unequalised) 9th of 16,
because it ignores how far a site's rounding error travels downstream. This measures it instead:
in the float model, only one site's tensors (the gate input and the depthwise input) are
fake-quantized per-tensor to uint8, and the damage is the share of images whose top-1 prediction
changes. Done on the plain and the equalised model; the site's predicted value is the damage
equalisation removes. Float everywhere else, so it isolates the site, and it needs no device.

The result is compared with the leave-one-site-out values measured on the S24
(examples/qaihub/results/efficientnet_b0-site-value-*.json) by rank correlation.

    python site_sensitivity.py --model efficientnet_b0 --images 512
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


def fake_quant(model_path: Path, dst: Path, tensors: dict[str, tuple[float, float]]) -> Path:
    """Copy of the model with a per-tensor uint8 Quantize/Dequantize pair on each named tensor."""
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(model_path))
    g = model.graph
    for i, (t, (lo, hi)) in enumerate(tensors.items()):
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        scale = max(hi - lo, 1e-12) / 255.0
        zp = int(np.clip(round(-lo / scale), 0, 255))
        s_name, z_name, q, dq = f"fq{i}_scale", f"fq{i}_zp", f"{t}__fq_q", f"{t}__fq"
        g.initializer.extend([numpy_helper.from_array(np.array(scale, np.float32), s_name),
                              numpy_helper.from_array(np.array(zp, np.uint8), z_name)])
        for node in g.node:
            for j, inp in enumerate(node.input):
                if inp == t:
                    node.input[j] = dq
        g.node.extend([helper.make_node("QuantizeLinear", [t, s_name, z_name], [q], name=f"fq{i}_q"),
                       helper.make_node("DequantizeLinear", [q, s_name, z_name], [dq], name=f"fq{i}_dq")])
    onnx.save(model, str(dst))
    return dst


def predict(path: Path, xs: list[np.ndarray]) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep the Q/DQ pairs
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return np.concatenate([s.run(None, {name: x})[0].argmax(1) for x in xs])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    import onnx

    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import _name_unnamed_nodes, channel_ranges, equalise, find_sites, rank_sites

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--images", type=int, default=512)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    work = ROOT / "scratch" / "site_sensitivity" / args.model
    work.mkdir(parents=True, exist_ok=True)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64)
    batches = list(calib.calibration_batches(64))
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=32, limit=args.images)
    xs = [x for x, _ in ev.batches()]

    eq = work / "equalised.onnx"
    equalise(src, eq, batches)
    named = work / "named.onnx"  # site ids are node names; name unnamed nodes the same way equalise does
    m = onnx.load(str(src))
    _name_unnamed_nodes(m)
    onnx.save(m, str(named))
    sites = find_sites(m)
    ranges = {}
    for path, key in ((named, "plain"), (eq, "eq")):
        r = channel_ranges(onnx.load(str(path)), sorted({t for s in sites for t in (s.x, s.y)}), batches)
        ranges[key] = {t: (float(lo.min()), float(hi.max())) for t, (lo, hi) in r.items()}
    ref = {"plain": predict(named, xs), "eq": predict(eq, xs)}
    assert np.mean(ref["plain"] == ref["eq"]) > 0.99, "equalisation must be exact in float"

    rows = {}
    for s in sites:
        dmg = {}
        for key, path in (("plain", named), ("eq", eq)):
            fq = fake_quant(path, work / f"{key}-{len(rows)}.onnx", {t: ranges[key][t] for t in (s.x, s.y)})
            dmg[key] = float(np.mean(predict(fq, xs) != ref[key]))
        rows[s.id] = {"damage_plain": dmg["plain"], "damage_eq": dmg["eq"], "value": dmg["plain"] - dmg["eq"]}
        print(f"  {s.id[:52]:52s} damage {100 * dmg['plain']:5.1f}% -> {100 * dmg['eq']:5.1f}%", flush=True)

    # In context, as on the device: every site's tensors fake-quantized, all sites equalised but one.
    all_t = sorted({t for s in sites for t in (s.x, s.y)})
    ctx_ref = predict(fake_quant(eq, work / "ctx-all.onnx", {t: ranges["eq"][t] for t in all_t}), xs)
    ctx_damage_all = float(np.mean(ctx_ref != ref["plain"]))
    for i, s in enumerate(sites):
        minus = work / f"minus-{i}.onnx"
        equalise(src, minus, batches, sites=[o.id for o in sites if o.id != s.id])
        r = {t: (float(lo.min()), float(hi.max())) for t, (lo, hi) in channel_ranges(onnx.load(str(minus)), all_t, batches).items()}
        d = float(np.mean(predict(fake_quant(minus, work / f"ctx-minus-{i}.onnx", r), xs) != ref["plain"]))
        rows[s.id]["context_damage_minus"] = d
        rows[s.id]["context_value"] = d - ctx_damage_all
        print(f"  in context, without {s.id[:44]:44s} damage {100 * ctx_damage_all:5.1f}% -> {100 * d:5.1f}%", flush=True)

    predicted = dict(rank_sites(src, batches))
    device = None
    dev_files = sorted((ROOT / "examples" / "qaihub" / "results").glob(f"{args.model}-site-value-*.json"))
    if dev_files:
        v = json.loads(dev_files[0].read_text(encoding="utf-8"))["variants"]
        device = {x["left_out_site"]: -x["vs_all"]["delta_pp"] for x in v.values() if "left_out_site" in x and "vs_all" in x}
    out = {"model": args.model, "n": len(ref["plain"]), "sites": rows, "rank_sites_gain": predicted}
    if device:
        common = [s for s in rows if s in device]
        dv = np.array([device[s] for s in common])
        out["device_loo_value_pp"] = {s: device[s] for s in common}
        out["spearman_vs_device"] = {
            "noise_injection_value": spearman(np.array([rows[s]["value"] for s in common]), dv),
            "noise_injection_damage_plain": spearman(np.array([rows[s]["damage_plain"] for s in common]), dv),
            "in_context_leave_one_out": spearman(np.array([rows[s]["context_value"] for s in common]), dv),
            "rank_sites_gain": spearman(np.array([predicted[s] for s in common]), dv),
        }
        print("Spearman vs device leave-one-out value:", json.dumps(out["spearman_vs_device"]), flush=True)
    path = HERE / f"{args.model}_site_sensitivity.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
