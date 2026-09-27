"""Screen fixes for TI's TDA4VM locally: onnxruntime with TIDL's 8-bit rules, paired against FP32.

TIDL (examples/tidl/run_tidl.py measures the real thing, ~2 h per run in CI) quantizes weights per
tensor and feature maps symmetrically with power-of-two scales. The emulation here uses exactly
those rules (per_channel=False, int8 symmetric activations, pow2_activation_scales), so a fix can
be screened in minutes and only the promising ones sent to the real emulator. Its fidelity is
checked against the real numbers first (MobileNetV2: TIDL 8-bit -10.5pp, uncapped CLE -16.7pp).

    python emulate_study.py --model mobilenet_v2 --images 1000
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
TIDL_LIKE = {"per_channel": False, "activation_type": "int8", "activation_symmetric": True,
             "pow2_activation_scales": True, "calibrate_method": "minmax", "calib_samples": 64}


def prepare(kind: str, src: Path, dst: Path, batches: list[np.ndarray]) -> Path:
    from anneal.core.cle import cross_layer_equalise
    from anneal.core.equalize import equalise

    if kind.endswith("+clip"):  # a Clip before every gate (HardSigmoid: exact; Sigmoid: +-8)
        from anneal.core.surrogate import clip_gate_inputs

        base = prepare(kind[:-5], src, dst.with_name(dst.stem[:-5] + ".onnx"), batches)
        print(f"  {kind}: clipped {clip_gate_inputs(base, dst)} gate inputs", flush=True)
        return dst
    if kind.endswith("+bc"):  # analytic bias correction for per-tensor int8 weights, last
        from anneal.core.bias_correction import correct_biases

        base = prepare(kind[:-3], src, dst.with_name(dst.stem[:-3] + ".onnx"), batches)
        r = correct_biases(base, dst, batches, per_channel=TIDL_LIKE["per_channel"])
        print(f"  {kind}: corrected {len(r.layers)} biases", flush=True)
        return dst
    if kind == "plain":
        return src
    if kind.startswith("relu"):  # relu-float (float cost of ReLU6 -> ReLU), relu-cle-t0.5
        from anneal.core.cle import relax_relu6

        relax_relu6(src, dst)
        if "cle" in kind:
            t = float(kind.split("-t")[1]) if "-t" in kind else 1.0
            cross_layer_equalise(dst, dst, max_scale=1e3, batches=batches if t < 1 else None, t=t)
        return dst
    if kind.startswith("cle"):  # cle, cle16, cle4, cle-t0.5, cle-t0
        cap = 16.0 if kind == "cle16" else 4.0 if kind == "cle4" else 1e3
        t = float(kind.split("-t")[1]) if "-t" in kind else 1.0
        cross_layer_equalise(src, dst, max_scale=cap, batches=batches if t < 1 else None, t=t)
        return dst
    if kind.startswith("eq"):  # eq, eq-pt (per-tensor aware)
        if kind == "eq":
            equalise(src, dst, batches)
        else:  # eq-pt, eq-pt-p2 (power-of-two range alignment), eq-pt-grid (1/s on the int8 grid)
            # eq-pt-derived / eq-pt-grid-derived: noise-optimal scales (anneal.core.equalize_opt)
            equalise(src, dst, batches, residual=True, se=True, mix=(0.5, 0.5), pow2_align=kind.endswith("-p2"),
                     grid_inverse="-grid" in kind, derived=kind.endswith("-derived"))
            cross_layer_equalise(dst, dst, max_scale=4.0, batches=batches, t=0.5)
        return dst
    raise ValueError(kind)


def main() -> None:
    import onnxruntime as ort

    from anneal.core.artifact import ModelArtifact, sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v2")
    ap.add_argument("--images", type=int, default=1000)
    ap.add_argument("--variants", default="plain,cle,cle16,cle4,cle-t0.5,cle-t0")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    shape = sample_shape(src)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64, sample_shape=shape)
    batches = list(calib.calibration_batches(64))
    work = ROOT / "scratch" / "tidl_emulate" / args.model
    work.mkdir(parents=True, exist_ok=True)
    ctx = TransformContext(workdir=work, calibset=calib)
    built = {}
    for kind in [v.strip() for v in args.variants.split(",") if v.strip()]:
        fp32 = prepare(kind, src, work / f"{kind}.onnx", batches)
        built[kind] = str(fp32) if kind.endswith("-float") else str(
            apply_transform("quantize_static_int8", dict(TIDL_LIKE), ModelArtifact(path=fp32), ctx).path)
        print(f"  built {kind}", flush=True)
    emu = ort.SessionOptions()
    emu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    emu.enable_cpu_mem_arena = False
    sessions = {"fp32": ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]),
                **{k: ort.InferenceSession(p, emu, providers=["CPUExecutionProvider"]) for k, p in built.items()}}
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=32, limit=args.images, sample_shape=shape)
    inp = sessions["fp32"].get_inputs()[0].name
    preds, ys = {k: [] for k in sessions}, []
    for x, y in ev.batches():
        for k, s in sessions.items():
            preds[k].append(np.asarray(ev.decode(s.run(None, {inp: x})[0])))
        ys.append(y)
    y = np.concatenate(ys)
    fp = np.concatenate(preds["fp32"]) == y
    rows = {}
    plain = np.concatenate(preds["plain"]) == y if "plain" in preds else None
    for k in built:
        right = np.concatenate(preds[k]) == y
        b, c = int(np.sum(fp & ~right)), int(np.sum(~fp & right))
        d, lo, hi = paired_delta_ci(b, c, len(y))
        row = {"delta_pp": d, "ci95_pp": [lo, hi]}
        extra = ""
        if plain is not None and k != "plain":
            b2, c2 = int(np.sum(plain & ~right)), int(np.sum(~plain & right))
            d2, lo2, hi2 = paired_delta_ci(b2, c2, len(y))
            row["vs_plain_pp"] = [d2, lo2, hi2, mcnemar_exact(b2, c2)]
            extra = f" | vs plain {d2:+.2f} [{lo2:+.2f},{hi2:+.2f}] p={mcnemar_exact(b2, c2):.2g}"
        rows[k] = row
        print(f"  {k:10s} {d:+7.2f}pp vs FP32 [{lo:+.2f},{hi:+.2f}]{extra}", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"emulate_{args.model}.json").write_text(
        json.dumps({"model": args.model, "n": int(len(y)), "rules": TIDL_LIKE, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
