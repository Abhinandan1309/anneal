"""Where does EfficientNet-B1's remaining INT8 loss sit after equalisation?

On ImageNet the advised recipe takes B1 from -79.4pp to -3.5pp, and `anneal imbalance` flags no
starved convolution input after equalisation. This keeps one group of ops in float at a time on
top of the advised recipe and measures how much of the residual each recovers (Imagenette,
emulated 32-bit, paired against FP32 and against the advised recipe):

* ``compute ops only``  only Conv/MatMul/Gemm quantized: element-wise Mul/Add/Sigmoid in float
* ``float gates``       the equalised gate branches (inserted Mul + Sigmoid) in float
* percentiles           tighter clipping (99.95, 99.9), since 99.99 beat 99.999 by 3.5pp

    python residual_localise.py --model efficientnet_b1 --images 2048
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

BASE = {"per_channel": True, "activation_type": "uint8", "calib_samples": 64, "calib_stride": 1}
ADVISED = {**BASE, "equalize": True, "calibrate_method": "percentile_asym", "calib_percentile": 99.99, "float_stem": True}
VARIANTS = {
    "advised": ADVISED,
    "advised, compute ops only": {**ADVISED, "quantize_ops": "compute"},
    "advised, float gates": {**ADVISED, "float_gates": True},
    "advised at 99.95": {**ADVISED, "calib_percentile": 99.95},
    "advised at 99.9": {**ADVISED, "calib_percentile": 99.9},
    # tensor_sensitivity.py: after equalisation the stem's output alone flips 13.7% of top-1.
    "advised + stem int16": {**ADVISED, "stem_int16": True},
}


def main() -> None:
    import onnxruntime as ort

    from anneal.core.artifact import ModelArtifact, sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b1")
    ap.add_argument("--images", type=int, default=2048)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    shape = sample_shape(src)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64, sample_shape=shape)
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=32, limit=args.images, sample_shape=shape)
    ctx = TransformContext(workdir=ROOT / "scratch" / "residual_localise" / args.model, calibset=calib)

    built = {}
    for label, params in VARIANTS.items():
        built[label] = apply_transform("quantize_static_int8", dict(params), ModelArtifact(path=src), ctx).path
        gc.collect()
        print(f"  built {label}", flush=True)

    def lean(p: Path, emulated: bool) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.enable_cpu_mem_arena = False
        if emulated:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])

    sessions = {"fp32": lean(src, False), **{k: lean(p, True) for k, p in built.items()}}
    inp = sessions["fp32"].get_inputs()[0].name
    preds = {k: [] for k in sessions}
    ys = []
    for x, y in ev.batches():
        for k, s in sessions.items():
            preds[k].append(np.asarray(ev.decode(s.run(None, {inp: x})[0])))
        ys.append(y)
    y = np.concatenate(ys)
    p = {k: np.concatenate(v) for k, v in preds.items()}
    fp, adv = p["fp32"] == y, p["advised"] == y
    rows = {}
    for k in VARIANTS:
        right = p[k] == y
        out = {"accuracy": float(right.mean()), "params": VARIANTS[k]}
        for name, ref in (("vs_fp32", fp), ("vs_advised", adv)):
            b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right))
            d, lo, hi = paired_delta_ci(b, c, len(y))
            out[name] = {"delta_pp": d, "ci95_pp": [lo, hi], "mcnemar_p": mcnemar_exact(b, c)}
        rows[k] = out
        print(f"  {k:28s} {out['vs_fp32']['delta_pp']:+6.2f}pp vs FP32 | {out['vs_advised']['delta_pp']:+6.2f}pp vs advised "
              f"(p={out['vs_advised']['mcnemar_p']:.2g})", flush=True)
    path = HERE / f"{args.model}_residual_localise.json"
    path.write_text(json.dumps({"model": args.model, "n": int(len(y)), "fp32_accuracy": float(fp.mean()),
                                "eval": "imagenette", "rows": rows}, indent=1), encoding="utf-8")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
