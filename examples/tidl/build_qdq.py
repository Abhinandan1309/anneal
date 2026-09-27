"""Quantize a float ONNX model to QDQ with Anneal's quantizer, for TIDL's pre-quantized import.

Runs outside TIDL's Python environment (its onnx/onnxruntime pair cannot import
onnxruntime.quantization), in a clean one: run_tidl.py calls it through $QDQ_PYTHON.

    python build_qdq.py SRC DST [--per-channel] [--pow2] [--calib 64]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CACHE = Path.home() / ".anneal_cache"


def main() -> None:
    from anneal.core.artifact import ModelArtifact, sample_shape
    from anneal.core.dataset import load_calibset
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--pow2", action="store_true")
    ap.add_argument("--calib", type=int, default=64)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=1, limit=args.calib, sample_shape=sample_shape(src))
    ctx = TransformContext(workdir=dst.parent / f"{dst.stem}-work", calibset=calib)
    params = {"per_channel": args.per_channel, "activation_type": "int8", "activation_symmetric": True,
              "pow2_activation_scales": args.pow2, "calibrate_method": "minmax", "calib_samples": args.calib,
              "float_mixed_outputs": False}
    out = apply_transform("quantize_static_int8", params, ModelArtifact(path=src), ctx).path
    shutil.copy(out, dst)
    import onnx

    # TIDL's older onnx/onnxruntime read this file: keep the source model's IR version
    ir = onnx.load(str(src), load_external_data=False).ir_version
    q = onnx.load(str(dst))
    if q.ir_version > ir:
        q.ir_version = ir
        onnx.save(q, str(dst))
    print(f"built {dst} (ir {q.ir_version}, opset {[o.version for o in q.opset_import]})", file=sys.stderr)


if __name__ == "__main__":
    main()
