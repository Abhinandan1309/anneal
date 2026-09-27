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
#: activations TIDL fuses into the Conv before them; its QDQ import wants no Q/DQ in between
#: (docs/quantization.md: such patterns "only require QDQ nodes at the end"). With one there,
#: TIDL's import of MobileNetV3 failed ("Error in topologically sorting the network") at the
#: inlined HardSwish.
FUSED_ACTIVATIONS = ("HardSwish", "Relu", "Clip")


def drop_qdq_before_fused_activation(model) -> int:
    """Conv -> Q -> DQ -> act (Q, DQ, act single-consumer) becomes Conv -> act. Returns the count."""
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    outputs = {o.name for o in g.output}
    drop = []
    for act in g.node:
        if act.op_type not in FUSED_ACTIVATIONS:
            continue
        dq = prod.get(act.input[0])
        q = prod.get(dq.input[0]) if dq is not None and dq.op_type == "DequantizeLinear" else None
        conv = prod.get(q.input[0]) if q is not None and q.op_type == "QuantizeLinear" else None
        if conv is None or conv.op_type != "Conv":
            continue
        if len(cons.get(conv.output[0], [])) != 1 or len(cons.get(q.output[0], [])) != 1 \
                or len(cons.get(dq.output[0], [])) != 1 or dq.output[0] in outputs:
            continue
        act.input[0] = conv.output[0]
        drop += [q, dq]
    for n in drop:
        g.node.remove(n)
    return len(drop) // 2


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
    fused = drop_qdq_before_fused_activation(q)
    q.ir_version = min(q.ir_version, ir)
    # TIDL's import needs every tensor's shape in the file ("Input/output shape unknown" on every
    # node of the quantizer's output made it fail)
    del q.graph.value_info[:]
    q = onnx.shape_inference.infer_shapes(q, strict_mode=True)
    known = {v.name for v in list(q.graph.value_info) + list(q.graph.input) + list(q.graph.output)}
    missing = [o for n in q.graph.node for o in n.output if o not in known]
    onnx.save(q, str(dst))
    print(f"shapes: {len(q.graph.value_info)} inferred, {len(missing)} missing {missing[:3]}", file=sys.stderr)
    print(f"removed {fused} Q/DQ pairs between a Conv and its fused activation", file=sys.stderr)
    print(f"built {dst} (ir {q.ir_version}, opset {[o.version for o in q.opset_import]})", file=sys.stderr)


if __name__ == "__main__":
    main()
