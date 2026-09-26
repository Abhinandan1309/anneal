"""Float sanity check of cross-layer equalisation (with ReLU6 ceilings) on MobileNetV2.

Loads the torchvision MobileNetV2 export (``examples/models/mobilenet_v2-fp32.onnx``, or the
path given), runs :func:`anneal.core.cle.cross_layer_equalise` and prints the pairs found, the
Clip(0, 6) nodes rewritten as Relu -> Min(per-channel ceiling) and the largest change of the
float output on one random input. Exits quietly if the model is not there.

    PYTHONPATH=src python examples/tidl/cle_float_check.py [model.onnx]
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "examples" / "models" / "mobilenet_v2-fp32.onnx"


def main() -> int:
    import onnx

    from anneal.core.cle import cross_layer_equalise

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not src.exists():
        print(f"{src} not found; nothing to check.")
        return 0
    model = onnx.load(str(src))
    ops = Counter(n.op_type for n in model.graph.node)
    print(f"model: {src}  ({ops.get('Conv', 0)} Conv, {ops.get('Clip', 0)} Clip, "
          f"{ops.get('BatchNormalization', 0)} BatchNormalization)")
    if ops.get("BatchNormalization"):
        print("warning: batch-norm is not folded; CLE only sees Conv -> Clip -> Conv chains")

    shape = [d.dim_value if d.dim_value > 0 else 1 for d in model.graph.input[0].type.tensor_type.shape.dim]
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "cle.onnx"
        result = cross_layer_equalise(src, dst, check_batch=x)
        out_ops = Counter(n.op_type for n in onnx.load(str(dst)).graph.node)
        import onnxruntime as ort

        ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(None, {model.graph.input[0].name: x})[0]

    print(f"summary: {result.summary()}")
    kinds = Counter(p.kind for p in result.pairs)
    print(f"pairs: {len(result.pairs)} {dict(kinds)}; clips converted: {len(result.clips_converted)}; "
          f"ops after: Clip {out_ops.get('Clip', 0)}, Min {out_ops.get('Min', 0)}, Relu {out_ops.get('Relu', 0)}")
    if result.pairs:
        bb = np.median([p.balance_before for p in result.pairs])
        ba = np.median([p.balance_after for p in result.pairs])
        print(f"median channel balance: {bb:.3f} -> {ba:.3f}; max scale {result.max_scale:.1f}; "
              f"clamped channels {sum(p.channels_clamped for p in result.pairs)}")
    change = result.max_abs_output_change
    print(f"max |float output change|: {change:.3e}  (max |output| {np.abs(ref).max():.3f}, "
          f"relative {change / np.abs(ref).max():.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
