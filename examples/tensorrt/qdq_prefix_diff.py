"""Locate where TensorRT's build of an explicit-QDQ ONNX model diverges from onnxruntime.

Cuts the model after successive tensors (onnx.utils.extract_model), builds each prefix with
TensorRT (INT8 + FP16, scales from the QDQ nodes), runs one input through both, and prints the
SQNR of TensorRT against onnxruntime per cut. The first cut that collapses holds the culprit.

    python qdq_prefix_diff.py model-qdq.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    import onnx
    import onnxruntime as ort
    from onnx.utils import extract_model

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_trt import build, run_engine

    src = Path(sys.argv[1])
    m = onnx.load(str(src))
    inp = m.graph.input[0].name
    # cut points: every DequantizeLinear output feeding a Conv, at ~12 evenly spaced depths
    prod = {o: n for n in m.graph.node for o in n.output}
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    cuts = []
    for c in convs[:: max(1, len(convs) // 12)]:
        t = c.output[0]
        cuts.append(t)
    gate_muls = [n.output[0] for n in m.graph.node if n.name.startswith("anneal_eq_gate_mul_")][:3]
    cuts = gate_muls + cuts
    x = np.random.default_rng(0).standard_normal([d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]).astype(np.float32)
    work = src.parent / "prefix"
    work.mkdir(exist_ok=True)
    for k, t in enumerate(cuts):
        p = work / f"prefix_{k}.onnx"
        try:
            extract_model(str(src), str(p), [inp], [t])
            ref = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {inp: x})[0]
            eng = build(p, "qdq", None, work)
            out, _ = run_engine(eng, [x])
            y = out[0].reshape(ref.shape).astype(np.float64)
            err = ref.astype(np.float64) - y
            sq = 10 * np.log10(max((ref.astype(np.float64) ** 2).sum(), 1e-20) / max((err ** 2).sum(), 1e-20))
            print(f"cut {k:2d} {sq:7.1f} dB  {t}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"cut {k:2d}  FAILED {type(exc).__name__}: {str(exc)[:150]}  {t}", flush=True)


if __name__ == "__main__":
    main()
