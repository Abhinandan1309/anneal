"""clip_gate_inputs: a Clip before every gate, exact for HardSigmoid."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from anneal.core.surrogate import SIGMOID_CLIP, clip_gate_inputs


def _gate_model(path, op, **attrs):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    nodes = [helper.make_node(op, ["x"], ["g"], **attrs), helper.make_node("Mul", ["x", "g"], ["y"])]
    onnx.save(helper.make_model(helper.make_graph(nodes, "g", [x], [y]), opset_imports=[helper.make_opsetid("", 13)],
                                ir_version=8), str(path))
    return path


def _run(p, x):
    return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


def test_hard_sigmoid_clip_is_exact_and_bounded_by_the_linear_span(tmp_path):
    src = _gate_model(tmp_path / "h.onnx", "HardSigmoid", alpha=1 / 6, beta=0.5)
    dst = tmp_path / "hc.onnx"
    assert clip_gate_inputs(src, dst) == 1
    x = np.array([[-50.0, -2.0, 1.0, 40.0]], np.float32)
    assert np.array_equal(_run(src, x), _run(dst, x))
    clip = next(n for n in onnx.load(str(dst)).graph.node if n.op_type == "Clip")
    lo, hi = (onnx.numpy_helper.to_array(i) for i in onnx.load(str(dst)).graph.initializer if i.name in clip.input[1:])
    assert np.isclose(lo, -3.0) and np.isclose(hi, 3.0)


def test_sigmoid_clip_is_within_its_tolerance(tmp_path):
    src = _gate_model(tmp_path / "s.onnx", "Sigmoid")
    dst = tmp_path / "sc.onnx"
    assert clip_gate_inputs(src, dst) == 1
    x = np.array([[-20.0, -1.0, 2.0, 20.0]], np.float32)
    gate_err = np.abs(_run(src, x) - _run(dst, x)) / np.maximum(np.abs(x), 1)
    assert gate_err.max() <= 1 / (1 + np.exp(SIGMOID_CLIP)) + 1e-6
