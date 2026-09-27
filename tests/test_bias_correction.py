"""Analytic bias correction: Q(W) x + b' has the float layer's mean output, per channel."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper, TensorProto

from anneal.core.bias_correction import correct_biases, quantize_weights


def _model(path, w, b, group=1, gemm=False):
    if gemm:
        node = helper.make_node("Gemm", ["x", "w", "b"], ["y"], transB=1)
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, w.shape[1]])
    else:
        node = helper.make_node("Conv", ["x", "w"] + (["b"] if b is not None else []), ["y"], group=group)
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, w.shape[1] * group, 6, 6])
    inits = [numpy_helper.from_array(w, "w")] + ([numpy_helper.from_array(b, "b")] if b is not None else [])
    g = helper.make_graph([node], "g", [x], [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], inits)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), str(path))
    return path


def _run(path, x):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


@pytest.mark.parametrize("group,bias,gemm", [(1, True, False), (4, True, False), (1, False, False), (1, True, True)])
def test_quantized_layer_keeps_the_float_mean(tmp_path, group, bias, gemm):
    rng = np.random.default_rng(0)
    c_in, c_out = 8, 8
    shape = (c_out, c_in) if gemm else (c_out, c_in // group, 3, 3)
    w = (rng.standard_normal(shape) * np.linspace(0.02, 2.0, c_out).reshape(-1, *[1] * (len(shape) - 1))).astype(np.float32)
    b = rng.standard_normal(c_out).astype(np.float32) if bias else None
    xs = [(rng.standard_normal((4, c_in) if gemm else (4, c_in, 6, 6)) + 1.5).astype(np.float32) for _ in range(4)]
    src = _model(tmp_path / "f.onnx", w, b, group, gemm)
    r = correct_biases(src, tmp_path / "bc.onnx", xs, per_channel=False)
    assert len(r.layers) == 1
    # the deployed layer: quantized weights with the corrected bias
    bc = onnx.load(str(tmp_path / "bc.onnx"))
    b_new = numpy_helper.to_array(next(i for i in bc.graph.initializer if i.name != "w"))
    q = _model(tmp_path / "q.onnx", quantize_weights(w, False).astype(np.float32), b, group, gemm)
    qbc = _model(tmp_path / "qbc.onnx", quantize_weights(w, False).astype(np.float32), b_new, group, gemm)
    axes = (0,) if gemm else (0, 2, 3)
    mean = lambda p: np.mean([_run(p, x).mean(axis=axes) for x in xs], axis=0)  # noqa: E731
    before = np.abs(mean(q) - mean(src)).max()
    after = np.abs(mean(qbc) - mean(src)).max()
    assert before > 1e-3
    # exact for Gemm; for Conv each tap sees a slightly different window of x, so near-exact
    assert after < (1e-4 * max(1.0, float(np.abs(mean(src)).max())) if gemm else before / 10)


def test_per_channel_quantization_leaves_little_to_correct():
    w = np.random.default_rng(1).standard_normal((16, 8, 3, 3))
    per_t = np.abs(w - quantize_weights(w, False)).mean()
    per_c = np.abs(w - quantize_weights(w, True)).mean()
    assert per_c <= per_t
    pow2 = quantize_weights(w, False, pow2=True)
    assert np.abs(w - pow2).max() <= np.abs(w).max() / 127 * 2  # step at most doubles
