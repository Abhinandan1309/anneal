"""Concat equalisation: exact in float, and it fills a Concat's shared scale."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from anneal.core.equalize_concat import equalise_concat, find_concat_sites

C = 4


def _unet_block(path: Path, skip_other_consumer: bool = True, residual_leak: bool = False) -> Path:
    """skip = ReLU(conv_s(x)) -> [MaxPool -> conv_d] and Concat; up = Pad(ConvT(x)); Concat -> conv_c."""
    rng = np.random.default_rng(0)
    ws = (rng.standard_normal((C, 3, 3, 3)) * np.array([0.05, 1, 3, 0.2]).reshape(C, 1, 1, 1)).astype(np.float32)
    bs = rng.standard_normal(C).astype(np.float32) * 0.1
    wt = (rng.standard_normal((3, C, 2, 2)) * 20).astype(np.float32)  # ConvTranspose: (Cin, Cout, k, k)
    bt = rng.standard_normal(C).astype(np.float32)
    wc = rng.standard_normal((5, 2 * C, 3, 3)).astype(np.float32)
    wd = rng.standard_normal((6, C, 3, 3)).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["x", "ws", "bs"], ["s0"], name="conv_s", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["s0"], ["skip"], name="relu_s"),
        helper.make_node("MaxPool", ["x"], ["xp"], name="pool_in", kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("ConvTranspose", ["xp", "wt", "bt"], ["u0"], name="up", strides=[2, 2]),
        helper.make_node("Pad", ["u0", "pads"], ["up_out"], name="pad", mode="constant"),
        helper.make_node("Concat", ["skip", "up_out"], ["cat"], name="cat", axis=1),
        helper.make_node("Conv", ["cat", "wc"], ["out"], name="conv_c", pads=[1, 1, 1, 1]),
    ]
    outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", 5, 8, 8])]
    if skip_other_consumer:  # the next encoder stage also reads the skip tensor
        nodes += [helper.make_node("MaxPool", ["skip"], ["sp"], name="pool_s", kernel_shape=[2, 2], strides=[2, 2]),
                  helper.make_node("Conv", ["sp", "wd"], ["down"], name="conv_d", pads=[1, 1, 1, 1])]
        outputs.append(helper.make_tensor_value_info("down", TensorProto.FLOAT, ["N", 6, 4, 4]))
    if residual_leak:  # the skip also feeds an Add: not compensable, so no site
        nodes.append(helper.make_node("Add", ["skip", "skip"], ["leak"], name="leak"))
        outputs.append(helper.make_tensor_value_info("leak", TensorProto.FLOAT, ["N", C, 8, 8]))
    inits = [numpy_helper.from_array(a, n) for a, n in
             ((ws, "ws"), (bs, "bs"), (wt, "wt"), (bt, "bt"), (wc, "wc"), (wd, "wd"),
              (np.zeros(8, np.int64), "pads"))]
    graph = helper.make_graph(nodes, "unet", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 8, 8])],
                              outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _batches(n=3):
    rng = np.random.default_rng(1)
    return [rng.standard_normal((4, 3, 8, 8)).astype(np.float32) for _ in range(n)]


def _run(path: Path, x: np.ndarray) -> list[np.ndarray]:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})


def test_the_skip_concat_is_found_with_both_consumers_of_the_skip_compensated(tmp_path: Path):
    m = onnx.load(str(_unet_block(tmp_path / "u.onnx")))
    [site] = find_concat_sites(m)
    assert [(p.op_type, ch) for p, ch in site.branches] == [("Conv", C), ("ConvTranspose", C)]
    assert sorted(c.name for c, *_ in site.compensations) == ["conv_c", "conv_d"]


def test_the_rewrite_is_exact_in_float_and_fills_the_shared_range(tmp_path: Path):
    src = _unet_block(tmp_path / "u.onnx")
    dst = tmp_path / "eq.onnx"
    [report] = equalise_concat(src, dst, _batches())
    assert report["median_levels_after"] > report["median_levels_before"]
    x = _batches(1)[0]
    for before, after in zip(_run(src, x), _run(dst, x)):
        assert np.abs(after - before).max() <= 1e-4 * max(1.0, np.abs(before).max())


def test_a_tensor_with_an_uncompensable_consumer_blocks_the_site(tmp_path: Path):
    m = onnx.load(str(_unet_block(tmp_path / "u.onnx", residual_leak=True)))
    assert find_concat_sites(m) == []


def test_a_model_without_sites_is_copied_unchanged(tmp_path: Path):
    src = _unet_block(tmp_path / "u.onnx", residual_leak=True)
    assert equalise_concat(src, tmp_path / "eq.onnx", _batches()) == []
    x = _batches(1)[0]
    for a, b in zip(_run(src, x), _run(tmp_path / "eq.onnx", x)):
        assert np.array_equal(a, b)
