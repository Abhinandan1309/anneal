"""Cross-layer weight equalisation for per-tensor weight quantization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from anneal.core.artifact import ModelArtifact
from anneal.core.cle import cross_layer_equalise, find_cle_pairs
from anneal.core.dataset import SyntheticEvalSet
from anneal.core.transforms import TransformContext, apply_transform

C_IN, C1, C2, C3 = 3, 8, 6, 4


def _weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Three convs whose channels differ in range by up to ~500x, as batch-norm folding leaves them."""
    rng = np.random.default_rng(seed)
    g1 = np.array([30.0, 1.0, 0.05, 2.0, 0.1, 8.0, 0.02, 1.0])
    g2 = np.array([0.03, 5.0, 1.0, 0.2, 20.0, 0.5])
    w1 = rng.standard_normal((C1, C_IN, 3, 3)) * g1.reshape(-1, 1, 1, 1)
    b1 = rng.standard_normal(C1) * g1 * 0.1
    w2 = rng.standard_normal((C2, C1, 3, 3)) * g2.reshape(-1, 1, 1, 1) / np.sqrt(g1).reshape(1, -1, 1, 1)
    b2 = rng.standard_normal(C2) * g2 * 0.1
    w3 = rng.standard_normal((C3, C2, 1, 1)) / g2.reshape(1, -1, 1, 1)
    b3 = rng.standard_normal(C3)
    return {k: v.astype(np.float32) for k, v in
            dict(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3).items()}


def _model(path: Path, pattern: str = "relu") -> Path:
    """conv1 -> act -> [maxpool] -> conv2 [-> relu -> conv3]."""
    w = _weights()
    act1 = {"relu": "Relu", "maxpool": "Relu", "chain": "Relu", "clip": "Clip", "leaky": "LeakyRelu"}[pattern]
    nodes = [helper.make_node("Conv", ["input", "w1", "b1"], ["x1"], name="conv1", pads=[1, 1, 1, 1])]
    inits = ["w1", "b1", "w2", "b2"]
    extra = []
    if act1 == "Clip":
        nodes.append(helper.make_node("Clip", ["x1", "lo", "hi"], ["y1"], name="act1"))
        extra = [numpy_helper.from_array(np.array(0.0, np.float32), "lo"),
                 numpy_helper.from_array(np.array(6.0, np.float32), "hi")]
    elif act1 == "LeakyRelu":
        nodes.append(helper.make_node("LeakyRelu", ["x1"], ["y1"], name="act1", alpha=0.1))
    else:
        nodes.append(helper.make_node("Relu", ["x1"], ["y1"], name="act1"))
    feed = "y1"
    if pattern == "maxpool":
        nodes.append(helper.make_node("MaxPool", ["y1"], ["p1"], name="pool1", kernel_shape=[2, 2], strides=[2, 2]))
        feed = "p1"
    nodes.append(helper.make_node("Conv", [feed, "w2", "b2"], ["x2"], name="conv2", pads=[1, 1, 1, 1]))
    out, c_out = "x2", C2
    if pattern == "chain":
        nodes += [
            helper.make_node("Relu", ["x2"], ["y2"], name="act2"),
            helper.make_node("Conv", ["y2", "w3", "b3"], ["x3"], name="conv3"),
        ]
        inits += ["w3", "b3"]
        out, c_out = "x3", C3
    graph = helper.make_graph(
        nodes, "cle",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", C_IN, 8, 8])],
        [helper.make_tensor_value_info(out, TensorProto.FLOAT, ["batch", None, None, None])],
        [numpy_helper.from_array(w[k], k) for k in inits] + extra,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _run(path: Path, x: np.ndarray) -> np.ndarray:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, {"input": x})[0]


def _conv_weights(path: Path) -> dict[str, np.ndarray]:
    m = onnx.load(str(path))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    return {n.name: inits[n.input[1]] for n in m.graph.node if n.op_type == "Conv"}


def _fake_quant_per_tensor(src: Path, dst: Path) -> Path:
    """Round every conv weight to symmetric per-tensor int8, as TIDL/XINT8 do."""
    m = onnx.load(str(src))
    conv_w = {n.input[1] for n in m.graph.node if n.op_type == "Conv"}
    for init in m.graph.initializer:
        if init.name in conv_w:
            a = numpy_helper.to_array(init).astype(np.float64)
            step = np.abs(a).max() / 127
            init.CopyFrom(numpy_helper.from_array((np.round(a / step) * step).astype(np.float32), init.name))
    onnx.save(m, str(dst))
    return dst


@pytest.fixture
def x() -> np.ndarray:
    return np.random.default_rng(1).standard_normal((4, C_IN, 8, 8)).astype(np.float32)


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("relu", [("conv1", "conv2", ["Relu"])]),
        ("leaky", [("conv1", "conv2", ["LeakyRelu"])]),
        ("maxpool", [("conv1", "conv2", ["Relu", "MaxPool"])]),
        ("chain", [("conv1", "conv2", ["Relu"]), ("conv2", "conv3", ["Relu"])]),
        ("clip", []),
    ],
)
def test_pairs_are_found(tmp_path: Path, pattern, expected):
    pairs = find_cle_pairs(onnx.load(str(_model(tmp_path / "m.onnx", pattern))))
    assert [(p.producer.name, p.consumer.name, p.via) for p in pairs] == expected


@pytest.mark.parametrize("pattern", ["relu", "leaky", "maxpool", "chain"])
def test_float_output_is_unchanged_and_weights_are_balanced(tmp_path: Path, x, pattern):
    src = _model(tmp_path / "m.onnx", pattern)
    dst = tmp_path / "cle.onnx"
    result = cross_layer_equalise(src, dst, check_batch=x)
    assert result.converged and len(result.pairs) == (2 if pattern == "chain" else 1)
    ref = _run(src, x)
    assert np.abs(_run(dst, x) - ref).max() <= 1e-4 * np.abs(ref).max()
    assert result.max_abs_output_change is not None

    before, after = _conv_weights(src), _conv_weights(dst)
    for pair in result.pairs:
        assert pair.balance_before < 0.5
        r1 = np.abs(after[pair.producer]).reshape(after[pair.producer].shape[0], -1).max(axis=1)
        r2 = np.abs(after[pair.consumer]).max(axis=(0, 2, 3))
        np.testing.assert_allclose(r1, r2, rtol=1e-3)
        assert pair.balance_after > 0.999
    assert set(before) == set(after)


@pytest.mark.parametrize("pattern", ["relu", "maxpool", "chain"])
def test_per_tensor_int8_weights_lose_less_after_cle(tmp_path: Path, x, pattern):
    src = _model(tmp_path / "m.onnx", pattern)
    dst = tmp_path / "cle.onnx"
    cross_layer_equalise(src, dst)
    ref = _run(src, x)

    def err(path: Path) -> float:
        q = _run(_fake_quant_per_tensor(path, path.with_name(path.stem + "-q.onnx")), x)
        return float(np.linalg.norm(q - ref) / np.linalg.norm(ref))

    assert err(dst) < 0.25 * err(src)


def test_clip_is_left_alone(tmp_path: Path):
    src = _model(tmp_path / "m.onnx", "clip")
    dst = tmp_path / "cle.onnx"
    result = cross_layer_equalise(src, dst)
    assert result.pairs == [] and result.iterations == 0
    for name, w in _conv_weights(src).items():
        np.testing.assert_array_equal(_conv_weights(dst)[name], w)
    assert [n.op_type for n in onnx.load(str(dst)).graph.node].count("Clip") == 1


def _depthwise_model(path: Path, pad: bool = True, multiplier: int = 2) -> Path:
    """pw conv -> relu -> [zero pad] -> depthwise conv (with a channel multiplier)."""
    rng = np.random.default_rng(3)
    g = np.array([10.0, 0.1, 1.0, 0.02])
    w1 = (rng.standard_normal((4, C_IN, 1, 1)) * g.reshape(-1, 1, 1, 1)).astype(np.float32)
    w2 = rng.standard_normal((4 * multiplier, 1, 3, 3)).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["input", "w1"], ["x1"], name="pw"),
        helper.make_node("Relu", ["x1"], ["y1"], name="act"),
    ]
    if pad:
        nodes.append(helper.make_node("Pad", ["y1", "pads"], ["p1"], name="pad"))
    nodes.append(helper.make_node("Conv", ["p1" if pad else "y1", "w2"], ["out"], name="dw", group=4))
    graph = helper.make_graph(
        nodes,
        "dw",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", C_IN, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", None, None, None])],
        [numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(w2, "w2"),
         numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads")],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))
    return path


def test_depthwise_consumer_keeps_its_output(tmp_path: Path, x):
    src = _depthwise_model(tmp_path / "dw.onnx")
    dst = tmp_path / "cle.onnx"
    result = cross_layer_equalise(src, dst)
    assert [(p.kind, p.via) for p in result.pairs] == [("depthwise", ["Relu", "Pad"])]
    ref = _run(src, x)
    assert np.abs(_run(dst, x) - ref).max() <= 1e-4 * np.abs(ref).max()


def test_tied_weights_are_skipped(tmp_path: Path):
    src = _model(tmp_path / "m.onnx", "relu")
    m = onnx.load(str(src))
    # A second consumer of w2 elsewhere in the graph ties it.
    m.graph.node.append(helper.make_node("Conv", ["x2", "w2"], ["x_tied"], name="tied", pads=[1, 1, 1, 1]))
    m.graph.output.append(helper.make_tensor_value_info("x_tied", TensorProto.FLOAT, ["batch", None, None, None]))
    onnx.save(m, str(src))
    result = cross_layer_equalise(src, tmp_path / "cle.onnx")
    assert result.pairs == [] and result.skipped == ["conv1"]


def test_static_quantization_with_cle_records_lineage(tmp_path: Path):
    src = _model(tmp_path / "m.onnx", "chain")
    calib = SyntheticEvalSet(shape=(C_IN, 8, 8), n=16, batch_size=8, n_classes=4)
    ctx = TransformContext(workdir=tmp_path / "w", calibset=calib)
    out = apply_transform("quantize_static_int8", {"per_channel": False, "cle": True},
                          ModelArtifact(path=src), ctx)
    assert out.lineage[-1].params["cle"] is True
    assert out.meta["cle"]["pairs"] == 2
    assert out.meta["cle"]["max_abs_output_change"] < 1e-2  # outputs reach ~350

    plain = apply_transform("quantize_static_int8", {"per_channel": False},
                            ModelArtifact(path=src), TransformContext(workdir=tmp_path / "p", calibset=calib))
    assert "cle" not in plain.lineage[-1].params and "cle" not in plain.meta


def test_cle_runs_before_activation_equalisation(tmp_path: Path):
    src = _depthwise_model(tmp_path / "dw.onnx", pad=False, multiplier=1)
    calib = SyntheticEvalSet(shape=(C_IN, 8, 8), n=16, batch_size=8, n_classes=4)
    out = apply_transform("quantize_static_int8", {"per_channel": True, "cle": True, "equalize": True},
                          ModelArtifact(path=src), TransformContext(workdir=tmp_path / "w", calibset=calib))
    assert out.meta["cle"]["pairs"] == 1
    # equalize read the CLE'd model: its site is the same pw -> dw pair.
    assert out.meta["equalised_site_ids"] == ["pw"]
    assert out.lineage[-1].params["cle"] is True and out.lineage[-1].params["equalize"] is True


@pytest.mark.parametrize("trans_b", [0, 1])
def test_gemm_pairs_are_exact_and_balanced(tmp_path: Path, trans_b):
    rng = np.random.default_rng(5)
    g = np.array([20.0, 0.05, 1.0, 0.3, 3.0])
    w1 = rng.standard_normal((5, 7)) * g.reshape(-1, 1)  # [N, K] layout
    w2 = rng.standard_normal((3, 5)) / g.reshape(1, -1)
    b1 = rng.standard_normal(5) * g
    if not trans_b:
        w1, w2 = w1.T, w2.T
    f = lambda a, n: numpy_helper.from_array(a.astype(np.float32), n)  # noqa: E731
    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["input", "w1", "b1"], ["h"], name="fc1", transB=trans_b),
            helper.make_node("Relu", ["h"], ["r"], name="act"),
            helper.make_node("Gemm", ["r", "w2"], ["out"], name="fc2", transB=trans_b),
        ],
        "mlp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 7])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 3])],
        [f(w1, "w1"), f(b1, "b1"), f(w2, "w2")],
    )
    src = tmp_path / "mlp.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(src))
    dst = tmp_path / "cle.onnx"
    result = cross_layer_equalise(src, dst)
    assert [p.kind for p in result.pairs] == ["gemm"] and result.pairs[0].balance_after > 0.999
    xin = rng.standard_normal((4, 7)).astype(np.float32)
    run = lambda p: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"input": xin})[0]  # noqa: E731
    ref = run(src)
    assert np.abs(run(dst) - ref).max() <= 1e-4 * np.abs(ref).max()
