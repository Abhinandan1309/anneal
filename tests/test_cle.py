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


#: Clip patterns: (form, lo, hi). form "input" = initializer inputs, "const" = Constant-node
#: inputs, "attr" = opset-10 attributes. None = bound left out.
CLIPS = {
    "clip": ("input", 0.0, 6.0),
    "clip_chain": ("input", 0.0, 6.0),
    "clip_const": ("const", 0.0, 6.0),
    "clip_attr": ("attr", 0.0, 6.0),
    "clip_nomax": ("input", 0.0, None),
    "clip_lo": ("input", 0.5, 6.0),
    "clip_neg": ("input", -1.0, 1.0),
    "clip_attr_lo": ("attr", 0.5, 6.0),
}


def _clip(x: str, y: str, name: str, form: str, lo, hi) -> tuple[list, list]:
    """A Clip node (plus any Constant nodes) and its initializers."""
    if form == "attr":
        attrs = {k: v for k, v in (("min", lo), ("max", hi)) if v is not None}
        return [helper.make_node("Clip", [x], [y], name=name, **attrs)], []
    nodes, inits, ins = [], [], [x]
    for tag, v in (("lo", lo), ("hi", hi)):
        if v is None:
            ins.append("")
            continue
        t = numpy_helper.from_array(np.array(v, np.float32), f"{name}_{tag}")
        if form == "const":
            nodes.append(helper.make_node("Constant", [], [t.name], name=f"{name}_{tag}_c", value=t))
        else:
            inits.append(t)
        ins.append(t.name)
    while ins[-1] == "":
        ins.pop()
    return nodes + [helper.make_node("Clip", ins, [y], name=name)], inits


def _model(path: Path, pattern: str = "relu") -> Path:
    """conv1 -> act -> [maxpool] -> conv2 [-> act -> conv3]."""
    w = _weights()
    act1 = "Clip" if pattern in CLIPS else {"relu": "Relu", "maxpool": "Relu", "chain": "Relu",
                                             "leaky": "LeakyRelu"}[pattern]
    opset = 10 if pattern in CLIPS and CLIPS[pattern][0] == "attr" else 13
    nodes = [helper.make_node("Conv", ["input", "w1", "b1"], ["x1"], name="conv1", pads=[1, 1, 1, 1])]
    inits = ["w1", "b1", "w2", "b2"]
    extra = []
    if act1 == "Clip":
        clip_nodes, extra = _clip("x1", "y1", "act1", *CLIPS[pattern])
        nodes += clip_nodes
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
    if pattern == "clip_chain":
        clip_nodes, more = _clip("x2", "y2", "act2", *CLIPS[pattern])
        nodes += clip_nodes + [helper.make_node("Conv", ["y2", "w3", "b3"], ["x3"], name="conv3")]
        extra += more
        inits += ["w3", "b3"]
        out, c_out = "x3", C3
    graph = helper.make_graph(
        nodes, "cle",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", C_IN, 8, 8])],
        [helper.make_tensor_value_info(out, TensorProto.FLOAT, ["batch", None, None, None])],
        [numpy_helper.from_array(w[k], k) for k in inits] + extra,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)
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
        ("clip", [("conv1", "conv2", ["Clip"])]),
        ("clip_chain", [("conv1", "conv2", ["Clip"]), ("conv2", "conv3", ["Clip"])]),
        ("clip_const", [("conv1", "conv2", ["Clip"])]),
        ("clip_attr", [("conv1", "conv2", ["Clip"])]),
        ("clip_nomax", [("conv1", "conv2", ["Clip"])]),
        ("clip_lo", []),
        ("clip_neg", []),
        ("clip_attr_lo", []),
    ],
)
def test_pairs_are_found(tmp_path: Path, pattern, expected):
    pairs = find_cle_pairs(onnx.load(str(_model(tmp_path / "m.onnx", pattern))))
    assert [(p.producer.name, p.consumer.name, p.via) for p in pairs] == expected
    for p in pairs:
        want = [6.0] if pattern.startswith("clip") and pattern != "clip_nomax" else []
        assert [hi for _, hi in p.ceilings] == want


@pytest.mark.parametrize(
    "pattern", ["relu", "leaky", "maxpool", "chain", "clip", "clip_chain", "clip_const", "clip_attr", "clip_nomax"]
)
def test_float_output_is_unchanged_and_weights_are_balanced(tmp_path: Path, x, pattern):
    src = _model(tmp_path / "m.onnx", pattern)
    dst = tmp_path / "cle.onnx"
    result = cross_layer_equalise(src, dst, check_batch=x)
    chain = pattern in ("chain", "clip_chain")
    assert result.converged and len(result.pairs) == (2 if chain else 1)
    ref = _run(src, x)
    assert np.abs(_run(dst, x) - ref).max() <= 1e-4 * np.abs(ref).max()
    assert result.max_abs_output_change is not None

    ops = [n.op_type for n in onnx.load(str(dst)).graph.node]
    if pattern.startswith("clip") and pattern != "clip_nomax":
        # Every Clip(0, 6) became Relu -> Min(., 6 s) with a [1, C, 1, 1] ceiling; bounds are gone.
        assert result.clips_converted == (["act1", "act2"] if chain else ["act1"])
        assert "Clip" not in ops and "Constant" not in ops and ops.count("Min") == len(result.pairs)
        m = onnx.load(str(dst))
        inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
        assert not any(k.endswith(("_lo", "_hi")) for k in inits)
        for pair in result.pairs:
            ceil = inits[f"{'act1' if pair.producer == 'conv1' else 'act2'}_ceiling"]
            assert ceil.shape == (1, pair.channels, 1, 1)
            assert ceil.min() == pytest.approx(6 * pair.scale_min, rel=1e-5)
            assert ceil.max() == pytest.approx(6 * pair.scale_max, rel=1e-5)
    else:
        assert result.clips_converted == [] and "Min" not in ops
        assert ops.count("Clip") == (1 if pattern == "clip_nomax" else 0)

    before, after = _conv_weights(src), _conv_weights(dst)
    for pair in result.pairs:
        assert pair.balance_before < 0.5
        r1 = np.abs(after[pair.producer]).reshape(after[pair.producer].shape[0], -1).max(axis=1)
        r2 = np.abs(after[pair.consumer]).max(axis=(0, 2, 3))
        np.testing.assert_allclose(r1, r2, rtol=1e-3)
        assert pair.balance_after > 0.999
    assert set(before) == set(after)


@pytest.mark.parametrize("pattern", ["relu", "maxpool", "chain", "clip", "clip_chain"])
def test_per_tensor_int8_weights_lose_less_after_cle(tmp_path: Path, x, pattern):
    src = _model(tmp_path / "m.onnx", pattern)
    dst = tmp_path / "cle.onnx"
    cross_layer_equalise(src, dst)
    ref = _run(src, x)

    def err(path: Path) -> float:
        q = _run(_fake_quant_per_tensor(path, path.with_name(path.stem + "-q.onnx")), x)
        return float(np.linalg.norm(q - ref) / np.linalg.norm(ref))

    assert err(dst) < 0.25 * err(src)


@pytest.mark.parametrize("pattern", ["clip_lo", "clip_neg", "clip_attr_lo"])
def test_clip_with_nonzero_floor_is_left_alone(tmp_path: Path, pattern):
    src = _model(tmp_path / "m.onnx", pattern)
    dst = tmp_path / "cle.onnx"
    result = cross_layer_equalise(src, dst)
    assert result.pairs == [] and result.iterations == 0
    for name, w in _conv_weights(src).items():
        np.testing.assert_array_equal(_conv_weights(dst)[name], w)
    assert [n.op_type for n in onnx.load(str(dst)).graph.node].count("Clip") == 1


def _depthwise_model(path: Path, pad: bool = True, multiplier: int = 2, relu6: bool = False) -> Path:
    """pw conv -> relu (or Clip(0, 6)) -> [zero pad] -> depthwise conv (with a channel multiplier)."""
    rng = np.random.default_rng(3)
    g = np.array([10.0, 0.1, 1.0, 0.02])
    w1 = (rng.standard_normal((4, C_IN, 1, 1)) * g.reshape(-1, 1, 1, 1)).astype(np.float32)
    w2 = rng.standard_normal((4 * multiplier, 1, 3, 3)).astype(np.float32)
    nodes = [helper.make_node("Conv", ["input", "w1"], ["x1"], name="pw")]
    clip_inits = []
    if relu6:
        clip_nodes, clip_inits = _clip("x1", "y1", "act", "input", 0.0, 6.0)
        nodes += clip_nodes
    else:
        nodes.append(helper.make_node("Relu", ["x1"], ["y1"], name="act"))
    if pad:
        nodes.append(helper.make_node("Pad", ["y1", "pads"], ["p1"], name="pad"))
    nodes.append(helper.make_node("Conv", ["p1" if pad else "y1", "w2"], ["out"], name="dw", group=4))
    graph = helper.make_graph(
        nodes,
        "dw",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", C_IN, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", None, None, None])],
        [numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(w2, "w2"),
         numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads")] + clip_inits,
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), str(path))
    return path


def test_depthwise_consumer_keeps_its_output(tmp_path: Path, x):
    src = _depthwise_model(tmp_path / "dw.onnx")
    dst = tmp_path / "cle.onnx"
    result = cross_layer_equalise(src, dst)
    assert [(p.kind, p.via) for p in result.pairs] == [("depthwise", ["Relu", "Pad"])]
    ref = _run(src, x)
    assert np.abs(_run(dst, x) - ref).max() <= 1e-4 * np.abs(ref).max()


@pytest.mark.parametrize("pad", [False, True])
def test_relu6_into_depthwise_is_exact_balanced_and_quantizes_better(tmp_path: Path, x, pad):
    src = _depthwise_model(tmp_path / "dw.onnx", pad=pad, multiplier=1, relu6=True)
    dst = tmp_path / "cle.onnx"
    result = cross_layer_equalise(src, dst)
    assert [(p.kind, p.via) for p in result.pairs] == [("depthwise", ["Clip", "Pad"] if pad else ["Clip"])]
    assert result.clips_converted == ["act"]
    pair = result.pairs[0]
    assert pair.balance_before < 0.5 and pair.balance_after > 0.999
    ops = [n.op_type for n in onnx.load(str(dst)).graph.node]
    assert "Clip" not in ops and ops[:3] == ["Conv", "Relu", "Min"]

    xs = 3 * x  # drive the pw outputs well past 6 so the ceiling is exercised
    ref = _run(src, xs)
    w1 = _conv_weights(src)["pw"][:, :, 0, 0]
    pre = np.einsum("oc,nchw->nohw", w1, xs)
    assert (pre > 6).any() and ((pre > 0) & (pre < 6)).any()
    assert np.abs(_run(dst, xs) - ref).max() <= 1e-4 * np.abs(ref).max()

    def err(path: Path) -> float:
        q = _run(_fake_quant_per_tensor(path, path.with_name(path.stem + "-q.onnx")), xs)
        return float(np.linalg.norm(q - ref) / np.linalg.norm(ref))

    assert err(dst) < 0.25 * err(src)


def test_relu6_of_skipped_pair_is_not_rewritten(tmp_path: Path):
    src = _model(tmp_path / "m.onnx", "clip")
    m = onnx.load(str(src))
    m.graph.node.append(helper.make_node("Conv", ["x2", "w2"], ["x_tied"], name="tied", pads=[1, 1, 1, 1]))
    m.graph.output.append(helper.make_tensor_value_info("x_tied", TensorProto.FLOAT, ["batch", None, None, None]))
    onnx.save(m, str(src))
    dst = tmp_path / "cle.onnx"
    result = cross_layer_equalise(src, dst)
    assert result.pairs == [] and result.clips_converted == []
    assert [n.op_type for n in onnx.load(str(dst)).graph.node].count("Clip") == 1


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
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), str(src))
    dst = tmp_path / "cle.onnx"
    result = cross_layer_equalise(src, dst)
    assert [p.kind for p in result.pairs] == ["gemm"] and result.pairs[0].balance_after > 0.999
    xin = rng.standard_normal((4, 7)).astype(np.float32)
    run = lambda p: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"input": xin})[0]  # noqa: E731
    ref = run(src)
    assert np.abs(run(dst) - ref).max() <= 1e-4 * np.abs(ref).max()


@pytest.mark.parametrize("t", [0.0, 0.5])
def test_activation_aware_cle_stays_exact(tmp_path, t):
    import onnxruntime as ort

    from anneal.core.cle import cross_layer_equalise

    src = _model(tmp_path / "m.onnx", "relu")
    rng = np.random.default_rng(9)
    s = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    shape = [d if isinstance(d, int) else 2 for d in s.get_inputs()[0].shape]
    batches = [rng.standard_normal(shape).astype(np.float32) for _ in range(3)]
    dst = tmp_path / "cle.onnx"
    res = cross_layer_equalise(src, dst, batches=batches, t=t)
    assert res.pairs
    before = s.run(None, {name: batches[0]})[0]
    after = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(None, {name: batches[0]})[0]
    assert np.abs(after - before).max() <= 1e-4 * max(1.0, np.abs(before).max())


def test_cle_max_scale_caps_the_scales_and_is_recorded_only_when_set(tmp_path: Path):
    src = _model(tmp_path / "m.onnx", "chain")
    calib = SyntheticEvalSet(shape=(C_IN, 8, 8), n=16, batch_size=8, n_classes=4)

    def run(params, name):
        return apply_transform("quantize_static_int8", {"per_channel": False, "cle": True, **params},
                               ModelArtifact(path=src), TransformContext(workdir=tmp_path / name, calibset=calib))

    free = run({}, "free")
    assert "cle_max_scale" not in free.lineage[-1].params
    capped = run({"cle_max_scale": 1.5}, "capped")
    assert capped.lineage[-1].params["cle_max_scale"] == 1.5
    assert capped.meta["cle"]["max_scale"] <= 1.5 + 1e-3 < free.meta["cle"]["max_scale"]
    assert capped.lineage_key != free.lineage_key


@pytest.mark.parametrize("params", [
    {"cle": True, "cle_max_scale": 0.5},
    {"cle": True, "cle_max_scale": True},
    {"cle": True, "cle_max_scale": "4"},
    {"cle": True, "cle_max_scale": float("inf")},
    {"cle_max_scale": 4},  # without cle
])
def test_cle_max_scale_is_validated(tmp_path: Path, params):
    from anneal.core.transforms import TransformError

    calib = SyntheticEvalSet(shape=(C_IN, 8, 8), n=8, batch_size=8, n_classes=4)
    with pytest.raises(TransformError):
        apply_transform("quantize_static_int8", {"per_channel": False, **params},
                        ModelArtifact(path=_model(tmp_path / "m.onnx", "chain")),
                        TransformContext(workdir=tmp_path / "w", calibset=calib))
