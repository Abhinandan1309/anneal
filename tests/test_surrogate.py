"""The HardSigmoid-sum sigmoid surrogate for AMD NPUs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from anneal.core.artifact import ModelArtifact
from anneal.core.dataset import SyntheticEvalSet
from anneal.core.surrogate import (
    ALPHA,
    BETA,
    fit_surrogate,
    replace_sigmoids,
    sigmoid_gates,
    surrogate,
)
from anneal.core.transforms import TransformContext, TransformError, apply_transform

from test_equalize import C_IN, _model


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture
def calib() -> SyntheticEvalSet:
    return SyntheticEvalSet(shape=(C_IN, 8, 8), n=32, batch_size=8, n_classes=4)


def _run(path: Path, x: np.ndarray) -> np.ndarray:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})[0]


# ----- the fit -------------------------------------------------------------------


@pytest.mark.parametrize("silu", [True, False])
def test_the_fitted_surrogate_has_exact_asymptotes_and_a_small_error(silu):
    xs = np.random.default_rng(0).normal(-1.0, 3.0, 50_000)
    w, k, b = fit_surrogate(xs, silu, 3)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert (w > 0).all() and (k > 0).all()
    # Exactly 0 and 1 beyond every term's knee, as sigmoid tends to.
    assert surrogate(np.array([-1e4]), w, k, b)[0] == 0.0
    assert surrogate(np.array([1e4]), w, k, b)[0] == pytest.approx(1.0, abs=1e-12)
    grid = np.linspace(np.percentile(xs, 0.5), np.percentile(xs, 99.5), 2001)
    err = np.abs(surrogate(grid, w, k, b) - _sig(grid))
    assert err.max() < 0.03
    if silu:
        assert (np.abs(grid) * err).max() < 0.1


def test_more_terms_fit_better_than_one_and_the_fit_is_deterministic():
    xs = np.random.default_rng(1).normal(0.0, 4.0, 30_000)
    loss = []
    for K in (1, 3):
        w, k, b = fit_surrogate(xs, True, K)
        loss.append(np.mean((xs * (surrogate(xs, w, k, b) - _sig(xs))) ** 2))
    assert loss[1] < 0.2 * loss[0]
    a, b_ = fit_surrogate(xs, False, 2, seed=3), fit_surrogate(xs, False, 2, seed=3)
    for p, q in zip(a, b_):
        np.testing.assert_array_equal(p, q)


def test_fitting_without_samples_is_refused():
    with pytest.raises(ValueError):
        fit_surrogate(np.array([]), False, 3)
    with pytest.raises(ValueError):
        fit_surrogate(np.zeros(10), False, 0)


# ----- the gates -----------------------------------------------------------------


def _se_model(path: Path) -> Path:
    """Squeeze-excite: the Sigmoid's output scales the feature map, not its own input."""
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((C_IN, C_IN, 1, 1)).astype(np.float32)
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"], name="squeeze"),
        helper.make_node("Conv", ["pooled", "w1"], ["logit"], name="excite"),
        helper.make_node("Sigmoid", ["logit"], ["g"], name="se_gate"),
        helper.make_node("Mul", ["input", "g"], ["out"], name="scale"),
    ]
    graph = helper.make_graph(
        nodes, "se",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", C_IN, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", C_IN, 8, 8])],
        [numpy_helper.from_array(w1, "w1")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_a_silu_is_flagged_and_a_squeeze_excite_gate_is_not(tmp_path: Path):
    _model(tmp_path / "silu.onnx", "silu")
    (silu,) = sigmoid_gates(onnx.load(str(tmp_path / "silu.onnx")))
    assert silu.silu and silu.input == "x" and silu.node == "gate"
    (se,) = sigmoid_gates(onnx.load(str(_se_model(tmp_path / "se.onnx"))))
    assert not se.silu and se.input == "logit"


def test_an_equalised_silu_is_still_a_silu(tmp_path: Path, calib):
    """Equalisation feeds the gate x'/s while the Mul multiplies by x'."""
    from anneal.core.equalize import equalise

    _model(tmp_path / "m.onnx", "silu")
    equalise(tmp_path / "m.onnx", tmp_path / "eq.onnx", calib.calibration_batches(32))
    (gate,) = sigmoid_gates(onnx.load(str(tmp_path / "eq.onnx")))
    assert gate.input != "x" and gate.silu


# ----- the rewrite ---------------------------------------------------------------


def test_replacing_the_sigmoids_keeps_the_outputs_close(tmp_path: Path, calib):
    src = tmp_path / "m.onnx"
    _model(src, "silu")
    report = replace_sigmoids(src, tmp_path / "s.onnx", calib.calibration_batches(32), k_terms=3)
    model = onnx.load(str(tmp_path / "s.onnx"))
    ops = [n.op_type for n in model.graph.node]
    assert "Sigmoid" not in ops and ops.count("HardSigmoid") == 3
    for n in model.graph.node:
        if n.op_type == "HardSigmoid":  # the DPU's fixed gate, nothing folded into it
            attrs = {a.name: a.f for a in n.attribute}
            assert attrs["alpha"] == pytest.approx(ALPHA) and attrs["beta"] == pytest.approx(BETA)
    assert report["gates"] == 1 and report["silu_gates"] == 1
    assert report["per_gate"][0]["max_abs_error"] < 0.03

    x = np.random.default_rng(5).standard_normal((4, C_IN, 8, 8)).astype(np.float32)
    ref = _run(src, x)
    change = np.abs(_run(tmp_path / "s.onnx", x) - ref).max()
    # AMD's plain Sigmoid -> HardSigmoid swap, for scale.
    plain = onnx.load(str(src))
    for n in plain.graph.node:
        if n.op_type == "Sigmoid":
            n.op_type = "HardSigmoid"
            n.attribute.extend([helper.make_attribute("alpha", ALPHA), helper.make_attribute("beta", BETA)])
    onnx.save(plain, str(tmp_path / "plain.onnx"))
    plain_change = np.abs(_run(tmp_path / "plain.onnx", x) - ref).max()
    assert change < 0.03 * np.abs(ref).max()
    assert change < 0.25 * plain_change


def test_one_term_needs_no_weight_and_squeeze_excite_gates_are_replaced_too(tmp_path: Path, calib):
    src = _se_model(tmp_path / "se.onnx")
    report = replace_sigmoids(src, tmp_path / "s.onnx", calib.calibration_batches(32), k_terms=1)
    ops = [n.op_type for n in onnx.load(str(tmp_path / "s.onnx")).graph.node]
    assert ops == ["GlobalAveragePool", "Conv", "Mul", "Add", "HardSigmoid", "Mul"]
    assert report["per_gate"][0]["silu"] is False
    assert report["max_abs_output_change"] < 0.1


# ----- as part of static quantization --------------------------------------------


def test_the_transform_option_runs_and_records_its_lineage(tmp_path: Path, calib):
    src = _model(tmp_path / "m.onnx", "silu")
    out = apply_transform(
        "quantize_static_int8",
        {"per_channel": True, "equalize": True, "float_gates": True, "sigmoid_surrogate": 3},
        ModelArtifact(path=src),
        TransformContext(workdir=tmp_path / "w", calibset=calib),
    )
    assert out.lineage[-1].params["sigmoid_surrogate"] == 3
    assert out.meta["sigmoid_surrogate"]["gates"] == 1
    assert out.meta["sigmoid_surrogate"]["silu_gates"] == 1  # seen through equalisation's Mul
    assert out.meta["equalisation"]["sites"] == 1
    q = onnx.load(str(out.path))
    assert "Sigmoid" not in {n.op_type for n in q.graph.node}
    # float_gates carried over to the surrogate: no QuantizeLinear feeds its HardSigmoids.
    producer = {o: n for n in q.graph.node for o in n.output}
    for n in q.graph.node:
        if n.op_type == "HardSigmoid":
            assert producer[n.input[0]].op_type == "Add"


def test_recipes_without_the_surrogate_keep_their_lineage_keys(tmp_path: Path, calib):
    src = _model(tmp_path / "m.onnx", "silu")
    out = apply_transform(
        "quantize_static_int8", {"per_channel": True, "sigmoid_surrogate": 0},
        ModelArtifact(path=src), TransformContext(workdir=tmp_path / "w", calibset=calib),
    )
    assert "sigmoid_surrogate" not in out.lineage[-1].params
    assert "sigmoid_surrogate" not in out.meta
    with pytest.raises(TransformError):
        apply_transform("quantize_static_int8", {"sigmoid_surrogate": True}, ModelArtifact(path=src),
                        TransformContext(workdir=tmp_path / "w2", calibset=calib))


# ----- folding into equalisation's gate conv ---------------------------------------


def _as_gate_mul(src: Path, dst: Path) -> Path:
    """The same equalised model with each gate conv written as the equivalent Mul(x', 1/s)."""
    from anneal.core.equalize import GATE_CONV_PREFIX

    m = onnx.load(str(src))
    invs = set()
    for n in m.graph.node:
        if n.name.startswith(GATE_CONV_PREFIX):
            invs.add(n.input[1])
            n.op_type = "Mul"
            del n.input[2]
            del n.attribute[:]
    for i in m.graph.initializer:
        if i.name in invs:
            i.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(i).reshape(1, -1, 1, 1), i.name))
    onnx.save(m, str(dst))
    return dst


def test_the_surrogate_folds_into_a_gate_conv(tmp_path: Path, calib):
    from anneal.core.equalize import GATE_CONV_PREFIX, equalise

    src = _model(tmp_path / "m.onnx", "silu")
    eq = tmp_path / "eq.onnx"
    equalise(src, eq, calib.calibration_batches(32), gate_conv=True)
    (gate,) = sigmoid_gates(onnx.load(str(eq)))
    assert gate.silu and gate.conv is not None  # a SiLU through the conv, and foldable

    report = replace_sigmoids(eq, tmp_path / "s.onnx", calib.calibration_batches(32), k_terms=3)
    [g] = report["per_gate"]
    assert g["folded"].startswith(GATE_CONV_PREFIX)
    model = onnx.load(str(tmp_path / "s.onnx"))
    names = {n.name for n in model.graph.node}
    assert g["folded"] not in names and set(g["nodes"]) <= names
    producer = {o: n for n in model.graph.node for o in n.output}
    hs = [n for n in model.graph.node if n.op_type == "HardSigmoid"]
    assert len(hs) == 3
    for n in hs:  # Conv -> HardSigmoid (the DPU's fixed gate), the conv reading the balanced x'
        attrs = {a.name: a.f for a in n.attribute}
        assert attrs["alpha"] == pytest.approx(ALPHA) and attrs["beta"] == pytest.approx(BETA)
        conv = producer[n.input[0]]
        assert conv.op_type == "Conv" and conv.input[0] == "x"
    # Only the w_i Muls, the sum and the SiLU's own Mul remain: no Mul(k)/Add(b) before a gate.
    ops = [n.op_type for n in model.graph.node]
    assert ops.count("Mul") == 3 + 1 and ops.count("Add") == 2
    # The folded conv's 1/s and zero bias went with it.
    used = {i for n in model.graph.node for i in n.input}
    assert all(i.name in used for i in model.graph.initializer)

    # Folding is exact against the unfolded surrogate of the same fit ...
    replace_sigmoids(_as_gate_mul(eq, tmp_path / "eq_mul.onnx"), tmp_path / "s_mul.onnx",
                     calib.calibration_batches(32), k_terms=3)
    x = np.random.default_rng(5).standard_normal((4, C_IN, 8, 8)).astype(np.float32)
    ref = _run(src, x)
    folded, mul = _run(tmp_path / "s.onnx", x), _run(tmp_path / "s_mul.onnx", x)
    assert np.abs(folded - mul).max() <= 1e-4 * max(1.0, np.abs(ref).max())
    # ... and as close to the original model as the plain surrogate is.
    assert np.abs(folded - ref).max() < 0.03 * np.abs(ref).max()


def test_a_gate_conv_with_another_consumer_is_not_folded(tmp_path: Path, calib):
    from anneal.core.equalize import GATE_CONV_PREFIX, equalise

    src = _model(tmp_path / "m.onnx", "silu")
    eq = tmp_path / "eq.onnx"
    equalise(src, eq, calib.calibration_batches(32), gate_conv=True)
    m = onnx.load(str(eq))
    conv = next(n for n in m.graph.node if n.name.startswith(GATE_CONV_PREFIX))
    m.graph.output.append(helper.make_tensor_value_info(conv.output[0], TensorProto.FLOAT, ["N", "C", "H", "W"]))
    onnx.save(m, str(tmp_path / "side.onnx"))
    (gate,) = sigmoid_gates(onnx.load(str(tmp_path / "side.onnx")))
    assert gate.conv is None and gate.silu
    report = replace_sigmoids(tmp_path / "side.onnx", tmp_path / "s.onnx", calib.calibration_batches(32))
    assert report["per_gate"][0]["folded"] is None
    assert conv.name in {n.name for n in onnx.load(str(tmp_path / "s.onnx")).graph.node}


def test_the_transform_folds_the_surrogate_into_gate_convs(tmp_path: Path, calib):
    src = _model(tmp_path / "m.onnx", "silu")
    out = apply_transform(
        "quantize_static_int8",
        {"per_channel": True, "equalize": True, "equalize_gate_conv": True, "float_gates": True,
         "sigmoid_surrogate": 3},
        ModelArtifact(path=src),
        TransformContext(workdir=tmp_path / "w", calibset=calib),
    )
    params = out.lineage[-1].params
    assert params["equalize_gate_conv"] is True and params["sigmoid_surrogate"] == 3
    assert out.meta["sigmoid_surrogate"]["silu_gates"] == 1
    assert out.meta["sigmoid_surrogate_gates"][0]["folded"]
    q = onnx.load(str(out.path))
    assert "Sigmoid" not in {n.op_type for n in q.graph.node}
