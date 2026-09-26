"""Exact channel equalisation across gated activations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from anneal.core.artifact import ModelArtifact
from anneal.core.dataset import SyntheticEvalSet
from anneal.core.equalize import (
    GATE_MUL_PREFIX,
    choose_scales,
    equalise,
    find_sites,
    rank_sites,
    site_gain,
)
from anneal.core.transforms import TransformContext, TransformError, apply_transform

C_IN, C = 4, 6


def _imbalanced_weights(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A 1x1 conv whose output channels differ in range by ~1000x, one of them negative.

    The depthwise weights are largest on the small channels, as batch-norm folding tends to
    leave them in real networks, so their rounding error is amplified.
    """
    rng = np.random.default_rng(seed)
    wa = rng.standard_normal((C, C_IN, 1, 1)).astype(np.float32)
    gains = np.array([40.0, 1.0, 0.3, 0.05, 1.0, 0.02], dtype=np.float32)
    wa *= gains.reshape(C, 1, 1, 1)
    ba = np.array([0.0, 0.0, 0.0, -0.5, 0.0, 0.0], dtype=np.float32)  # channel 3 sits below zero
    wb = rng.standard_normal((C, 1, 3, 3)).astype(np.float32) / gains.reshape(C, 1, 1, 1)
    return wa, ba, wb


def _model(path: Path, activation: str = "silu", branch: bool = False) -> Path:
    wa, ba, wb = _imbalanced_weights()
    nodes = [helper.make_node("Conv", ["input", "wa", "ba"], ["x"], name="conv_a")]
    if activation == "silu":
        nodes += [
            helper.make_node("Sigmoid", ["x"], ["g"], name="gate"),
            helper.make_node("Mul", ["x", "g"], ["y"], name="act"),
        ]
    elif activation == "hardswish_fused":
        nodes += [helper.make_node("HardSwish", ["x"], ["y"], name="act")]
    elif activation == "hardswish":
        nodes += [
            helper.make_node("HardSigmoid", ["x"], ["g"], name="gate", alpha=1 / 6, beta=0.5),
            helper.make_node("Mul", ["x", "g"], ["y"], name="act"),
        ]
    else:
        nodes += [helper.make_node("Relu", ["x"], ["y"], name="act")]
    nodes += [
        helper.make_node("Conv", ["y", "wb"], ["z"], name="conv_b", group=C, pads=[1, 1, 1, 1])
    ]
    out = "z"
    if branch:  # x also feeds a second consumer, so rescaling it would change the model
        nodes += [helper.make_node("ReduceMean", ["x"], ["xm"], name="side", keepdims=1, axes=[2, 3])]
        nodes += [helper.make_node("Add", ["z", "xm"], ["out"], name="join")]
        out = "out"
    graph = helper.make_graph(
        nodes,
        "eq",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", C_IN, 8, 8])],
        [helper.make_tensor_value_info(out, TensorProto.FLOAT, ["N", C, 8, 8])],
        [numpy_helper.from_array(wa, "wa"), numpy_helper.from_array(ba, "ba"),
         numpy_helper.from_array(wb, "wb")],
    )
    opset = 14 if activation == "hardswish_fused" else 13
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _batches(n: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((8, C_IN, 8, 8)).astype(np.float32) for _ in range(n)]


def _run(path: Path, x: np.ndarray) -> np.ndarray:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, {"input": x})[0]


# ----- choosing scales -------------------------------------------------------


def test_small_channels_are_scaled_up_to_the_largest_one():
    lo = np.array([0.0, 0.0])
    hi = np.array([100.0, 1.0])
    s = choose_scales([(lo, hi)], slack=0.0)
    assert s == pytest.approx([1.0, 100.0])


def test_a_channel_below_zero_is_mirrored_into_the_positive_range():
    # SiLU-like: the tensor's floor is -0.278, one channel lives entirely in the negative lobe.
    lo = np.array([-0.278, -0.27])
    hi = np.array([82.0, -0.10])
    s = choose_scales([(lo, hi)], slack=0.0)
    assert s[1] < 0
    assert abs(s[1]) == pytest.approx(82.0 / 0.27, rel=1e-4)
    # Without mirroring the same channel could hardly be scaled at all.
    assert choose_scales([(lo, hi)], slack=0.0, allow_negative=False)[1] < 1.1


def test_scales_never_grow_the_top_of_the_range_and_respect_the_slack():
    rng = np.random.default_rng(0)
    lo = -rng.uniform(0, 1, 50)
    hi = rng.uniform(0, 20, 50)
    for slack in (0.0, 0.1, 0.5):
        s = choose_scales([(lo, hi)], slack=slack).astype(np.float64)
        new_lo, new_hi = np.minimum(s * lo, s * hi), np.maximum(s * lo, s * hi)
        r = hi.max() - lo.min()
        assert new_hi.max() <= hi.max() * (1 + 1e-6)
        assert new_lo.min() >= lo.min() - slack * r - 1e-6
        assert np.all(np.abs(s) >= 1.0)


def test_scales_satisfy_every_tensor_they_multiply():
    # x has room for channel 1, y does not: y's bound must win.
    x = (np.array([0.0, 0.0]), np.array([10.0, 1.0]))
    y = (np.array([0.0, 0.0]), np.array([10.0, 5.0]))
    assert choose_scales([x, y], slack=0.0)[1] == pytest.approx(2.0)


def test_dead_channels_are_left_alone():
    s = choose_scales([(np.array([0.0, 0.0]), np.array([5.0, 0.0]))], slack=0.1)
    assert s[1] == 1.0


# ----- the graph rewrite ------------------------------------------------------


@pytest.mark.parametrize("activation", ["silu", "hardswish", "hardswish_fused", "relu"])
def test_rewrite_is_found_and_exact_in_float(tmp_path: Path, activation: str):
    src = _model(tmp_path / "m.onnx", activation)
    kind = "relu" if activation == "relu" else "gated"
    assert [s.kind for s in find_sites(onnx.load(str(src)))] == [kind]

    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), slack=0.1)
    assert len(result.sites) == 1
    assert result.sites[0].levels_after > result.sites[0].levels_before

    x = _batches(1, seed=7)[0]
    before, after = _run(src, x), _run(dst, x)
    assert np.abs(after - before).max() <= 1e-4 * max(1.0, np.abs(before).max())


def test_relu_sites_are_never_mirrored(tmp_path: Path):
    result = equalise(_model(tmp_path / "m.onnx", "relu"), tmp_path / "eq.onnx", _batches())
    assert result.sites[0].channels_mirrored == 0
    assert result.gate_nodes == []


def test_gate_branch_nodes_are_reported_for_exclusion(tmp_path: Path):
    result = equalise(_model(tmp_path / "m.onnx", "silu"), tmp_path / "eq.onnx", _batches())
    assert result.gate_nodes == [f"{GATE_MUL_PREFIX}0", "gate"]
    names = {n.name for n in onnx.load(str(tmp_path / "eq.onnx")).graph.node}
    assert set(result.gate_nodes) <= names


def test_a_tensor_with_another_consumer_is_not_rewritten(tmp_path: Path):
    src = _model(tmp_path / "m.onnx", "silu", branch=True)
    assert find_sites(onnx.load(str(src))) == []
    result = equalise(src, tmp_path / "eq.onnx", _batches())
    assert result.sites == []
    x = _batches(1, seed=3)[0]
    assert np.array_equal(_run(src, x), _run(tmp_path / "eq.onnx", x))


def test_equalisation_reduces_int8_error(tmp_path: Path):
    """The point of the rewrite: the same INT8 recipe is closer to float afterwards."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    src = _model(tmp_path / "m.onnx", "silu")
    eq = tmp_path / "eq.onnx"
    equalise(src, eq, _batches(), slack=0.1)

    class Reader:
        def __init__(self):
            self.it = iter({"input": b} for b in _batches())

        def get_next(self):
            return next(self.it, None)

    def int8(path: Path, out: Path) -> Path:
        quantize_static(str(path), str(out), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=True, calibrate_method=CalibrationMethod.MinMax)
        return out

    x = _batches(1, seed=11)[0]
    ref = _run(src, x)

    def rel_err(path: Path) -> float:
        return float(np.linalg.norm(_run(path, x) - ref) / np.linalg.norm(ref))

    base_err = rel_err(int8(src, tmp_path / "q.onnx"))
    eq_err = rel_err(int8(eq, tmp_path / "qe.onnx"))
    assert eq_err < 0.5 * base_err


# ----- as part of static quantization ------------------------------------------


@pytest.fixture
def calib() -> SyntheticEvalSet:
    return SyntheticEvalSet(shape=(C_IN, 8, 8), n=32, batch_size=8, n_classes=4)


def test_static_quantization_with_equalisation_records_what_it_did(tmp_path: Path, calib):
    src = _model(tmp_path / "m.onnx", "silu")
    out = apply_transform(
        "quantize_static_int8",
        {"per_channel": True, "equalize": True, "float_gates": True},
        ModelArtifact(path=src),
        TransformContext(workdir=tmp_path / "w", calibset=calib),
    )
    assert out.meta["equalisation"]["sites"] == 1
    assert out.meta["equalisation"]["max_abs_logit_change"] < 1e-3
    assert out.lineage[-1].params["equalize"] is True
    # The gate branch stayed in float: no QuantizeLinear feeds the Sigmoid.
    q = onnx.load(str(out.path))
    producer = {o: n for n in q.graph.node for o in n.output}
    sig = next(n for n in q.graph.node if n.op_type == "Sigmoid")
    assert producer[sig.input[0]].op_type == "Mul"


def test_recipes_without_equalisation_keep_their_lineage_keys(tmp_path: Path, calib):
    src = _model(tmp_path / "m.onnx", "silu")
    out = apply_transform(
        "quantize_static_int8", {"per_channel": True}, ModelArtifact(path=src),
        TransformContext(workdir=tmp_path / "w", calibset=calib),
    )
    assert "equalize" not in out.lineage[-1].params
    assert "equalisation" not in out.meta


@pytest.mark.parametrize(
    "params",
    [
        {"per_channel": False, "equalize": True},
        {"per_channel": True, "float_gates": True},
        {"per_channel": True, "equalize": True, "equalize_slack": -1.0},
    ],
)
def test_invalid_equalisation_settings_are_rejected(tmp_path: Path, calib, params):
    with pytest.raises(TransformError):
        apply_transform(
            "quantize_static_int8", params, ModelArtifact(path=_model(tmp_path / "m.onnx")),
            TransformContext(workdir=tmp_path / "w", calibset=calib),
        )


def test_unnamed_gate_nodes_are_named_so_they_can_be_excluded(tmp_path: Path):
    src = _model(tmp_path / "m.onnx", "hardswish")
    m = onnx.load(str(src))
    for n in m.graph.node:
        if n.op_type == "HardSigmoid":
            n.name = ""  # as torchvision's MobileNetV3 export leaves them
    onnx.save(m, str(src))
    result = equalise(src, tmp_path / "eq.onnx", _batches())
    assert len(result.sites) == 1
    assert all(result.gate_nodes)
    names = {n.name for n in onnx.load(str(tmp_path / "eq.onnx")).graph.node}
    assert set(result.gate_nodes) <= names


# ----- selective equalisation ---------------------------------------------------

#: Per-channel gains of each block's producer conv: heavily, not, and moderately imbalanced.
_BLOCK_GAINS = [
    [40.0, 1.0, 0.3, 0.05, 1.0, 0.02],
    [1.0, 1.1, 0.9, 1.0, 1.2, 0.8],
    [5.0, 1.0, 0.2, 1.0, 0.5, 0.1],
]


def _chain(path: Path) -> Path:
    """Three Conv -> SiLU -> depthwise blocks in a row (sites conv_a0, conv_a1, conv_a2)."""
    rng = np.random.default_rng(0)
    nodes, inits, t = [], [], "input"
    for i, gains in enumerate(_BLOCK_GAINS):
        g = np.array(gains, dtype=np.float32).reshape(C, 1, 1, 1)
        wa = rng.standard_normal((C, C_IN if i == 0 else C, 1, 1)).astype(np.float32) * g
        wb = rng.standard_normal((C, 1, 3, 3)).astype(np.float32) / g
        inits += [numpy_helper.from_array(wa, f"wa{i}"), numpy_helper.from_array(wb, f"wb{i}")]
        nodes += [
            helper.make_node("Conv", [t, f"wa{i}"], [f"x{i}"], name=f"conv_a{i}"),
            helper.make_node("Sigmoid", [f"x{i}"], [f"g{i}"], name=f"gate{i}"),
            helper.make_node("Mul", [f"x{i}", f"g{i}"], [f"y{i}"], name=f"act{i}"),
            helper.make_node("Conv", [f"y{i}", f"wb{i}"], [f"z{i}"], name=f"conv_b{i}", group=C,
                             pads=[1, 1, 1, 1]),
        ]
        t = f"z{i}"
    graph = helper.make_graph(
        nodes, "chain",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", C_IN, 8, 8])],
        [helper.make_tensor_value_info(t, TensorProto.FLOAT, ["N", C, 8, 8])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _gate_muls(path: Path) -> list[str]:
    return [n.name for n in onnx.load(str(path)).graph.node if n.name.startswith(GATE_MUL_PREFIX)]


def test_site_gain_counts_rescued_channels_not_already_fine_ones():
    lo, hi = np.array([0.0, 0.0]), np.array([100.0, 0.2])  # channel 1 spans half a step
    rescued = site_gain(lo, hi, np.array([1.0, 500.0]))
    assert rescued == pytest.approx(1.0, abs=0.01)  # one channel's worth of signal recovered
    fine = site_gain(np.array([0.0, 0.0]), np.array([100.0, 20.0]), np.array([1.0, 5.0]))
    assert 0 < fine < 0.01
    assert site_gain(lo, hi, np.ones(2)) == 0.0


def test_ranking_covers_every_site_best_first(tmp_path: Path):
    src = _chain(tmp_path / "m.onnx")
    ranking = rank_sites(src, _batches())
    assert sorted(r.site for r in ranking) == ["conv_a0", "conv_a1", "conv_a2"]
    gains = [r.gain for r in ranking]
    assert gains == sorted(gains, reverse=True)
    assert ranking[0].site == "conv_a0"  # the heavily imbalanced block
    assert ranking[-1].site == "conv_a1"  # the balanced one has almost nothing to gain
    # equalise measures the same thing.
    result = equalise(src, tmp_path / "eq.onnx", _batches())
    assert result.ranking == ranking
    assert {s.producer: s.predicted_gain for s in result.sites} == dict(ranking)


def test_only_the_chosen_sites_are_rewritten_and_the_model_stays_exact(tmp_path: Path):
    src = _chain(tmp_path / "m.onnx")
    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), sites=["conv_a2"])
    assert [s.producer for s in result.sites] == ["conv_a2"]
    assert len(result.ranking) == 3
    assert _gate_muls(dst) == [f"{GATE_MUL_PREFIX}2"]
    assert result.gate_nodes == [f"{GATE_MUL_PREFIX}2", "gate2"]
    # Untouched sites keep their weights.
    before = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(src)).graph.initializer}
    after = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(dst)).graph.initializer}
    for name in ("wa0", "wb0", "wa1", "wb1"):
        assert np.array_equal(before[name], after[name])
    assert not np.array_equal(before["wa2"], after["wa2"])
    x = _batches(1, seed=7)[0]
    ref, out = _run(src, x), _run(dst, x)
    assert np.abs(out - ref).max() <= 1e-4 * max(1.0, np.abs(ref).max())


def test_top_k_takes_the_best_ranked_sites(tmp_path: Path):
    src = _chain(tmp_path / "m.onnx")
    ranking = rank_sites(src, _batches())
    result = equalise(src, tmp_path / "eq.onnx", _batches(), top_k=2)
    assert {s.producer for s in result.sites} == {r.site for r in ranking[:2]}
    assert len(_gate_muls(tmp_path / "eq.onnx")) == 2


def test_top_k_zero_is_no_equalisation_and_top_k_n_is_all(tmp_path: Path):
    src = _chain(tmp_path / "m.onnx")
    none = equalise(src, tmp_path / "none.onnx", _batches(), top_k=0)
    assert none.sites == [] and none.gate_nodes == [] and len(none.ranking) == 3
    x = _batches(1, seed=5)[0]
    assert np.array_equal(_run(src, x), _run(tmp_path / "none.onnx", x))

    full = equalise(src, tmp_path / "full.onnx", _batches())
    for k in (3, 10):
        some = equalise(src, tmp_path / f"k{k}.onnx", _batches(), top_k=k)
        assert [s.to_dict() for s in some.sites] == [s.to_dict() for s in full.sites]
        assert some.gate_nodes == full.gate_nodes
        assert np.array_equal(_run(tmp_path / f"k{k}.onnx", x), _run(tmp_path / "full.onnx", x))


@pytest.mark.parametrize(
    "kwargs",
    [{"sites": ["conv_b0"]}, {"sites": ["nope"]}, {"top_k": -1}, {"top_k": True},
     {"sites": ["conv_a0"], "top_k": 1}],
)
def test_invalid_site_selection_is_rejected(tmp_path: Path, kwargs):
    with pytest.raises(ValueError):
        equalise(_chain(tmp_path / "m.onnx"), tmp_path / "eq.onnx", _batches(), **kwargs)


def _static(tmp_path: Path, calib, src: Path, name: str, **params):
    return apply_transform(
        "quantize_static_int8", {"per_channel": True, **params}, ModelArtifact(path=src),
        TransformContext(workdir=tmp_path / name, calibset=calib),
    )


def test_static_quantization_top_k_records_the_chosen_sites(tmp_path: Path, calib):
    src = _chain(tmp_path / "m.onnx")
    out = _static(tmp_path, calib, src, "k1", equalize=True, equalize_top_k=1)
    assert out.lineage[-1].params["equalize_top_k"] == 1
    assert out.meta["equalised_site_ids"] == [out.meta["equalisation_ranking"][0]["site"]]
    assert [r["site"] for r in out.meta["equalisation_ranking"]] == [
        r.site for r in rank_sites(src, calib.calibration_batches(out.meta["calib_samples"]))
    ]
    assert out.meta["equalisation"]["sites"] == 1
    assert out.meta["equalisation"]["candidates"] == 3
    assert out.meta["equalisation"]["max_abs_logit_change"] < 1e-3


def test_static_quantization_top_k_limits_match_off_and_full(tmp_path: Path, calib):
    src = _chain(tmp_path / "m.onnx")
    x = _batches(1, seed=9)[0]
    off = _run(_static(tmp_path, calib, src, "off").path, x)
    k0 = _static(tmp_path, calib, src, "k0", equalize=True, equalize_top_k=0)
    assert k0.meta["equalised_site_ids"] == []
    assert np.array_equal(_run(k0.path, x), off)

    full = _static(tmp_path, calib, src, "full", equalize=True)
    kn = _static(tmp_path, calib, src, "kn", equalize=True, equalize_top_k=3)
    assert kn.meta["equalised_sites"] == full.meta["equalised_sites"]
    assert np.array_equal(_run(kn.path, x), _run(full.path, x))
    assert "equalize_top_k" not in full.lineage[-1].params


@pytest.mark.parametrize(
    "params",
    [
        {"per_channel": True, "equalize_top_k": 2},
        {"per_channel": True, "equalize": True, "equalize_top_k": -1},
        {"per_channel": True, "equalize": True, "equalize_top_k": 1.5},
        {"per_channel": True, "equalize": True, "equalize_top_k": True},
    ],
)
def test_invalid_top_k_settings_are_rejected(tmp_path: Path, calib, params):
    with pytest.raises(TransformError):
        apply_transform(
            "quantize_static_int8", params, ModelArtifact(path=_chain(tmp_path / "m.onnx")),
            TransformContext(workdir=tmp_path / "w", calibset=calib),
        )


def test_float_stem_keeps_the_first_conv_and_its_activation_out_of_quantization(tmp_path: Path, calib):
    from anneal.core.transforms import stem_nodes

    src = _model(tmp_path / "m.onnx", "silu")
    assert stem_nodes(src) == ["conv_a", "gate", "act"]
    out = apply_transform(
        "quantize_static_int8",
        {"per_channel": True, "equalize": True, "float_stem": True,
         "calibrate_method": "percentile_asym"},
        ModelArtifact(path=src), TransformContext(workdir=tmp_path / "w", calibset=calib),
    )
    q = onnx.load(str(out.path))
    producer = {o: n for n in q.graph.node for o in n.output}
    conv_a = next(n for n in q.graph.node if n.name == "conv_a")
    # Its weight arrives as a float initializer, not through a DequantizeLinear.
    assert conv_a.input[1] not in producer
    assert out.lineage[-1].params["float_stem"] is True
    assert out.lineage[-1].params["calibrate_method"] == "percentile_asym"


def test_compute_only_quantization_leaves_other_ops_in_float(tmp_path: Path, calib):
    src = _model(tmp_path / "m.onnx", "silu")
    out = apply_transform(
        "quantize_static_int8", {"per_channel": True, "quantize_ops": "compute"},
        ModelArtifact(path=src), TransformContext(workdir=tmp_path / "w", calibset=calib),
    )
    q = onnx.load(str(out.path))
    sig = next(n for n in q.graph.node if n.op_type == "Sigmoid")
    # The Sigmoid's output feeds the Mul directly: element-wise ops get no Q/DQ of their own.
    consumers = [n.op_type for n in q.graph.node if sig.output[0] in n.input]
    assert consumers == ["Mul"]
    assert out.lineage[-1].params["quantize_ops"] == "compute"
    with pytest.raises(TransformError):
        apply_transform("quantize_static_int8", {"quantize_ops": "everything"}, ModelArtifact(path=src),
                        TransformContext(workdir=tmp_path / "w2", calibset=calib))


def test_min_gain_keeps_only_sites_worth_their_cost(tmp_path: Path):
    src = _chain(tmp_path / "m.onnx")
    ranking = rank_sites(src, _batches())
    cut = (ranking[0].gain + ranking[1].gain) / 2  # between the best site and the rest
    result = equalise(src, tmp_path / "eq.onnx", _batches(), min_gain=cut)
    assert [s.producer for s in result.sites] == [ranking[0].site]
    none = equalise(src, tmp_path / "none.onnx", _batches(), min_gain=ranking[0].gain * 10)
    assert none.sites == [] and len(none.ranking) == 3
    full = equalise(src, tmp_path / "full.onnx", _batches(), min_gain=0.0)
    assert len(full.sites) == 3


@pytest.mark.parametrize("kwargs", [{"min_gain": -1.0}, {"min_gain": True}, {"min_gain": 1.0, "top_k": 1}])
def test_invalid_min_gain_is_rejected(tmp_path: Path, kwargs):
    with pytest.raises(ValueError):
        equalise(_chain(tmp_path / "m.onnx"), tmp_path / "eq.onnx", _batches(), **kwargs)


@pytest.mark.parametrize(
    "params",
    [
        {"per_channel": True, "equalize_min_gain": 1.0},
        {"per_channel": True, "equalize": True, "equalize_min_gain": -1},
        {"per_channel": True, "equalize": True, "equalize_min_gain": 1.0, "equalize_top_k": 1},
    ],
)
def test_invalid_min_gain_settings_are_rejected(tmp_path: Path, calib, params):
    with pytest.raises(TransformError):
        apply_transform(
            "quantize_static_int8", params, ModelArtifact(path=_chain(tmp_path / "m.onnx")),
            TransformContext(workdir=tmp_path / "w", calibset=calib),
        )


def test_static_quantization_min_gain_records_the_chosen_sites(tmp_path: Path, calib):
    src = _chain(tmp_path / "m.onnx")
    art = _static(tmp_path, calib, src, "mg", equalize=True, equalize_min_gain=1e9)
    assert art.meta["equalised_site_ids"] == [] and art.meta["equalisation"]["candidates"] == 3
    assert art.lineage[-1].params["equalize_min_gain"] == 1e9
