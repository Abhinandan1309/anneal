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


def test_min_gain_is_all_or_nothing_on_the_models_total_gain(tmp_path: Path):
    src = _chain(tmp_path / "m.onnx")
    total = sum(r.gain for r in rank_sites(src, _batches()))
    all_ = equalise(src, tmp_path / "all.onnx", _batches(), min_gain=total * 0.99)
    assert len(all_.sites) == 3  # a model worth equalising gets every site, not the top ones
    none = equalise(src, tmp_path / "none.onnx", _batches(), min_gain=total * 1.01)
    assert none.sites == [] and len(none.ranking) == 3
    x = _batches(1, seed=5)[0]
    assert np.array_equal(_run(src, x), _run(tmp_path / "none.onnx", x))


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


def test_strided_calibration_is_recorded_and_close_to_one_shot(tmp_path: Path, calib):
    src = _chain(tmp_path / "m.onnx")
    one = _static(tmp_path, calib, src, "one", calibrate_method="percentile")
    strided = _static(tmp_path, calib, src, "strided", calibrate_method="percentile", calib_stride=1)
    assert strided.lineage[-1].params["calib_stride"] == 1
    assert "calib_stride" not in one.lineage[-1].params
    x = _batches(1, seed=3)[0]
    a, b = _run(Path(one.path), x), _run(Path(strided.path), x)
    assert np.abs(a - b).max() <= 0.1 * max(1.0, np.abs(a).max())
    with pytest.raises(TransformError):
        _static(tmp_path, calib, src, "bad", calib_stride=0)


def test_stem_int16_puts_one_16_bit_quantizer_on_the_stem_output(tmp_path: Path, calib):
    src = _chain(tmp_path / "m.onnx")
    art = _static(tmp_path, calib, src, "s16", equalize=True, stem_int16=True)
    assert art.lineage[-1].params["stem_int16"] is True
    m = onnx.load(str(art.path))
    zero_points = {i.name: i.data_type for i in m.graph.initializer}
    sixteen = [n for n in m.graph.node if n.op_type == "QuantizeLinear" and len(n.input) > 2
               and zero_points.get(n.input[2]) == onnx.TensorProto.UINT16]
    assert len(sixteen) == 1
    x = _batches(1, seed=2)[0]
    assert np.isfinite(_run(Path(art.path), x)).all()


def test_int16_tensors_are_quantized_to_16_bits_and_unknown_names_rejected(tmp_path: Path, calib):
    src = _chain(tmp_path / "m.onnx")
    y_tensors = [n.output[0] for n in onnx.load(str(src)).graph.node if n.op_type == "Mul"][:2]
    art = _static(tmp_path, calib, src, "t16", int16_tensors=y_tensors)
    m = onnx.load(str(art.path))
    types = {i.name: i.data_type for i in m.graph.initializer}
    sixteen = [n for n in m.graph.node if n.op_type == "QuantizeLinear" and len(n.input) > 2
               and types.get(n.input[2]) == onnx.TensorProto.UINT16]
    assert len(sixteen) == 2
    with pytest.raises(TransformError):
        _static(tmp_path, calib, src, "bad16", int16_tensors=["no_such_tensor"])


# ----- residual sites -----------------------------------------------------------


def _residual_model(path: Path, consumer_group: int = 1) -> Path:
    """EfficientViT's stem: y = hswish(A(input)) feeds a depthwise conv and a residual Add."""
    wa, ba, wb = _imbalanced_weights()
    rng = np.random.default_rng(3)
    wp = rng.standard_normal((C, C, 1, 1)).astype(np.float32) * 0.3
    bp = rng.standard_normal(C).astype(np.float32)
    wc = (rng.standard_normal((5, C, 1, 1)) if consumer_group == 1 else rng.standard_normal((C, 1, 3, 3))).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["input", "wa", "ba"], ["x"], name="conv_a"),
        helper.make_node("HardSwish", ["x"], ["y"], name="act"),
        helper.make_node("Conv", ["y", "wb"], ["d"], name="conv_b", group=C, pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["d"], ["dr"], name="relu_b"),
        helper.make_node("Conv", ["dr", "wp", "bp"], ["p"], name="conv_p"),
        helper.make_node("Add", ["p", "y"], ["z"], name="res"),
        helper.make_node("Conv", ["z", "wc"], ["out"], name="conv_c",
                         **({} if consumer_group == 1 else {"group": C, "pads": [1, 1, 1, 1]})),
    ]
    c_out = 5 if consumer_group == 1 else C
    graph = helper.make_graph(
        nodes, "res",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", C_IN, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", c_out, 8, 8])],
        [numpy_helper.from_array(a, n) for a, n in ((wa, "wa"), (ba, "ba"), (wb, "wb"), (wp, "wp"), (bp, "bp"), (wc, "wc"))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


@pytest.mark.parametrize("consumer_group", [1, C])
def test_residual_sites_are_opt_in_and_exact_in_float(tmp_path: Path, consumer_group: int):
    src = _residual_model(tmp_path / "m.onnx", consumer_group)
    assert find_sites(onnx.load(str(src))) == []
    [site] = find_sites(onnx.load(str(src)), residual=True)
    assert (site.conv_p.name, site.z, [c.name for c in site.consumers_z]) == ("conv_p", "z", ["conv_c"])

    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), residual=True)
    assert [s.kind for s in result.sites] == ["gated-residual"]
    assert result.summary()["by_kind"]["gated-residual"] == 1
    x = _batches(1, seed=7)[0]
    before, after = _run(src, x), _run(dst, x)
    assert np.abs(after - before).max() <= 1e-4 * max(1.0, np.abs(before).max())


def test_residual_site_needs_a_conv_on_the_other_branch(tmp_path: Path):
    src = _residual_model(tmp_path / "m.onnx")
    m = onnx.load(str(src))
    m.graph.node.remove(next(n for n in m.graph.node if n.name == "conv_p"))
    next(n for n in m.graph.node if n.name == "res").input[0] = "dr"
    assert find_sites(m, residual=True) == []


def test_static_quantization_records_residual_equalisation(tmp_path: Path, calib):
    src = _residual_model(tmp_path / "m.onnx")
    out = _static(tmp_path, calib, src, "r", per_channel=True, equalize=True, equalize_residual=True)
    assert out.lineage[-1].params["equalize_residual"] is True
    assert out.meta["equalisation"]["by_kind"]["gated-residual"] == 1
    with pytest.raises(TransformError):
        _static(tmp_path, calib, src, "bad", per_channel=True, equalize_residual=True)


def test_a_weight_tied_to_another_node_is_left_alone(tmp_path: Path):
    src = _model(tmp_path / "m.onnx", "silu")
    m = onnx.load(str(src))
    m.graph.node.append(helper.make_node("Conv", ["input", "wa"], ["x2"], name="twin"))
    # conv_a's weight is tied to a second conv: rescaling it for the site would change that one.
    m.graph.output.append(helper.make_tensor_value_info("x2", TensorProto.FLOAT, ["N", C, 8, 8]))
    onnx.save(m, str(tmp_path / "tied.onnx"))
    result = equalise(tmp_path / "tied.onnx", tmp_path / "eq.onnx", _batches())
    assert result.sites == []


def test_residual_consumer_that_starts_the_next_site_keeps_both_sites(tmp_path: Path):
    """The stem's residual consumer is the next block's expand conv: both sites are rewritten."""
    src = _residual_model(tmp_path / "m.onnx")
    m = onnx.load(str(src))
    rng = np.random.default_rng(5)
    m.graph.output.pop()
    m.graph.node.extend([
        helper.make_node("HardSwish", ["out"], ["y2"], name="act2"),
        helper.make_node("Conv", ["y2", "wd"], ["out2"], name="conv_d", group=5, pads=[1, 1, 1, 1]),
    ])
    m.graph.initializer.append(numpy_helper.from_array(rng.standard_normal((5, 1, 3, 3)).astype(np.float32), "wd"))
    m.graph.output.append(helper.make_tensor_value_info("out2", TensorProto.FLOAT, ["N", 5, 8, 8]))
    onnx.save(m, str(tmp_path / "two.onnx"))
    dst = tmp_path / "eq.onnx"
    result = equalise(tmp_path / "two.onnx", dst, _batches(), residual=True)
    assert sorted(s.kind for s in result.sites) == ["gated", "gated-residual"]
    x = _batches(1, seed=7)[0]
    before, after = _run(tmp_path / "two.onnx", x), _run(dst, x)
    assert np.abs(after - before).max() <= 1e-4 * max(1.0, np.abs(before).max())


@pytest.mark.parametrize("mix", [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)])
def test_mixed_scales_stay_exact_in_float(tmp_path: Path, mix):
    src = _model(tmp_path / "m.onnx", "silu")
    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), mix=mix)
    assert len(result.sites) == 1
    x = _batches(1, seed=7)[0]
    before, after = _run(src, x), _run(dst, x)
    assert np.abs(after - before).max() <= 1e-4 * max(1.0, np.abs(before).max())


def test_mix_one_zero_is_the_default_rewrite(tmp_path: Path):
    src = _model(tmp_path / "m.onnx", "silu")
    a = equalise(src, tmp_path / "a.onnx", _batches())
    b = equalise(src, tmp_path / "b.onnx", _batches(), mix=(1.0, 0.0))
    assert a.sites[0].scale_max == pytest.approx(b.sites[0].scale_max)


def test_min_damage_switch_measures_and_decides(tmp_path: Path, calib):
    src = _model(tmp_path / "m.onnx", "silu")
    on = _static(tmp_path, calib, src, "on", equalize=True, equalize_min_damage=0.0)
    assert 0.0 <= on.meta["joint_damage"] <= 1.0
    assert on.meta["equalisation"]["sites"] == 1
    assert on.lineage[-1].params["equalize_min_damage"] == 0.0
    off = _static(tmp_path, calib, src, "off", equalize=True, equalize_min_damage=1.0)
    if off.meta["joint_damage"] < 1.0:
        assert "equalisation" not in off.meta and "equalisation_skipped" in off.meta


@pytest.mark.parametrize("params", [{"equalize_min_damage": 0.3},  # without equalize
                                    {"equalize": True, "equalize_min_damage": 1.5},
                                    {"equalize": True, "equalize_min_damage": 0.3, "equalize_top_k": 2}])
def test_invalid_min_damage_settings_are_rejected(tmp_path: Path, calib, params):
    with pytest.raises(TransformError):
        _static(tmp_path, calib, _model(tmp_path / "m.onnx", "silu"), "bad", **params)


def test_mean_minmax_ranges_sit_inside_the_extremes_and_quantize(tmp_path: Path, calib):
    from anneal.core.transforms import mean_minmax_ranges

    src = _model(tmp_path / "m.onnx", "silu")
    ranges = mean_minmax_ranges(src, calib.calibration_batches(16))
    assert {"x", "y", "z", "input"} <= set(ranges)
    x = np.concatenate(list(calib.calibration_batches(16)))
    s = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    z = s.run(None, {"input": x})[0]
    lo, hi = ranges["z"]
    assert min(z.min(), 0) <= lo <= 0 <= hi <= max(z.max(), 0)  # mean of per-image extremes
    out = _static(tmp_path, calib, src, "mm", calibrate_method="mean_minmax", equalize=True)
    assert out.lineage[-1].params["calibrate_method"] == "mean_minmax"


def _concat_model(path: Path) -> Path:
    """Two conv branches with ranges ~100x apart, concatenated, then a 1x1 conv."""
    rng = np.random.default_rng(4)
    w1 = rng.standard_normal((3, C_IN, 1, 1)).astype(np.float32) * 10
    w2 = rng.standard_normal((3, C_IN, 1, 1)).astype(np.float32) * 0.1
    w3 = rng.standard_normal((4, 6, 1, 1)).astype(np.float32)
    nodes = [helper.make_node("Conv", ["input", "w1"], ["a"], name="c1"),
             helper.make_node("Conv", ["input", "w2"], ["b"], name="c2"),
             helper.make_node("Concat", ["a", "b"], ["cat"], name="cat", axis=1),
             helper.make_node("Conv", ["cat", "w3"], ["out"], name="c3")]
    graph = helper.make_graph(
        nodes, "cat", [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", C_IN, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", 4, 8, 8])],
        [numpy_helper.from_array(w, n) for w, n in ((w1, "w1"), (w2, "w2"), (w3, "w3"))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_concat_shared_scale_gives_every_concat_input_one_scale(tmp_path: Path, calib):
    from anneal.core.transforms import concat_groups

    src = _concat_model(tmp_path / "c.onnx")
    plain = _static(tmp_path, calib, src, "p")
    [(names, _)] = concat_groups(plain.path)
    q = onnx.load(str(plain.path))

    def scales(model):
        inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        return {n.input[0]: float(inits[n.input[1]]) for n in model.graph.node if n.op_type == "QuantizeLinear"}

    before = scales(q)
    assert before["a"] > 10 * before["b"]  # onnxruntime: one scale per input
    shared = _static(tmp_path, calib, src, "s", concat_shared_scale=True)
    after = scales(onnx.load(str(shared.path)))
    assert after["a"] == pytest.approx(after["b"]) == pytest.approx(after["cat"])
    assert shared.meta["concat_groups_shared"] == 1


def _detector_like(path: Path) -> Path:
    """A detector-style output: pixel boxes (0-640) concatenated with sigmoid scores (0-1)."""
    rng = np.random.default_rng(6)
    wb = rng.standard_normal((4, C_IN, 1, 1)).astype(np.float32)
    ws = rng.standard_normal((3, C_IN, 1, 1)).astype(np.float32)
    nodes = [helper.make_node("Conv", ["input", "wb"], ["b"], name="box_conv"),
             helper.make_node("Sigmoid", ["b"], ["bs"], name="box_sig"),
             helper.make_node("Mul", ["bs", "img"], ["boxes"], name="to_pixels"),
             helper.make_node("Conv", ["input", "ws"], ["s"], name="cls_conv"),
             helper.make_node("Sigmoid", ["s"], ["scores"], name="cls_sig"),
             helper.make_node("Concat", ["boxes", "scores"], ["cat"], name="cat", axis=1),
             helper.make_node("Reshape", ["cat", "shape"], ["out"], name="flat")]
    graph = helper.make_graph(
        nodes, "det", [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", C_IN, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", 7, 64])],
        [numpy_helper.from_array(wb, "wb"), numpy_helper.from_array(ws, "ws"),
         numpy_helper.from_array(np.array(640.0, np.float32), "img"),
         numpy_helper.from_array(np.array([0, 7, 64], np.int64), "shape")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_a_mixed_range_output_keeps_its_tail_float_by_default(tmp_path: Path, calib):
    src = _detector_like(tmp_path / "d.onnx")
    out = _static(tmp_path, calib, src, "auto")
    [flag] = out.meta["mixed_range_outputs"]
    assert flag["output"] == "out" and flag["range_ratio"] >= 20
    q = onnx.load(str(out.path))
    quantized_inputs = {n.input[0] for n in q.graph.node if n.op_type == "QuantizeLinear"}
    assert "scores" not in quantized_inputs and "boxes" not in quantized_inputs  # tail stays float
    assert "input" in quantized_inputs  # the convolutions are still INT8
    off = _static(tmp_path, calib, src, "off", float_mixed_outputs=False)
    assert "mixed_range_outputs" not in off.meta
    assert off.lineage[-1].params["float_mixed_outputs"] is False
    # a classifier (no Concat output) is never touched and its lineage is unchanged
    plain = _static(tmp_path, calib, _model(tmp_path / "m.onnx", "silu"), "cls")
    assert "mixed_range_outputs" not in plain.meta and "float_mixed_outputs" not in plain.lineage[-1].params


def test_tidl_emulation_gives_symmetric_power_of_two_activation_scales(tmp_path: Path, calib):
    src = _model(tmp_path / "m.onnx", "relu")
    out = _static(tmp_path, calib, src, "tidl", per_channel=False, activation_type="int8",
                  activation_symmetric=True, pow2_activation_scales=True)
    q = onnx.load(str(out.path))
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    acts = [n for n in q.graph.node if n.op_type == "QuantizeLinear" and n.input[0] not in inits]
    assert acts and out.meta["pow2_activation_scales"] == len({n.input[1] for n in acts})
    for n in acts:
        scale, zp = float(inits[n.input[1]]), int(inits[n.input[2]])
        assert zp == 0 and np.log2(scale) == pytest.approx(round(np.log2(scale)))
    assert out.lineage[-1].params["pow2_activation_scales"] is True
# ----- squeeze-excite sites -----------------------------------------------------


def _se_model(path: Path, act2: str = "silu", gate2: str = "Sigmoid", pool: str = "gap",
              proj_group: int = 1) -> Path:
    """EfficientNet's block: A -> SiLU -> depthwise D -> act2 -> SE(pool, FC1, ReLU, FC2, gate2)
    -> Mul -> projection, with D's output channels imbalanced ~1000x (one of them below zero).
    act2 "relu" is MobileNetV3's ReLU block (depthwise -> ReLU -> SE -> projection)."""
    wa, ba, wb = _imbalanced_weights()
    rng = np.random.default_rng(8)
    gains2 = np.array([0.02, 30.0, 1.0, 0.1, 5.0, 0.5], dtype=np.float32)
    wd = wb * gains2.reshape(C, 1, 1, 1)
    bd = np.array([0.0, 0.0, 0.0, -0.1, 0.0, 0.0], dtype=np.float32) * gains2
    w1 = rng.standard_normal((3, C, 1, 1)).astype(np.float32) * 0.3
    b1 = rng.standard_normal(3).astype(np.float32)
    w2 = rng.standard_normal((C, 3, 1, 1)).astype(np.float32)
    b2 = rng.standard_normal(C).astype(np.float32)
    wc = (rng.standard_normal((5, C, 1, 1)) if proj_group == 1
          else rng.standard_normal((C, 1, 3, 3))).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["input", "wa", "ba"], ["x"], name="conv_a"),
        helper.make_node("Sigmoid", ["x"], ["g"], name="gate"),
        helper.make_node("Mul", ["x", "g"], ["y"], name="act"),
        helper.make_node("Conv", ["y", "wd", "bd"], ["x2"], name="conv_d", group=C, pads=[1, 1, 1, 1]),
    ]
    if act2 == "silu":
        nodes += [helper.make_node("Sigmoid", ["x2"], ["g2"], name="gate_d"),
                  helper.make_node("Mul", ["g2", "x2"], ["y2"], name="act_d")]
    elif act2 == "relu":
        nodes += [helper.make_node("Relu", ["x2"], ["y2"], name="act_d")]
    else:
        nodes += [helper.make_node("HardSwish", ["x2"], ["y2"], name="act_d")]
    if pool == "gap":
        nodes += [helper.make_node("GlobalAveragePool", ["y2"], ["p"], name="se_pool")]
    else:
        nodes += [helper.make_node("ReduceMean", ["y2"], ["p"], name="se_pool", axes=[-1, -2], keepdims=1)]
    gate_kw = {"alpha": 1 / 6, "beta": 0.5} if gate2 == "HardSigmoid" else {}
    nodes += [
        helper.make_node("Conv", ["p", "w1", "b1"], ["f1"], name="se_fc1"),
        helper.make_node("Relu", ["f1"], ["f1r"], name="se_relu"),
        helper.make_node("Conv", ["f1r", "w2", "b2"], ["f2"], name="se_fc2"),
        helper.make_node(gate2, ["f2"], ["e"], name="se_gate", **gate_kw),
        helper.make_node("Mul", ["y2", "e"], ["z"], name="se_mul"),
        helper.make_node("Conv", ["z", "wc"], ["out"], name="conv_c",
                         **({} if proj_group == 1 else {"group": C, "pads": [1, 1, 1, 1]})),
    ]
    c_out = 5 if proj_group == 1 else C
    graph = helper.make_graph(
        nodes, "se",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", C_IN, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", c_out, 8, 8])],
        [numpy_helper.from_array(a, n) for a, n in (
            (wa, "wa"), (ba, "ba"), (wd, "wd"), (bd, "bd"), (w1, "w1"),
            (b1, "b1"), (w2, "w2"), (b2, "b2"), (wc, "wc"))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _assert_same(a: Path, b: Path) -> None:
    x = _batches(1, seed=7)[0]
    before, after = _run(a, x), _run(b, x)
    assert np.abs(after - before).max() <= 1e-4 * max(1.0, np.abs(before).max())


@pytest.mark.parametrize("act2,gate2,pool,proj_group", [
    ("silu", "Sigmoid", "gap", 1),
    ("hardswish_fused", "HardSigmoid", "gap", 1),
    ("silu", "HardSigmoid", "reducemean", 1),
    ("hardswish_fused", "Sigmoid", "reducemean", C),
])
def test_se_sites_are_opt_in_and_exact_in_float(tmp_path: Path, act2, gate2, pool, proj_group):
    src = _se_model(tmp_path / "m.onnx", act2, gate2, pool, proj_group)
    assert [s.kind for s in find_sites(onnx.load(str(src)))] == ["gated"]
    sites = find_sites(onnx.load(str(src)), se=True)
    assert [s.kind for s in sites] == ["gated", "gated-se"]
    se = sites[1]
    assert (se.id, se.fc1.name, se.se_mul.name, se.z, [c.name for c in se.consumers_z]) == (
        "conv_d", "se_fc1", "se_mul", "z", ["conv_c"])
    assert se.tensors == ("x2", "y2", "z")

    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), se=True, sites=["conv_d"])
    [done] = result.sites
    assert done.kind == "gated-se" and done.consumer == "conv_c"
    assert result.summary()["by_kind"]["gated-se"] == 1
    assert done.levels_after > done.levels_before
    _assert_same(src, dst)


def test_se_site_composes_with_the_gated_site_on_the_same_depthwise_conv(tmp_path: Path):
    src = _se_model(tmp_path / "m.onnx", "hardswish_fused")
    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), se=True)
    # conv_d is the consumer of the first site and the producer of the second.
    assert sorted(s.kind for s in result.sites) == ["gated", "gated-se"]
    assert {s.producer for s in result.sites} == {"conv_a", "conv_d"}
    assert len(result.gate_nodes) == 4  # an inserted Mul and a gate per site
    _assert_same(src, dst)
    assert {r.site for r in rank_sites(src, _batches(), se=True)} == {"conv_a", "conv_d"}


@pytest.mark.parametrize("mix", [(0.0, 1.0), (0.5, 0.5)])
def test_se_site_with_mixed_scales_stays_exact(tmp_path: Path, mix):
    src = _se_model(tmp_path / "m.onnx")
    dst = tmp_path / "eq.onnx"
    assert len(equalise(src, dst, _batches(), se=True, mix=mix).sites) == 2
    _assert_same(src, dst)


def _edit(src: Path, dst: Path, fn) -> Path:
    m = onnx.load(str(src))
    fn(m)
    onnx.save(m, str(dst))
    return dst


def _node(m, name):
    return next(n for n in m.graph.node if n.name == name)


def _side_output(m, tensor: str) -> None:
    m.graph.node.append(helper.make_node("Relu", [tensor], ["side"], name="side"))
    m.graph.output.append(helper.make_tensor_value_info("side", TensorProto.FLOAT, None))


def test_se_site_matching_is_strict(tmp_path: Path):
    src = _se_model(tmp_path / "m.onnx")

    def se_count(path):
        return [s.kind for s in find_sites(onnx.load(str(path)), se=True)].count("gated-se")

    assert se_count(src) == 1

    def pool_over_channels(m):  # a mean over C mixes channels: not per-channel linear in s
        idx = list(m.graph.node).index(_node(m, "se_pool"))
        m.graph.node.remove(_node(m, "se_pool"))
        m.graph.node.insert(idx, helper.make_node("ReduceMean", ["y2"], ["p"], name="se_pool", axes=[1, 2, 3]))

    def no_gate(m):  # z = y2 * f2: not a squeeze-excite gate
        _node(m, "se_mul").input[1] = "f2"
        m.graph.node.remove(_node(m, "se_gate"))

    edits = {
        "pool_over_channels": pool_over_channels,
        "y2_second_consumer": lambda m: _side_output(m, "y2"),
        "p_second_consumer": lambda m: _side_output(m, "p"),
        "e_second_consumer": lambda m: _side_output(m, "e"),
        "z_feeds_non_conv": lambda m: _side_output(m, "z"),
        "z_is_output": lambda m: m.graph.output.append(
            helper.make_tensor_value_info("z", TensorProto.FLOAT, None)),
        "no_gate": no_gate,
    }
    for name, fn in edits.items():
        assert se_count(_edit(src, tmp_path / f"{name}.onnx", fn)) == 0, name


def test_se_site_with_a_tied_weight_is_left_alone(tmp_path: Path):
    src = _se_model(tmp_path / "m.onnx")

    def share(weight: str, input_: str):
        def fn(m):  # the weight also feeds a conv on another branch
            if input_ == "p2":  # a second pool for the twin, so FC1's input keeps one consumer
                m.graph.node.append(helper.make_node("GlobalAveragePool", ["y"], ["p2"], name="pool2"))
            m.graph.node.append(helper.make_node("Conv", [input_, weight], ["twin"], name="twin"))
            m.graph.output.append(helper.make_tensor_value_info("twin", TensorProto.FLOAT, ["N", "C", "H", "W"]))
        return fn

    for weight, input_ in (("w1", "p2"), ("wc", "y")):
        path = _edit(src, tmp_path / f"shared_{weight}.onnx", share(weight, input_))
        assert [s.kind for s in find_sites(onnx.load(str(path)), se=True)].count("gated-se") == 1
        dst = tmp_path / f"eq_{weight}.onnx"
        assert equalise(path, dst, _batches(), se=True, sites=["conv_d"]).sites == []
        _assert_same(path, dst)


def test_static_quantization_records_se_equalisation(tmp_path: Path, calib):
    src = _se_model(tmp_path / "m.onnx")
    out = _static(tmp_path, calib, src, "se", equalize=True, equalize_se=True)
    assert out.lineage[-1].params["equalize_se"] is True
    assert out.meta["equalisation"]["by_kind"]["gated-se"] == 1
    assert out.meta["equalisation"]["max_abs_logit_change"] < 1e-3
    plain = _static(tmp_path, calib, src, "plain", equalize=True)
    assert "equalize_se" not in plain.lineage[-1].params
    assert plain.meta["equalisation"]["by_kind"]["gated-se"] == 0
    with pytest.raises(TransformError):
        _static(tmp_path, calib, src, "bad", equalize_se=True)


# ----- ReLU squeeze-excite sites ---------------------------------------------------


@pytest.mark.parametrize("gate2,pool,proj_group", [
    ("Sigmoid", "gap", 1),
    ("HardSigmoid", "reducemean", 1),
    ("HardSigmoid", "gap", C),
])
def test_relu_se_sites_are_opt_in_positive_and_exact_in_float(tmp_path: Path, gate2, pool, proj_group):
    src = _se_model(tmp_path / "m.onnx", "relu", gate2, pool, proj_group)
    assert [s.kind for s in find_sites(onnx.load(str(src)))] == ["gated"]
    sites = find_sites(onnx.load(str(src)), se=True)
    assert [s.kind for s in sites] == ["gated", "relu-se"]
    se = sites[1]
    assert (se.id, se.act.name, se.gate, se.fc1.name, se.se_mul.name, se.z) == (
        "conv_d", "act_d", None, "se_fc1", "se_mul", "z")
    assert se.tensors == ("x2", "y2", "z")

    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), se=True, sites=["conv_d"])
    [done] = result.sites
    assert done.kind == "relu-se" and done.consumer == "conv_c" and done.gate is None
    assert done.channels_mirrored == 0 and done.scale_max > 1.5
    assert result.gate_nodes == []  # ReLU(s x) = s ReLU(x): no gate Mul inserted
    assert result.summary()["by_kind"]["relu-se"] == 1
    assert done.levels_after > done.levels_before
    _assert_same(src, dst)
    ops = [n.op_type for n in onnx.load(str(dst)).graph.node]
    assert ops == [n.op_type for n in onnx.load(str(src)).graph.node]


def test_relu_se_site_composes_with_the_gated_site_on_the_same_depthwise_conv(tmp_path: Path):
    src = _se_model(tmp_path / "m.onnx", "relu")
    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), se=True)
    assert sorted(s.kind for s in result.sites) == ["gated", "relu-se"]
    assert {s.producer for s in result.sites} == {"conv_a", "conv_d"}
    assert len(result.gate_nodes) == 2  # only the gated site gets a Mul and a gate
    _assert_same(src, dst)
    assert {r.site for r in rank_sites(src, _batches(), se=True)} == {"conv_a", "conv_d"}


@pytest.mark.parametrize("mix", [(0.0, 1.0), (0.5, 0.5)])
def test_relu_se_site_with_mixed_scales_stays_exact_and_positive(tmp_path: Path, mix):
    src = _se_model(tmp_path / "m.onnx", "relu")
    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), se=True, mix=mix)
    assert len(result.sites) == 2
    assert next(s for s in result.sites if s.kind == "relu-se").channels_mirrored == 0
    _assert_same(src, dst)


def test_relu_se_site_matching_is_strict(tmp_path: Path):
    src = _se_model(tmp_path / "m.onnx", "relu")

    def count(path):
        return [s.kind for s in find_sites(onnx.load(str(path)), se=True)].count("relu-se")

    assert count(src) == 1

    def pool_over_channels(m):
        idx = list(m.graph.node).index(_node(m, "se_pool"))
        m.graph.node.remove(_node(m, "se_pool"))
        m.graph.node.insert(idx, helper.make_node("ReduceMean", ["y2"], ["p"], name="se_pool", axes=[1, 2, 3]))

    def no_gate(m):
        _node(m, "se_mul").input[1] = "f2"
        m.graph.node.remove(_node(m, "se_gate"))

    def leaky(m):  # not ReLU: the rewrite would change the negative slope's output
        _node(m, "act_d").op_type = "LeakyRelu"

    edits = {
        "pool_over_channels": pool_over_channels,
        "x2_second_consumer": lambda m: _side_output(m, "x2"),
        "y2_second_consumer": lambda m: _side_output(m, "y2"),
        "p_second_consumer": lambda m: _side_output(m, "p"),
        "e_second_consumer": lambda m: _side_output(m, "e"),
        "z_feeds_non_conv": lambda m: _side_output(m, "z"),
        "z_is_output": lambda m: m.graph.output.append(
            helper.make_tensor_value_info("z", TensorProto.FLOAT, None)),
        "no_gate": no_gate,
        "leaky_relu": leaky,
    }
    for name, fn in edits.items():
        assert count(_edit(src, tmp_path / f"{name}.onnx", fn)) == 0, name


def test_static_quantization_records_relu_se_equalisation(tmp_path: Path, calib):
    src = _se_model(tmp_path / "m.onnx", "relu")
    out = _static(tmp_path, calib, src, "se", equalize=True, equalize_se=True)
    assert out.meta["equalisation"]["by_kind"]["relu-se"] == 1
    assert out.meta["equalisation"]["max_abs_logit_change"] < 1e-3
    plain = _static(tmp_path, calib, src, "plain", equalize=True)
    assert plain.meta["equalisation"]["by_kind"]["relu-se"] == 0


# ----- per-tensor weights -----------------------------------------------------------


def test_per_tensor_equalisation_defaults_to_a_half_mix_and_records_it(tmp_path: Path, calib):
    src = _model(tmp_path / "m.onnx", "silu")
    out = _static(tmp_path, calib, src, "pt", per_channel=False, equalize=True)
    params = out.lineage[-1].params
    assert params["per_channel"] is False and params["equalize_mix"] == 0.5
    assert out.meta["equalisation"]["sites"] == 1
    assert out.meta["equalisation"]["max_abs_logit_change"] < 1e-3
    # The mix reaches the rewrite: t=0 is the plain (per-channel) scale, t=0.5 is not.
    t0 = _static(tmp_path, calib, src, "pt0", per_channel=False, equalize=True, equalize_mix=0)
    assert t0.lineage[-1].params["equalize_mix"] == 0.0
    plain = _static(tmp_path, calib, src, "pc", equalize=True)
    assert t0.meta["equalised_sites"][0]["scale_max"] == plain.meta["equalised_sites"][0]["scale_max"]
    assert out.meta["equalised_sites"][0]["scale_max"] != t0.meta["equalised_sites"][0]["scale_max"]


def test_per_channel_equalisation_keeps_its_lineage_keys(tmp_path: Path, calib):
    src = _model(tmp_path / "m.onnx", "silu")
    out = _static(tmp_path, calib, src, "pc", equalize=True)
    assert "equalize_mix" not in out.lineage[-1].params
    mixed = _static(tmp_path, calib, src, "pcm", equalize=True, equalize_mix=0.25)
    assert mixed.lineage[-1].params["equalize_mix"] == 0.25
    assert mixed.meta["equalisation"]["max_abs_logit_change"] < 1e-3


@pytest.mark.parametrize("params", [
    {"equalize_mix": 0.5},  # without equalize
    {"per_channel": False, "equalize_mix": 0.5},
    {"equalize": True, "equalize_mix": 1.5},
    {"equalize": True, "equalize_mix": -0.1},
    {"equalize": True, "equalize_mix": True},
    {"equalize": True, "equalize_mix": "half"},
])
def test_invalid_equalize_mix_is_rejected(tmp_path: Path, calib, params):
    with pytest.raises(TransformError):
        _static(tmp_path, calib, _model(tmp_path / "m.onnx", "silu"), "bad", **params)


# ----- the joint-damage switch sees the sites the rewrite takes ------------------------


def test_min_damage_probe_includes_se_site_tensors(tmp_path: Path, calib, monkeypatch):
    import anneal.core.activation_sensitivity as act_sens

    seen: list[list[str]] = []
    real = act_sens.joint_damage

    def spy(src, tensors, *args, **kwargs):
        seen.append(list(tensors))
        return real(src, tensors, *args, **kwargs)

    monkeypatch.setattr(act_sens, "joint_damage", spy)
    src = _se_model(tmp_path / "m.onnx", "relu")
    _static(tmp_path, calib, src, "plain", equalize=True, equalize_min_damage=0.0)
    _static(tmp_path, calib, src, "se", equalize=True, equalize_se=True, equalize_min_damage=0.0)
    assert seen[0] == ["x", "y"]
    assert seen[1] == ["x", "x2", "y", "y2", "z"]


def test_pow2_alignment_puts_the_peak_on_a_power_of_two_grid_and_stays_exact(tmp_path: Path):
    from anneal.core.equalize import pow2_alignment_gain

    g = pow2_alignment_gain(np.array([-1.0, 0.0]), np.array([2.0, 5.0]), np.array([1.0, 1.0]))
    assert 1.0 <= g < 2.0 and np.log2(5.0 * g / 127) == pytest.approx(round(np.log2(5.0 * g / 127)))
    src = _model(tmp_path / "m.onnx", "silu")
    dst = tmp_path / "eq.onnx"
    equalise(src, dst, _batches(), pow2_align=True)
    x = _batches(1, seed=7)[0]
    before, after = _run(src, x), _run(dst, x)
    assert np.abs(after - before).max() <= 1e-4 * max(1.0, np.abs(before).max())


# ----- gate conv: the gate fed through a depthwise 1x1 Conv instead of a Mul ------


@pytest.mark.parametrize("activation", ["silu", "hardswish", "hardswish_fused"])
def test_gate_conv_is_exact_in_float_and_replaces_the_gate_mul(tmp_path: Path, activation: str):
    from anneal.core.equalize import GATE_CONV_PREFIX

    src = _model(tmp_path / "m.onnx", activation)
    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), gate_conv=True)
    assert len(result.sites) == 1
    _assert_same(src, dst)
    m = onnx.load(str(dst))
    assert _gate_muls(dst) == []
    [conv] = [n for n in m.graph.node if n.name.startswith(GATE_CONV_PREFIX)]
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    w = inits[conv.input[1]]
    assert conv.op_type == "Conv" and w.shape == (C, 1, 1, 1) and not inits[conv.input[2]].any()
    assert {a.name: helper.get_attribute_value(a) for a in conv.attribute}["group"] == C
    # The gate reads the conv; the conv reads the balanced tensor x'.
    gate = next(n for n in m.graph.node if n.input and n.input[0] == conv.output[0])
    assert gate.op_type in ("Sigmoid", "HardSigmoid") and conv.input[0] == "x"
    assert result.gate_nodes == [conv.name, gate.name]
    # Same scales as the Mul form: the conv's weight is its 1/s.
    plain = equalise(src, tmp_path / "mul.onnx", _batches())
    mm = onnx.load(str(tmp_path / "mul.onnx"))
    [mul] = [n for n in mm.graph.node if n.name.startswith(GATE_MUL_PREFIX)]
    inv = {i.name: numpy_helper.to_array(i) for i in mm.graph.initializer}[mul.input[1]]
    np.testing.assert_allclose(w.ravel(), inv.ravel(), rtol=1e-6)
    assert plain.sites[0].scale_max == result.sites[0].scale_max


@pytest.mark.parametrize("act2", ["silu", "hardswish_fused"])
def test_gate_conv_composes_with_se_mix_and_pow2_align(tmp_path: Path, act2: str):
    from anneal.core.equalize import GATE_CONV_PREFIX

    src = _se_model(tmp_path / "m.onnx", act2)
    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), se=True, mix=(0.5, 0.5), pow2_align=True, gate_conv=True)
    assert sorted(s.kind for s in result.sites) == ["gated", "gated-se"]
    assert len(result.gate_nodes) == 4
    names = {n.name for n in onnx.load(str(dst)).graph.node}
    assert set(result.gate_nodes) <= names
    assert sum(n.startswith(GATE_CONV_PREFIX) for n in result.gate_nodes) == 2
    assert _gate_muls(dst) == []
    _assert_same(src, dst)


def test_static_quantization_records_gate_conv(tmp_path: Path, calib):
    src = _se_model(tmp_path / "m.onnx")
    out = _static(tmp_path, calib, src, "gc", equalize=True, equalize_se=True, equalize_gate_conv=True,
                  float_gates=True)
    assert out.lineage[-1].params["equalize_gate_conv"] is True
    assert out.meta["equalisation"]["max_abs_logit_change"] < 1e-3
    plain = _static(tmp_path, calib, src, "plain", equalize=True)
    assert "equalize_gate_conv" not in plain.lineage[-1].params
    with pytest.raises(TransformError):
        _static(tmp_path, calib, src, "bad", equalize_gate_conv=True)


def test_grid_scales_put_every_inverse_on_one_power_of_two_int8_grid():
    from anneal.core.equalize import grid_scales

    s = np.array([1.0, 3.7, 279.0, -12.5, 0.9, 41.0])
    g = grid_scales(s)
    inv = 1.0 / g
    step = np.abs(inv).max() / 127  # what a symmetric per-tensor int8 quantizer would pick
    assert np.isclose(np.log2(step), round(np.log2(step)))  # a power of two
    k = inv / step
    assert np.allclose(k, np.round(k)) and np.all(np.abs(np.round(k)) >= 1)
    assert np.all(np.sign(g) == np.sign(s))


@pytest.mark.parametrize("activation", ["silu", "hardswish"])
def test_grid_inverse_keeps_the_rewrite_exact(tmp_path: Path, activation: str):
    from onnx import numpy_helper

    src = _model(tmp_path / "m.onnx", activation)
    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), grid_inverse=True)
    assert result.sites
    x = _batches(1, seed=7)[0]
    assert np.allclose(_run(src, x), _run(dst, x), rtol=1e-4, atol=1e-4)
    for init in onnx.load(str(dst)).graph.initializer:
        if init.name.startswith("anneal_eq_inv"):
            inv = numpy_helper.to_array(init).astype(np.float64).reshape(-1)
            k = inv / (np.abs(inv).max() / 127)
            assert np.allclose(k, np.round(k), atol=1e-3)


def test_residual_sites_take_pure_activation_scales_even_with_a_mix(tmp_path: Path):
    """The weight mix is not applied to a gated residual site (it zeroed LRASPP's stem channels)."""
    src = _residual_model(tmp_path / "m.onnx")
    plain = equalise(src, tmp_path / "a.onnx", _batches(), residual=True)
    mixed = equalise(src, tmp_path / "b.onnx", _batches(), residual=True, mix=(0.5, 0.5))
    a = [s for s in plain.sites if s.kind == "gated-residual"][0]
    b = [s for s in mixed.sites if s.kind == "gated-residual"][0]
    assert np.isclose(a.scale_median, b.scale_median)
    x = _batches(1, seed=7)[0]
    assert np.allclose(_run(src, x), _run(tmp_path / "b.onnx", x), rtol=1e-4, atol=1e-4)
