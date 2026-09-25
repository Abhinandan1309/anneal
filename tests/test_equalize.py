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
from anneal.core.equalize import GATE_MUL_PREFIX, choose_scales, equalise, find_sites
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
