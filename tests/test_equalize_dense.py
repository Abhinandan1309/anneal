"""Equalisation into dense consumers: found, exact in float, and only applied where it helps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from anneal.core.equalize_dense import equalise_dense, find_dense_sites, site_error

D, H = 8, 16


def _mlp(path: Path, gains: np.ndarray | None = None) -> Path:
    """ConvNeXt's MLP: MatMul + bias -> GELU (Erf form) -> x0.5 -> MatMul, channels last."""
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((D, H)).astype(np.float32)
    if gains is not None:
        w1 *= gains[None, :]
    inits = [
        numpy_helper.from_array(w1, "w1"),
        numpy_helper.from_array(rng.standard_normal(H).astype(np.float32) * 0.1, "b1"),
        numpy_helper.from_array(rng.standard_normal((H, D)).astype(np.float32), "w2"),
        numpy_helper.from_array(np.array(1.4142135, np.float32), "sqrt2"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
    ]
    nodes = [
        helper.make_node("MatMul", ["x", "w1"], ["h0"], name="fc1"),
        helper.make_node("Add", ["b1", "h0"], ["h"], name="fc1_bias"),
        helper.make_node("Div", ["h", "sqrt2"], ["hd"], name="gelu_div"),
        helper.make_node("Erf", ["hd"], ["e"], name="gelu_erf"),
        helper.make_node("Add", ["e", "one"], ["e1"], name="gelu_add"),
        helper.make_node("Mul", ["h", "e1"], ["he"], name="gelu_mul"),
        helper.make_node("Mul", ["he", "half"], ["gelu"], name="gelu_half"),
        helper.make_node("MatMul", ["gelu", "w2"], ["y"], name="fc2"),
    ]
    g = helper.make_graph(nodes, "mlp", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 5, D])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 5, D])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return path


def _fused_mbconv(path: Path) -> Path:
    """EfficientNetV2's Fused-MBConv: 3x3 Conv -> SiLU -> 1x1 Conv, NCHW."""
    rng = np.random.default_rng(1)
    gains = np.array([30.0, 1, 1, 0.05, 1, 0.02], np.float32)
    wa = rng.standard_normal((6, 4, 3, 3)).astype(np.float32) * gains[:, None, None, None]
    inits = [numpy_helper.from_array(wa, "wa"), numpy_helper.from_array(np.zeros(6, np.float32), "ba"),
             numpy_helper.from_array(rng.standard_normal((5, 6, 1, 1)).astype(np.float32), "wb")]
    nodes = [
        helper.make_node("Conv", ["x", "wa", "ba"], ["a"], name="expand", pads=[1, 1, 1, 1]),
        helper.make_node("Sigmoid", ["a"], ["g"], name="gate"),
        helper.make_node("Mul", ["a", "g"], ["s"], name="silu"),
        helper.make_node("Conv", ["s", "wb"], ["y"], name="project"),
    ]
    g = helper.make_graph(nodes, "fm", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 4, 8, 8])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 5, 8, 8])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return path


def _run(path: Path, x: np.ndarray) -> np.ndarray:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})[0]


IMBALANCED = np.array([50.0] + [1.0] * 7 + [0.02] * 8, np.float32)


def test_both_layouts_are_found(tmp_path: Path):
    mlp = find_dense_sites(onnx.load(str(_mlp(tmp_path / "m.onnx"))))
    assert len(mlp) == 1 and mlp[0].channel_axis == -1 and mlp[0].a_bias == "b1"
    assert mlp[0].y == "gelu"  # past the scalar 0.5
    conv = find_dense_sites(onnx.load(str(_fused_mbconv(tmp_path / "c.onnx"))))
    assert len(conv) == 1 and conv[0].channel_axis == 1


@pytest.mark.parametrize("builder", ["mlp", "conv"])
def test_rewrite_is_exact_in_float_and_lowers_the_simulated_error(tmp_path: Path, builder):
    rng = np.random.default_rng(2)
    if builder == "mlp":
        src = _mlp(tmp_path / "m.onnx", IMBALANCED)
        batches = [rng.standard_normal((8, 5, D)).astype(np.float32) for _ in range(3)]
    else:
        src = _fused_mbconv(tmp_path / "c.onnx")
        batches = [rng.standard_normal((8, 4, 8, 8)).astype(np.float32) for _ in range(3)]
    sites, gates, change = equalise_dense(src, tmp_path / "eq.onnx", batches)
    assert len(sites) == 1
    assert sites[0].alpha > 0 and sites[0].error_after < sites[0].error_before
    assert gates and gates[0].startswith("anneal_eq_gate_mul_d")
    before, after = _run(src, batches[0]), _run(tmp_path / "eq.onnx", batches[0])
    assert np.abs(after - before).max() <= 1e-4 * max(1.0, np.abs(before).max())
    assert change is not None and change < 1e-3


def test_a_balanced_site_is_left_alone(tmp_path: Path):
    rng = np.random.default_rng(3)
    src = _mlp(tmp_path / "m.onnx")  # no imbalance to fix
    batches = [rng.standard_normal((8, 5, D)).astype(np.float32) for _ in range(2)]
    sites, gates, _ = equalise_dense(src, tmp_path / "eq.onnx", batches)
    # Either untouched, or touched only because it measurably lowers the error.
    for s in sites:
        assert s.error_after < s.error_before


def test_site_error_is_the_baseline_at_unit_scale_and_rewards_rescuing_a_starved_channel():
    rng = np.random.default_rng(4)
    y = rng.standard_normal((512, 4)) * np.array([100.0, 1.0, 0.01, 0.01])
    w = rng.standard_normal((4, 3)) * np.array([0.01, 1.0, 100.0, 100.0])[:, None]
    base = site_error(y, w, np.ones(4))
    assert base == pytest.approx(site_error(y, w, np.ones(4)))
    assert site_error(y, w, np.array([1.0, 1.0, 100.0, 100.0])) < base


def test_static_quantization_records_dense_equalisation(tmp_path: Path):
    from anneal.core.artifact import ModelArtifact
    from anneal.core.transforms import TransformContext, apply_transform

    class Calib:
        name, split = "synthetic", "calib"

        def calibration_batches(self, limit):
            rng = np.random.default_rng(5)
            for _ in range(3):
                yield rng.standard_normal((8, 5, D)).astype(np.float32)

    out = apply_transform(
        "quantize_static_int8", {"per_channel": True, "equalize_dense": True},
        ModelArtifact(path=_mlp(tmp_path / "m.onnx", IMBALANCED)),
        TransformContext(workdir=tmp_path / "w", calibset=Calib()),
    )
    assert out.meta["dense_equalisation"]["sites_rewritten"] == 1
    assert out.lineage[-1].params["equalize_dense"] is True
