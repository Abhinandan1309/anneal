"""Emulated 16-bit pair saturation, and the analyser on a real quantized graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anneal.core.saturation import (
    I16_MAX,
    analyse,
    pair_sums,
    saturation_stats,
    summarise,
    weight_pair_risk,
)


def test_pair_sums_are_exact_adjacent_pairs():
    a = np.array([[1, 2, 3, 4]])
    w = np.array([[10, 20, 30, 40]])
    # (1*10 + 2*20), (3*30 + 4*40)
    assert pair_sums(a, w).tolist() == [[[50, 250]]]


def test_odd_reduction_length_pads_with_zero():
    a = np.array([[1, 2, 3]])
    w = np.array([[1, 1, 1]])
    assert pair_sums(a, w).tolist() == [[[3, 3]]]


def test_full_range_weights_with_maximal_activations_saturate():
    # 255*127 + 255*127 = 64,770 > 32,767: the case that breaks non-VNNI x86.
    a = np.full((1, 2), 255)
    w = np.full((1, 2), 127)
    stats = saturation_stats(a, w)
    assert stats["saturated_pairs"] == 1
    assert stats["accumulators_affected"] == 1


def test_seven_bit_weights_can_never_saturate():
    # reduce_range: |w| <= 63, so the worst pair is 255*63*2 = 32,130 <= 32,767.
    assert 255 * 63 * 2 <= I16_MAX
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(64, 128))
    w = rng.integers(-63, 64, size=(32, 128))
    assert saturation_stats(a, w)["saturated_pairs"] == 0
    assert weight_pair_risk(w) == 0.0


def test_saturation_error_is_measured_against_exact_arithmetic():
    a = np.array([[255, 255, 0, 0]])
    w = np.array([[127, 127, 5, 5]])
    stats = saturation_stats(a, w)
    exact = 255 * 127 * 2
    assert stats["relative_error_sum"] == pytest.approx((exact - I16_MAX) / exact)


def test_weight_pair_risk_counts_only_pairs_that_can_exceed_the_limit():
    w = np.array([[127, 127, 10, 10]])  # first pair risky, second not
    assert weight_pair_risk(w) == pytest.approx(0.5)


def _static_quantize(tiny_onnx: Path, tmp_path: Path, *, reduce_range: bool) -> Path:
    from anneal.core.artifact import ModelArtifact
    from anneal.core.dataset import SyntheticEvalSet
    from anneal.core.transforms import TransformContext, apply_transform

    ctx = TransformContext(
        workdir=tmp_path / ("rr" if reduce_range else "full"),
        calibset=SyntheticEvalSet(shape=(3, 16, 16), n=16, batch_size=8, seed=1),
        calib_samples=16,
    )
    return apply_transform(
        "quantize_static_int8",
        {"per_channel": True, "reduce_range": reduce_range},
        ModelArtifact(path=tiny_onnx),
        ctx,
    ).path


def test_analyser_finds_the_quantized_layers_of_a_real_graph(tiny_onnx: Path, tmp_path: Path):
    model = _static_quantize(tiny_onnx, tmp_path, reduce_range=False)
    probe = [np.random.default_rng(0).standard_normal((4, 3, 16, 16), dtype=np.float32)]
    results = analyse(model, probe, n_positions=32)

    assert {r.op_type for r in results} == {"Conv", "Gemm"}
    for r in results:
        assert r.pairs_checked > 0
        assert r.weight_max_abs <= 127
        assert 0.0 <= r.accumulator_rate <= 1.0
    summary = summarise(results)
    assert summary["layers"] == len(results)


def test_reduce_range_makes_saturation_impossible_on_a_real_graph(tiny_onnx: Path, tmp_path: Path):
    model = _static_quantize(tiny_onnx, tmp_path, reduce_range=True)
    probe = [np.random.default_rng(0).standard_normal((4, 3, 16, 16), dtype=np.float32)]
    summary = summarise(analyse(model, probe, n_positions=32))
    assert summary["layers_saturating"] == 0
    assert not summary["saturation_possible"]


def test_an_unquantized_model_has_nothing_to_analyse(tiny_onnx: Path):
    probe = [np.zeros((1, 3, 16, 16), dtype=np.float32)]
    assert analyse(tiny_onnx, probe) == []


def test_cpu_features_classifies_the_int8_path():
    from anneal.core.environment import cpu_features

    info = cpu_features()
    assert info["int8_path"] in {"x86-avx2-16bit", "x86-vnni", "arm-dotprod", "unknown"}
    assert isinstance(info["int8_features"], list)


def test_matmul_activations_reduce_over_the_last_axis(tmp_path):
    """A transformer's (batch, tokens, K) activation is batch*tokens rows of K, not batch rows."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    from anneal.core.saturation import analyse

    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 8)).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")], "t",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 5, 8])],
        [numpy_helper.from_array(w, "w")],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    src, out = tmp_path / "m.onnx", tmp_path / "q.onnx"
    onnx.save(m, str(src))
    batches = [np.abs(rng.standard_normal((2, 5, 16))).astype(np.float32) for _ in range(2)]

    class R:
        def __init__(self):
            self.it = iter({"x": b} for b in batches)

        def get_next(self):
            return next(self.it, None)

    quantize_static(str(src), str(out), R(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    layers = analyse(out, batches[:1], n_positions=64)
    assert len(layers) == 1 and layers[0].reduction_len == 16
