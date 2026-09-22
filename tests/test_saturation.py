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
