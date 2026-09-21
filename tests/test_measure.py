"""Percentiles, baseline comparison, and the fields that keep benchmarks honest."""

from __future__ import annotations

import numpy as np
import pytest

from anneal.core.measure import Measurement, _compare_to_baseline, _percentile
from conftest import make_measurement


def test_percentile_returns_a_real_observation():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    # Nearest-rank never invents a value between two samples.
    assert _percentile(values, 50) in values
    assert _percentile(values, 99) == 100.0
    assert _percentile(values, 0) == 1.0


def test_percentile_handles_empty_input():
    assert np.isnan(_percentile([], 50))


def test_p99_tracks_the_tail_not_the_mean():
    # One slow outlier barely moves the median but must dominate p99.
    values = [10.0] * 99 + [500.0]
    assert _percentile(values, 50) == 10.0
    assert _percentile(values, 99) >= 10.0
    assert max(values) == 500.0


def test_throughput_accounts_for_batch_size():
    single = make_measurement(latency=10.0, batch_size=1)
    batched = make_measurement(latency=10.0, batch_size=8)
    assert single.throughput_ips == pytest.approx(100.0)
    assert batched.throughput_ips == pytest.approx(800.0)


def test_identical_outputs_agree_perfectly():
    logits = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    agreement, cosine = _compare_to_baseline(logits, logits)
    assert agreement == 1.0
    assert cosine == pytest.approx(1.0)


def test_flipped_predictions_are_caught_by_agreement():
    base = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    flipped = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    agreement, _ = _compare_to_baseline(base, flipped)
    assert agreement == 0.0


def test_agreement_catches_a_model_that_got_lucky():
    # Same top-1 accuracy against ground truth, completely different behaviour.
    base = np.array([[5.0, 1.0], [5.0, 1.0]], dtype=np.float32)
    different = np.array([[5.0, 1.0], [1.0, 5.0]], dtype=np.float32)
    agreement, cosine = _compare_to_baseline(base, different)
    assert agreement == 0.5
    assert cosine < 1.0


def test_mismatched_shapes_report_nothing_rather_than_guessing():
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((2, 4), dtype=np.float32)
    assert _compare_to_baseline(a, b) == (None, None)


def test_zero_vectors_do_not_produce_nan_cosine():
    zeros = np.zeros((2, 3), dtype=np.float32)
    agreement, cosine = _compare_to_baseline(zeros, zeros)
    assert agreement == 1.0
    assert cosine is None


def test_measurement_round_trip():
    original = make_measurement(latency=12.5, accuracy=0.873)
    restored = Measurement.from_dict(original.to_dict())
    assert restored.latency_ms_p50 == original.latency_ms_p50
    assert restored.accuracy == original.accuracy
    assert restored.providers_used == original.providers_used


def test_size_mb_uses_binary_megabytes():
    assert make_measurement(size_bytes=1024 * 1024).size_mb == pytest.approx(1.0)


def test_an_unloadable_graph_becomes_a_measurement_error_not_a_crash(tmp_path):
    # e.g. a quantized graph whose kernels this onnxruntime build does not ship. The loop
    # turns MeasurementError into a recorded failed trial, so the run carries on.
    from pathlib import Path

    from anneal.core.artifact import ModelArtifact
    from anneal.core.measure import Benchmarker, MeasurementError
    from anneal.core.targets import get_target

    bad = Path(tmp_path) / "broken.onnx"
    bad.write_bytes(b"not an onnx graph")
    with pytest.raises(MeasurementError, match="could not load"):
        Benchmarker(get_target("cpu-1t"), warmup=1, runs=1).measure(ModelArtifact(path=bad))
