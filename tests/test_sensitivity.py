"""Rank correlation, and the verdict on whether the cheap proxy earns its place."""

from __future__ import annotations

import pytest

from anneal.core.sensitivity import (
    LayerSensitivity,
    _average_ranks,
    proxy_agreement,
    spearman,
)


# ----- ranking -------------------------------------------------------------


def test_average_ranks_are_one_based():
    assert _average_ranks([10.0, 20.0, 30.0]) == [1.0, 2.0, 3.0]


def test_tied_values_share_the_averaged_rank():
    # Two values tied for ranks 1 and 2 both become 1.5; ignoring this biases Spearman.
    assert _average_ranks([5.0, 5.0, 9.0]) == [1.5, 1.5, 3.0]


def test_all_tied_values_collapse_to_one_rank():
    assert _average_ranks([7.0, 7.0, 7.0]) == [2.0, 2.0, 2.0]


# ----- spearman ------------------------------------------------------------


def test_perfect_agreement_is_one():
    assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)


def test_perfect_disagreement_is_minus_one():
    assert spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_monotonic_but_nonlinear_still_scores_one():
    # This is why Spearman and not Pearson: only the ordering matters.
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == pytest.approx(1.0)


def test_no_variance_is_undefined_rather_than_zero():
    assert spearman([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None


def test_too_few_points_is_undefined():
    assert spearman([1.0, 2.0], [1.0, 2.0]) is None


def test_mismatched_lengths_is_undefined():
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0]) is None


# ----- verdict -------------------------------------------------------------


def layer(node: str, proxy: float, changed: float | None, error: str | None = None):
    return LayerSensitivity(
        node=node, op_type="Conv", proxy_error=proxy, changed_fraction=changed, error=error
    )


def test_a_proxy_that_predicts_perfectly_is_called_strong():
    results = [layer(f"n{i}", float(i), float(i) / 10) for i in range(6)]
    verdict = proxy_agreement(results)
    assert verdict["spearman"] == pytest.approx(1.0)
    assert "strong" in verdict["verdict"]
    assert verdict["top5_overlap"] == 5


def test_a_proxy_that_predicts_backwards_is_called_out():
    results = [layer(f"n{i}", float(i), float(6 - i) / 10) for i in range(6)]
    verdict = proxy_agreement(results)
    assert verdict["spearman"] < 0
    assert "none" in verdict["verdict"]


def test_unmeasured_layers_are_excluded_from_the_verdict():
    results = [layer(f"n{i}", float(i), float(i) / 10) for i in range(5)]
    results.append(layer("broken", 99.0, None, error="quantization failed"))
    assert proxy_agreement(results)["n"] == 5


def test_too_few_measured_layers_refuses_to_judge():
    verdict = proxy_agreement([layer("a", 1.0, 0.1), layer("b", 2.0, 0.2)])
    assert verdict["spearman"] is None
    assert "too few" in verdict["verdict"]


def test_layer_serialises():
    d = layer("conv1", 0.015, 0.04).to_dict()
    assert d["node"] == "conv1"
    assert d["changed_fraction"] == 0.04


def test_measured_flag_reflects_whether_a_number_exists():
    assert layer("a", 1.0, 0.1).measured
    assert not layer("a", 1.0, None, error="boom").measured
