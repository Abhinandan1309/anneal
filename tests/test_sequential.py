"""Anytime-valid sequential acceptance testing.

The property that matters most is the error guarantee *under optional stopping*, so it is
checked by simulation at the hardest point: a true accuracy change exactly on the budget,
where both hypotheses are as close to true as they can be.
"""

from __future__ import annotations

import numpy as np
import pytest

from anneal.core.sequential import (
    ConfidenceSequence,
    SequentialBudgetTest,
    fixed_n_for_power,
    paired_scores,
    run_sequential,
)


def draw(rng: np.random.Generator, n: int, p_fix: float, p_break: float) -> np.ndarray:
    u = rng.random(n)
    return np.where(u < p_fix, 1, np.where(u < p_fix + p_break, -1, 0)).astype(int)


# ----- basics --------------------------------------------------------------


def test_paired_scores():
    orig = np.array([True, True, False, False])
    cand = np.array([True, False, True, False])
    assert paired_scores(orig, cand).tolist() == [0, -1, 1, 0]


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        SequentialBudgetTest(budget_pp=-1)
    with pytest.raises(ValueError):
        SequentialBudgetTest(alpha=1.5)
    with pytest.raises(ValueError):
        SequentialBudgetTest().update(2)


def test_no_decision_before_the_minimum_sample():
    test = SequentialBudgetTest(budget_pp=1.0, min_n=32, track_ci=False)
    for _ in range(31):
        assert test.update(-1) == "undecided"


def test_a_decision_is_final():
    test = SequentialBudgetTest(budget_pp=1.0, track_ci=False)
    for _ in range(200):
        test.update(-1)
    assert test.decision == "reject"
    n = test.n
    test.update(1)
    assert test.decision == "reject" and test.n == n


# ----- behaviour on clear cases -------------------------------------------


def test_a_clearly_broken_candidate_is_rejected_early():
    # Rates from ResNet-18 static INT8 with full-range per-channel weights: -4.2pp, ~17% of
    # images changing correctness.
    rng = np.random.default_rng(0)
    scores = draw(rng, 3925, p_fix=0.065, p_break=0.107)
    result = run_sequential(scores, budget_pp=1.0, track_ci=False)
    assert result.decision == "reject"
    assert result.n < 1500


def test_a_harmless_candidate_is_accepted():
    # Rates from Olive's static INT8 on ResNet-18: +0.33pp, ~3.7% discordant.
    rng = np.random.default_rng(1)
    scores = draw(rng, 3925, p_fix=0.020, p_break=0.017)
    result = run_sequential(scores, budget_pp=1.0, track_ci=False)
    assert result.decision == "accept"
    assert result.n < 3925


def test_an_identical_model_is_accepted_quickly():
    result = run_sequential([0] * 2000, budget_pp=1.0, track_ci=False)
    assert result.decision == "accept"
    assert result.n < 1000


# ----- the guarantee -------------------------------------------------------


@pytest.mark.parametrize("discordance", [0.04, 0.17])
def test_accept_error_is_controlled_when_loss_sits_exactly_on_the_budget(discordance):
    """True change = -1pp, budget = 1pp: accepting is an error. It must happen <= alpha."""
    alpha, reps, n = 0.05, 300, 3000
    delta = -0.01
    p_fix, p_break = (discordance + delta) / 2, (discordance - delta) / 2
    rng = np.random.default_rng(2)
    wrong = sum(
        run_sequential(draw(rng, n, p_fix, p_break), budget_pp=1.0, alpha=alpha,
                       track_ci=False).decision == "accept"
        for _ in range(reps)
    )
    # Binomial slack: 3 standard errors above alpha.
    assert wrong / reps <= alpha + 3 * (alpha * (1 - alpha) / reps) ** 0.5


def test_reject_error_is_controlled_when_loss_sits_exactly_on_the_budget():
    alpha, reps, n = 0.05, 300, 3000
    delta, discordance = -0.01, 0.10
    p_fix, p_break = (discordance + delta) / 2, (discordance - delta) / 2
    rng = np.random.default_rng(3)
    wrong = sum(
        run_sequential(draw(rng, n, p_fix, p_break), budget_pp=1.0, alpha=alpha,
                       track_ci=False).decision == "reject"
        for _ in range(reps)
    )
    assert wrong / reps <= alpha + 3 * (alpha * (1 - alpha) / reps) ** 0.5


def test_confidence_sequence_covers_the_true_change():
    alpha, reps, n = 0.05, 60, 800
    delta, discordance = -0.02, 0.12
    p_fix, p_break = (discordance + delta) / 2, (discordance - delta) / 2
    rng = np.random.default_rng(4)
    covered = 0
    for _ in range(reps):
        cs = ConfidenceSequence(alpha=alpha)
        for d in draw(rng, n, p_fix, p_break):
            cs.update(d)
        lo, hi = cs.interval()
        covered += lo <= delta <= hi
    assert covered / reps >= 1 - alpha - 3 * (alpha * (1 - alpha) / reps) ** 0.5


def test_confidence_sequence_narrows_with_data():
    cs = ConfidenceSequence()
    for _ in range(50):
        cs.update(0)
    wide = cs.interval()
    for _ in range(2000):
        cs.update(0)
    narrow = cs.interval()
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_result_reports_savings_and_describes_itself():
    result = run_sequential([0] * 2000, budget_pp=1.0)
    assert 0 < result.savings < 1
    assert "ACCEPT" in result.describe()
    assert result.to_dict()["n"] == result.n


def test_fixed_n_yardstick_grows_as_the_effect_approaches_the_budget():
    far = fixed_n_for_power(delta=-0.04, discordance=0.17, budget=0.01)
    near = fixed_n_for_power(delta=-0.015, discordance=0.17, budget=0.01)
    assert near > far > 0
