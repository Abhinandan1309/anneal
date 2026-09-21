"""Sequential, anytime-valid acceptance testing for an optimised model.

The problem this solves: evaluation is the expensive part of every trial, and the cheap
fixed-size alternative is unreliable — on ResNet-18 a 256-image eval set understated static
INT8's accuracy loss by more than a third. Checking a fixed-sample significance test after
every batch and stopping once it "looks decided" is not a fix: that inflates the error rate
far beyond the nominal α.

Anytime-valid tests make early stopping legitimate. Each image contributes a paired score

    d = +1   the candidate fixed an image the original got wrong
    d = -1   the candidate broke an image the original got right
    d =  0   no change in correctness

whose mean μ is exactly the accuracy change. Two *wealth processes* bet against the two
hypotheses about the budget m = -δ (δ = the accuracy the engineer is willing to lose):

    reject test   null μ >= m  (within budget)    wealth ∏(1 + λ(m - d))
    accept test   null μ <= m  (over budget)      wealth ∏(1 + λ(d - m))

Under its null each wealth is a non-negative supermartingale, so by Ville's inequality it
ever reaches 1/α with probability at most α — at *any* stopping time. Stopping the moment
either crosses therefore keeps each decision's error below α however early it happens.
Bets are sized by the approximate growth-rate-adaptive (aGRAPA) rule of Waudby-Smith &
Ramdas, "Estimating means of bounded random variables by betting" (JRSS-B, 2023).

**Images must arrive in random order.** The guarantee assumes exchangeable scores; feeding
images sorted by class would void it. Anneal's eval sets are deterministically shuffled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

Decision = Literal["accept", "reject", "undecided"]

#: Fraction of the maximum admissible bet actually used. Betting the maximum lets one bad
#: draw wipe out the wealth; half keeps every factor >= 0.5.
BET_FRACTION = 0.5


@dataclass
class _Bettor:
    """One-sided betting test that μ is on the far side of ``m``.

    ``direction=+1`` bets that μ > m (wealth grows on d > m); ``-1`` bets that μ < m.
    """

    m: float
    direction: int
    wealth: float = 1.0
    #: Regularised running moments; the prior pulls early bets towards zero.
    _sum: float = 0.0
    _sumsq: float = 0.0
    _n: int = 0

    def _bet(self) -> float:
        mean = (self.m + self._sum) / (self._n + 1)
        var = (0.25 + self._sumsq - (self._n + 1) * mean**2 + self.m**2) / (self._n + 1)
        var = max(var, 1e-6)
        edge = self.direction * (mean - self.m)
        cap = BET_FRACTION / ((1 + self.m) if self.direction > 0 else (1 - self.m))
        return min(max(edge / (var + edge**2), 0.0), cap)

    def update(self, d: float) -> None:
        lam = self._bet()
        self.wealth *= 1.0 + lam * self.direction * (d - self.m)
        self._sum += d
        self._sumsq += d * d
        self._n += 1


@dataclass
class SequentialResult:
    decision: Decision
    n: int
    n_max: int
    regressions: int
    fixes: int
    delta_pp: float
    budget_pp: float
    alpha: float
    ci_low_pp: float
    ci_high_pp: float
    trace: list[tuple[int, float, float]] = field(default_factory=list)

    @property
    def savings(self) -> float:
        """Fraction of the eval set that did not need to be evaluated."""
        return 1.0 - self.n / self.n_max if self.n_max else 0.0

    def describe(self) -> str:
        head = {
            "accept": f"ACCEPT: loss is within the {self.budget_pp:.2f}pp budget",
            "reject": f"REJECT: loss exceeds the {self.budget_pp:.2f}pp budget",
            "undecided": "UNDECIDED: the true change is too close to the budget to call",
        }[self.decision]
        return (
            f"{head} after {self.n} of {self.n_max} images ({self.savings * 100:.0f}% saved). "
            f"Estimated change {self.delta_pp:+.2f}pp; anytime-valid {100 * (1 - self.alpha):.0f}% "
            f"interval [{self.ci_low_pp:+.2f}, {self.ci_high_pp:+.2f}]pp. "
            f"Each decision's error rate is at most {self.alpha:g}, however early it stops."
        )

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "n": self.n,
            "n_max": self.n_max,
            "savings": self.savings,
            "regressions": self.regressions,
            "fixes": self.fixes,
            "delta_pp": self.delta_pp,
            "budget_pp": self.budget_pp,
            "alpha": self.alpha,
            "anytime_ci_pp": [self.ci_low_pp, self.ci_high_pp],
        }


class ConfidenceSequence:
    """Anytime-valid confidence interval for μ, from betting tests over a grid of values.

    A value m stays in the set until one of its two one-sided bettors (each at α/2) wins.
    The running intersection is kept, so the interval only ever shrinks.
    """

    def __init__(self, alpha: float = 0.05, lo: float = -0.5, hi: float = 0.5, step: float = 0.0025):
        self.alpha = alpha
        self.grid = np.arange(lo, hi + step / 2, step)
        g = self.grid
        # Log-wealth: a grid value far from the truth accumulates wealth past float range
        # within a few thousand images, and log1p of a factor >= 0.5 never overflows.
        self._log_up = np.zeros_like(g)
        self._log_down = np.zeros_like(g)
        self._sum = 0.0
        self._sumsq = 0.0
        self._n = 0
        self.alive = np.ones_like(g, dtype=bool)
        self._cap_up = BET_FRACTION / (1 + g)
        self._cap_down = BET_FRACTION / (1 - g)

    def update(self, d: float) -> None:
        g = self.grid
        mean = (g + self._sum) / (self._n + 1)
        var = np.maximum((0.25 + self._sumsq - (self._n + 1) * mean**2 + g**2) / (self._n + 1), 1e-6)
        edge_up = mean - g
        lam_up = np.clip(edge_up / (var + edge_up**2), 0.0, self._cap_up)
        lam_down = np.clip(-edge_up / (var + edge_up**2), 0.0, self._cap_down)
        self._log_up += np.log1p(lam_up * (d - g))
        self._log_down += np.log1p(lam_down * (g - d))
        self._sum += d
        self._sumsq += d * d
        self._n += 1
        threshold = np.log(2.0 / self.alpha)
        self.alive &= (self._log_up < threshold) & (self._log_down < threshold)

    def interval(self) -> tuple[float, float]:
        if not self.alive.any():  # pragma: no cover - only with a grid too narrow for the data
            return (float("nan"), float("nan"))
        live = self.grid[self.alive]
        return float(live.min()), float(live.max())


class SequentialBudgetTest:
    """Decide, as early as the evidence allows, whether an accuracy loss is within budget.

    Feed paired scores one image at a time with :meth:`update`; stop when :attr:`decision`
    is no longer ``"undecided"``.
    """

    def __init__(
        self, budget_pp: float = 1.0, alpha: float = 0.05, min_n: int = 32, track_ci: bool = True
    ):
        if budget_pp < 0:
            raise ValueError("budget_pp must be >= 0")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.m = -budget_pp / 100.0
        self.budget_pp = budget_pp
        self.alpha = alpha
        #: A floor on n before any decision: guards against a verdict from a handful of images
        #: that happen to agree, at no cost to validity.
        self.min_n = min_n
        self._accept = _Bettor(self.m, +1)
        self._reject = _Bettor(self.m, -1)
        self._cs = ConfidenceSequence(alpha=alpha) if track_ci else None
        self.n = 0
        self.regressions = 0
        self.fixes = 0
        self.decision: Decision = "undecided"
        self.trace: list[tuple[int, float, float]] = []

    def update(self, d: int) -> Decision:
        if d not in (-1, 0, 1):
            raise ValueError("paired score must be -1, 0 or +1")
        if self.decision != "undecided":
            return self.decision
        self._accept.update(d)
        self._reject.update(d)
        if self._cs is not None:
            self._cs.update(d)
        self.n += 1
        self.regressions += d == -1
        self.fixes += d == 1
        if self.n % 16 == 0:
            self.trace.append((self.n, self._accept.wealth, self._reject.wealth))
        if self.n >= self.min_n:
            if self._reject.wealth >= 1 / self.alpha:
                self.decision = "reject"
            elif self._accept.wealth >= 1 / self.alpha:
                self.decision = "accept"
        return self.decision

    def result(self, n_max: int) -> SequentialResult:
        lo, hi = self._cs.interval() if self._cs is not None else (float("nan"), float("nan"))
        return SequentialResult(
            decision=self.decision,
            n=self.n,
            n_max=n_max,
            regressions=int(self.regressions),
            fixes=int(self.fixes),
            delta_pp=(self.fixes - self.regressions) / max(self.n, 1) * 100,
            budget_pp=self.budget_pp,
            alpha=self.alpha,
            ci_low_pp=lo * 100,
            ci_high_pp=hi * 100,
            trace=list(self.trace),
        )


def paired_scores(original_right: np.ndarray, candidate_right: np.ndarray) -> np.ndarray:
    """Per-image score: +1 fixed, -1 broken, 0 unchanged."""
    return candidate_right.astype(np.int8) - original_right.astype(np.int8)


def run_sequential(
    scores: Iterable[int],
    *,
    budget_pp: float = 1.0,
    alpha: float = 0.05,
    n_max: int | None = None,
    track_ci: bool = True,
) -> SequentialResult:
    """Run the test over an already-computed score stream, stopping as soon as it decides."""
    scores = list(scores)
    test = SequentialBudgetTest(budget_pp=budget_pp, alpha=alpha, track_ci=track_ci)
    for d in scores:
        if test.update(int(d)) != "undecided":
            break
    return test.result(n_max or len(scores))


def fixed_n_for_power(delta: float, discordance: float, budget: float, alpha: float = 0.05,
                      power: float = 0.9) -> int:
    """Sample size a classical one-sided paired z-test needs for the same decision.

    ``delta`` is the true mean change (fraction), ``discordance`` the fraction of images whose
    correctness differs between the models, ``budget`` the margin as a fraction. Used only as
    the yardstick the sequential test is compared against.
    """
    from statistics import NormalDist

    gap = abs(delta + budget)
    if gap == 0:
        return math.inf  # type: ignore[return-value]
    sd = math.sqrt(max(discordance - delta**2, 1e-12))
    z = NormalDist().inv_cdf
    return math.ceil(((z(1 - alpha) + z(power)) * sd / gap) ** 2)
