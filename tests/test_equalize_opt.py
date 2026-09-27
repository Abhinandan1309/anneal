from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_equalize import (
    _batches,
    _chained_residual_model,
    _model,
    _residual_model,
    _run,
)

from anneal.core.equalize import equalise
from anneal.core.equalize_opt import (
    Term,
    _nelder_mead,
    _slope_sq,
    objective,
    optimal_scales,
)


def _terms(seed: int, c: int = 8) -> list[Term]:
    rng = np.random.default_rng(seed)
    return [Term("y", np.exp(rng.normal(0, 2, c)), np.exp(rng.normal(0, 1, c)), True),
            Term("x", np.exp(rng.normal(0, 2, c)), np.exp(rng.normal(0, 1, c)), True),
            Term("A", np.exp(rng.normal(0, 1, c)), np.exp(rng.normal(-1, 1, c)), True),
            Term("B", np.exp(rng.normal(0, 1.5, c)), np.exp(rng.normal(0, 1, c)), False)]


def test_objective_is_scale_invariant():
    t = _terms(0)
    s = np.exp(np.random.default_rng(1).normal(0, 1, 8))
    assert objective(t, s) == pytest.approx(objective(t, 7.3 * s), rel=1e-9)


def test_three_channels_match_a_grid_search():
    t = [Term("y", np.array([1.0, 10.0, 0.1]), np.ones(3), True),
         Term("A", np.array([1.0, 0.5, 2.0]), np.full(3, 0.3), True),
         Term("B", np.array([2.0, 0.1, 5.0]), np.full(3, 0.7), False)]
    s = optimal_scales(t, np.ones(3), max_spread=1e9)
    g = np.exp(np.linspace(-6, 6, 241))
    grid = min(objective(t, np.array([1.0, a, b])) for a in g for b in g)
    assert objective(t, s) <= grid * (1 + 1e-4)


@pytest.mark.parametrize("seed", range(6))
def test_no_direct_search_beats_it(seed):
    t = _terms(seed)
    ours = objective(t, optimal_scales(t, np.ones(8), max_spread=1e9))
    rng = np.random.default_rng(100 + seed)

    def f(z):
        return objective(t, np.exp(np.concatenate([[0.0], z])))

    direct = min(f(_nelder_mead(f, rng.normal(0, 2, 7), step=1.0, iters=3000)) for _ in range(4))
    assert ours <= direct * (1 + 1e-3)


def test_never_worse_than_its_start():
    t = _terms(3)
    s0 = np.exp(np.random.default_rng(4).normal(0, 2, 8))
    assert objective(t, optimal_scales(t, s0)) <= objective(t, s0) * (1 + 1e-12)


def test_without_divided_terms_it_keeps_the_start():
    t = [x for x in _terms(5) if x.up]
    s0 = np.arange(1.0, 9.0)
    np.testing.assert_allclose(optimal_scales(t, s0), s0 / s0.min())


def test_slopes():
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    np.testing.assert_allclose(_slope_sq(x, "relu"), [0, 0, 0, 1, 1])
    np.testing.assert_allclose(_slope_sq(x, "hswish"), [0, (1 / 6) ** 2, 0.25, (5 / 6) ** 2, 1])
    assert _slope_sq(np.array([0.0]), "silu")[0] == pytest.approx(0.25)


@pytest.mark.parametrize("builder", ["gated", "hswish", "residual", "chained"])
def test_derived_equalisation_stays_exact(tmp_path: Path, builder: str):
    src = {"gated": lambda p: _model(p), "hswish": lambda p: _model(p, activation="hardswish"),
           "residual": lambda p: _residual_model(p), "chained": lambda p: _chained_residual_model(p)}[builder](
        tmp_path / "m.onnx")
    dst = tmp_path / "eq.onnx"
    result = equalise(src, dst, _batches(), residual=True, se=True, mix=(0.5, 0.5), derived=True)
    assert result.sites
    x = _batches(1, seed=7)[0]
    before, after = _run(src, x), _run(dst, x)
    assert np.abs(after - before).max() <= 1e-4 * max(1.0, np.abs(before).max())
