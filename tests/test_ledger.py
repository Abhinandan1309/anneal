"""Pareto dominance, constraint selection, and round-tripping the ledger."""

from __future__ import annotations

from pathlib import Path

from anneal.core.ledger import Ledger, _dominates
from conftest import make_trial


def test_baseline_is_the_lineage_free_trial(ledger):
    assert ledger.baseline is not None
    assert ledger.baseline.index == 0
    assert ledger.baseline.artifact.lineage == ()


def test_dominance_requires_strictly_better_on_something():
    a = make_trial(0, latency=20.0, size_bytes=10, accuracy=0.9)
    identical = make_trial(1, latency=20.0, size_bytes=10, accuracy=0.9)
    faster = make_trial(2, latency=10.0, size_bytes=10, accuracy=0.9)

    objectives = (("latency_ms_p50", "min"), ("size_bytes", "min"), ("accuracy", "max"))
    assert not _dominates(a, identical, objectives)
    assert _dominates(faster, a, objectives)
    assert not _dominates(a, faster, objectives)


def test_pareto_front_excludes_dominated_trials(ledger):
    front = {t.index for t in ledger.pareto_front()}
    # Trial 3 is slower than 1, bigger than 2, and less accurate than both: dominated.
    assert 3 not in front
    # Trial 2 is fastest and smallest; trial 0/1 hold the accuracy high ground.
    assert 2 in front


def test_pareto_front_is_sorted_by_first_objective(ledger):
    front = ledger.pareto_front()
    latencies = [t.measurement.latency_ms_p50 for t in front]
    assert latencies == sorted(latencies)


def test_pareto_front_skips_trials_missing_an_objective(ledger, q8):
    ledger.add(make_trial(4, lineage=(q8,), latency=1.0, size_bytes=1, accuracy=None))
    assert 4 not in {t.index for t in ledger.pareto_front()}


def test_failed_trials_stay_in_the_ledger_but_out_of_the_front(ledger, q8):
    ledger.add(make_trial(4, lineage=(q8,), error="graph would not load"))
    assert len(ledger.trials) == 5
    assert len(ledger.successful()) == 4
    assert 4 not in {t.index for t in ledger.pareto_front()}


def test_best_under_constraints_respects_accuracy_floor(ledger):
    # Trial 2 is fastest at 20ms but only 87% accurate.
    assert ledger.best_under_constraints().index == 2
    assert ledger.best_under_constraints(min_accuracy=0.88).index == 1
    assert ledger.best_under_constraints(min_accuracy=0.99) is None


def test_best_under_constraints_respects_size_budget(ledger):
    pick = ledger.best_under_constraints(max_size_bytes=12_000_000)
    assert pick is not None and pick.index == 2


def test_speedup_is_relative_to_baseline(ledger):
    assert ledger.speedup_of(ledger.trials[2]) == 2.0
    assert ledger.speedup_of(ledger.trials[0]) == 1.0


def test_attempted_recipes_deduplicates_by_lineage(ledger):
    recipes = ledger.attempted_recipes()
    assert "baseline" in recipes
    assert "graph_optimize(level=all)" in recipes
    assert len(recipes) == 4


def test_save_load_round_trip(ledger, tmp_path: Path):
    path = ledger.save(tmp_path / "ledger.json")
    restored = Ledger.load(path)

    assert restored.run_id == ledger.run_id
    assert len(restored.trials) == len(ledger.trials)
    assert restored.baseline is not None
    assert restored.trials[2].measurement.latency_ms_p50 == 20.0
    assert restored.trials[2].artifact.label == ledger.trials[2].artifact.label
    assert {t.index for t in restored.pareto_front()} == {t.index for t in ledger.pareto_front()}
