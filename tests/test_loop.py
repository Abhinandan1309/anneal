"""The optimisation loop end to end, on a tiny real graph."""

from __future__ import annotations

from pathlib import Path

from anneal.agent.loop import OptimizationRun, RunConfig
from anneal.agent.policy import HeuristicPolicy
from anneal.core.artifact import ModelArtifact
from anneal.core.dataset import SyntheticEvalSet
from anneal.core.targets import get_target


def test_a_run_records_its_environment_and_rechecks_the_baseline(tiny_onnx: Path, tmp_path: Path):
    events = []
    run = OptimizationRun(
        ModelArtifact(path=tiny_onnx),
        get_target("cpu-1t"),
        HeuristicPolicy(),
        RunConfig(workdir=tmp_path / "run", budget=2, warmup=1, runs=3),
        evalset=SyntheticEvalSet(shape=(3, 16, 16), n=16, batch_size=8),
        calibset=SyntheticEvalSet(shape=(3, 16, 16), n=16, batch_size=8, seed=1),
        on_event=lambda kind, payload: events.append(kind),
    )
    ledger = run.run()

    env = ledger.config["environment"]
    assert "start" in env and "end" in env
    assert env["baseline_drift"] >= 0
    assert isinstance(env["latency_trustworthy"], bool)
    assert "drift" in events
    assert ledger.config["calibration_set"] == "synthetic"
    assert len(ledger.trials) == 3  # baseline + budget
