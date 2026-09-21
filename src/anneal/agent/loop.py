"""The optimisation loop.

propose -> apply -> **measure** -> record -> repeat.

The loop is deliberately thin. All the intelligence lives in the policy, all the rigour
lives in the benchmarker, and this module's only jobs are to keep them honest about
budget, to make sure a failing transform degrades into a recorded trial rather than a
crashed run, and to refuse to count the same recipe twice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from anneal.agent.policy import BASELINE, Constraints, Policy, Proposal, SearchState
from anneal.core import environment
from anneal.core.artifact import ModelArtifact
from anneal.core.ledger import Ledger, Trial
from anneal.core.measure import Benchmarker, EvalSet, MeasurementError
from anneal.core.targets import Target
from anneal.core.transforms import TransformContext, apply_transform, available_transforms

#: Give up if the policy proposes this many already-tried recipes back to back.
MAX_CONSECUTIVE_REJECTIONS = 4

EventHandler = Callable[[str, dict[str, Any]], None]


@dataclass
class RunConfig:
    """Everything that controls one run, and nothing that controls the model."""

    workdir: Path
    budget: int = 12
    batch_size: int = 1
    warmup: int = 10
    runs: int = 60
    calib_samples: int = 64
    seed: int = 0
    constraints: Constraints = field(default_factory=Constraints)
    #: Layer order from a saved `anneal sensitivity --measured` sweep, most damaging
    #: first. Saves re-running a sweep that costs one eval pass per layer.
    measured_ranking: list[str] | None = None
    measured_ranking_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "batch_size": self.batch_size,
            "warmup": self.warmup,
            "runs": self.runs,
            "calib_samples": self.calib_samples,
            "seed": self.seed,
            "sensitivity_source": self.measured_ranking_source,
            "constraints": {
                "max_accuracy_drop_pp": self.constraints.max_accuracy_drop_pp,
                "min_accuracy": self.constraints.min_accuracy,
                "max_size_bytes": self.constraints.max_size_bytes,
                "max_latency_ms": self.constraints.max_latency_ms,
            },
        }


class OptimizationRun:
    """Drives a policy against a target until the budget or the ideas run out."""

    def __init__(
        self,
        baseline: ModelArtifact,
        target: Target,
        policy: Policy,
        config: RunConfig,
        *,
        evalset: EvalSet | None = None,
        calibset: EvalSet | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        self.baseline = baseline
        self.target = target
        self.policy = policy
        self.config = config
        self.evalset = evalset
        self._emit = on_event or (lambda kind, payload: None)

        self.workdir = Path(config.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.benchmarker = Benchmarker(
            target,
            batch_size=config.batch_size,
            warmup=config.warmup,
            runs=config.runs,
            seed=config.seed,
        )
        self.ctx = TransformContext(
            workdir=self.workdir / "candidates",
            evalset=evalset,
            calibset=calibset,
            calib_samples=config.calib_samples,
        )
        if config.measured_ranking:
            self.ctx.extra["measured_ranking"] = list(config.measured_ranking)
        self.can_measure_sensitivity = evalset is not None or bool(config.measured_ranking)
        self.transforms = available_transforms()

        self.ledger = Ledger(
            target_fingerprint=target.fingerprint(),
            config={
                **config.to_dict(),
                "policy": getattr(policy, "name", "unknown"),
                "baseline_model": baseline.meta.get("source", str(baseline.path.name)),
                "evalset": getattr(evalset, "name", None),
                "evalset_size": len(evalset) if evalset is not None else 0,
                "evalset_synthetic": getattr(evalset, "synthetic", None),
                "calibration_set": (
                    f"{getattr(calibset, 'name', '?')} {getattr(calibset, 'split', '')}".strip()
                    if calibset is not None
                    else ("eval set (overlapping)" if evalset is not None else None)
                ),
                "transforms_available": sorted(self.transforms),
            },
        )
        start_env = environment.snapshot()
        self.ledger.config["environment"] = {
            "start": start_env,
            "warnings": environment.warnings_for(start_env),
        }

    # ----- main -----------------------------------------------------------

    def run(self) -> Ledger:
        self._measure_baseline()

        rejections = 0
        while self._spent() < self.config.budget:
            state = SearchState(
                ledger=self.ledger,
                transforms=self.transforms,
                budget_remaining=self.config.budget - self._spent(),
                constraints=self.config.constraints,
                baseline_artifact=self.baseline,
                can_measure_sensitivity=self.can_measure_sensitivity,
            )

            proposal = self.policy.propose(state)
            if proposal is None:
                reason = getattr(self.policy, "stop_reason", "") or "policy stopped"
                self._emit("stopped", {"reason": reason})
                break

            outcome = self._run_trial(proposal, state)
            if outcome == "duplicate":
                rejections += 1
                if rejections >= MAX_CONSECUTIVE_REJECTIONS:
                    self._emit(
                        "stopped",
                        {"reason": "policy kept proposing recipes already in the ledger"},
                    )
                    break
                continue
            rejections = 0

        self._check_drift()
        self._emit("finished", {"trials": len(self.ledger.trials)})
        return self.ledger

    def _check_drift(self) -> None:
        """Re-measure the baseline: if it moved, no latency in this run is comparable.

        A laptop that drops into battery saver halfway through a search makes every later
        trial look slow. Timing the same model at both ends is the cheapest way to notice.
        """
        base = self.ledger.baseline
        if base is None or base.measurement is None:
            return
        try:
            again = self.benchmarker.measure(self.baseline)
        except MeasurementError:
            return
        start = base.measurement.latency_ms_p50
        end = again.latency_ms_p50
        moved = environment.drift(start, end)
        end_env = environment.snapshot()
        env = self.ledger.config.setdefault("environment", {})
        env["end"] = end_env
        env["warnings"] = sorted(
            set(env.get("warnings", [])) | set(environment.warnings_for(end_env))
        )
        env["baseline_p50_start_ms"] = start
        env["baseline_p50_end_ms"] = end
        env["baseline_drift"] = moved
        env["latency_trustworthy"] = bool(
            moved <= environment.DRIFT_TOLERANCE and not env["warnings"]
        )
        self._emit("drift", {"start_ms": start, "end_ms": end, "drift": moved,
                             "warnings": env["warnings"]})

    # ----- steps ----------------------------------------------------------

    def _measure_baseline(self) -> None:
        self._emit("baseline_start", {"model": str(self.baseline.path.name)})
        started = time.perf_counter()
        try:
            measurement = self.benchmarker.measure(
                self.baseline, evalset=self.evalset, record_baseline=True
            )
            trial = Trial(
                index=0,
                artifact=self.baseline,
                measurement=measurement,
                proposer="harness",
                rationale="Unmodified model. Every other number is relative to this one.",
                duration_s=time.perf_counter() - started,
            )
        except MeasurementError as exc:
            trial = Trial(
                index=0,
                artifact=self.baseline,
                error=str(exc),
                proposer="harness",
                duration_s=time.perf_counter() - started,
            )
        self.ledger.add(trial)
        self._emit("trial", {"trial": trial, "ledger": self.ledger})

        if not trial.ok:
            raise RuntimeError(
                f"the baseline model could not be measured, so nothing can be compared "
                f"against it: {trial.error}"
            )

    def _run_trial(self, proposal: Proposal, state: SearchState) -> str:
        index = self.ledger.next_index()
        started = time.perf_counter()

        base = state.artifact(proposal.base_index)
        if base is None:
            self._record_failure(
                index,
                self.baseline,
                proposal,
                f"proposal referenced trial {proposal.base_index}, which is not a "
                f"successful trial in this ledger",
                time.perf_counter() - started,
            )
            return "failed"

        self._emit(
            "proposal",
            {
                "index": index,
                "transform": proposal.transform,
                "params": proposal.params,
                "base_index": proposal.base_index,
                "base_label": base.label,
                "rationale": proposal.rationale,
            },
        )

        try:
            candidate = apply_transform(proposal.transform, proposal.params, base, self.ctx)
        except Exception as exc:
            # Transforms call into vendor quantization code that raises a wide and
            # undocumented range of exception types. A transform that blows up is a fact
            # about this model and target, so it becomes a recorded trial rather than
            # ending the run. KeyboardInterrupt is not an Exception and still propagates.
            self._record_failure(
                index, base, proposal, f"{type(exc).__name__}: {exc}", time.perf_counter() - started
            )
            return "failed"

        if candidate.lineage_key in self.ledger.attempted_recipes():
            self._emit(
                "duplicate",
                {"recipe": candidate.lineage_key, "transform": proposal.transform},
            )
            return "duplicate"

        try:
            measurement = self.benchmarker.measure(candidate, evalset=self.evalset)
        except MeasurementError as exc:
            self._record_failure(
                index, candidate, proposal, str(exc), time.perf_counter() - started
            )
            return "failed"

        trial = self.ledger.add(
            Trial(
                index=index,
                artifact=candidate,
                measurement=measurement,
                proposer=getattr(self.policy, "name", "unknown"),
                rationale=proposal.rationale,
                duration_s=time.perf_counter() - started,
            )
        )
        self._emit("trial", {"trial": trial, "ledger": self.ledger})
        return "ok"

    def _record_failure(
        self,
        index: int,
        artifact: ModelArtifact,
        proposal: Proposal,
        error: str,
        duration: float,
    ) -> None:
        """A failed transform is data, not an exception. It stays in the ledger."""
        from anneal.core.artifact import TransformRecord

        failed_artifact = ModelArtifact(
            path=artifact.path,
            lineage=artifact.lineage + (TransformRecord(proposal.transform, proposal.params),),
            meta={**artifact.meta, "failed": True},
        )
        trial = self.ledger.add(
            Trial(
                index=index,
                artifact=failed_artifact,
                error=error,
                proposer=getattr(self.policy, "name", "unknown"),
                rationale=proposal.rationale,
                duration_s=duration,
            )
        )
        self._emit("trial", {"trial": trial, "ledger": self.ledger})

    def _spent(self) -> int:
        """Budget consumed: every trial after the baseline, successful or not."""
        return max(0, len(self.ledger.trials) - 1)
