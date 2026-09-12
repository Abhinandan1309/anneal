"""The ledger: every trial the agent ran, and the Pareto frontier over them.

Two design choices worth stating.

**Failures are recorded, not swallowed.** A transform that produced an unloadable graph
is a trial with an error, and it stays in the ledger. It is signal for the agent (don't
try that again) and it is signal for the reader (this tool does not hide its misses).

**The answer is a frontier, not a winner.** "The best model" is not well defined across
latency, accuracy and size. Anneal returns the Pareto-optimal set and lets the engineer —
who knows whether they are bound by a 30ms deadline or 8MB of flash — pick the point.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from anneal.core.artifact import ModelArtifact
from anneal.core.measure import Measurement

Direction = Literal["min", "max"]

#: The objectives Anneal optimises, and which way is better.
OBJECTIVES: tuple[tuple[str, Direction], ...] = (
    ("latency_ms_p50", "min"),
    ("size_bytes", "min"),
    ("accuracy", "max"),
)


@dataclass
class Trial:
    """One proposal, applied and measured (or one that failed, and why)."""

    index: int
    artifact: ModelArtifact
    measurement: Measurement | None = None
    error: str | None = None
    proposer: str = "unknown"
    rationale: str = ""
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.measurement is not None and self.error is None

    @property
    def label(self) -> str:
        return self.artifact.label

    def objective(self, key: str) -> float | None:
        if self.measurement is None:
            return None
        value = getattr(self.measurement, key, None)
        return None if value is None else float(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "artifact": self.artifact.to_dict(),
            "measurement": self.measurement.to_dict() if self.measurement else None,
            "error": self.error,
            "proposer": self.proposer,
            "rationale": self.rationale,
            "duration_s": round(self.duration_s, 4),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trial:
        return cls(
            index=d["index"],
            artifact=ModelArtifact.from_dict(d["artifact"]),
            measurement=Measurement.from_dict(d["measurement"]) if d.get("measurement") else None,
            error=d.get("error"),
            proposer=d.get("proposer", "unknown"),
            rationale=d.get("rationale", ""),
            duration_s=d.get("duration_s", 0.0),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class Ledger:
    """An append-only record of one optimisation run."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    target_fingerprint: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    trials: list[Trial] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    # ----- writing --------------------------------------------------------

    def add(self, trial: Trial) -> Trial:
        self.trials.append(trial)
        return trial

    def next_index(self) -> int:
        return len(self.trials)

    # ----- reading --------------------------------------------------------

    @property
    def baseline(self) -> Trial | None:
        for t in self.trials:
            if not t.artifact.lineage and t.ok:
                return t
        return None

    def successful(self) -> list[Trial]:
        return [t for t in self.trials if t.ok]

    def attempted_recipes(self) -> set[str]:
        """Every lineage already tried, successful or not — the agent should not repeat."""
        return {t.artifact.lineage_key for t in self.trials}

    def scored(self) -> list[Trial]:
        """Trials with a real accuracy number attached."""
        return [t for t in self.successful() if t.measurement and t.measurement.accuracy is not None]

    # ----- selection ------------------------------------------------------

    def pareto_front(
        self,
        objectives: Sequence[tuple[str, Direction]] = OBJECTIVES,
        candidates: Iterable[Trial] | None = None,
    ) -> list[Trial]:
        """Non-dominated trials. A dominates B if it is at least as good on every
        objective and strictly better on at least one."""
        pool = [
            t
            for t in (self.successful() if candidates is None else candidates)
            if all(t.objective(k) is not None for k, _ in objectives)
        ]

        front: list[Trial] = []
        for cand in pool:
            dominated = False
            for other in pool:
                if other is cand:
                    continue
                if _dominates(other, cand, objectives):
                    dominated = True
                    break
            if not dominated:
                front.append(cand)

        # Deduplicate recipes that landed on identical objective vectors.
        seen: set[tuple[float, ...]] = set()
        unique: list[Trial] = []
        for t in sorted(front, key=lambda x: x.objective(objectives[0][0]) or 0.0):
            key = tuple(round(t.objective(k) or 0.0, 6) for k, _ in objectives)
            if key in seen:
                continue
            seen.add(key)
            unique.append(t)
        return unique

    def best_under_constraints(
        self,
        *,
        min_accuracy: float | None = None,
        max_size_bytes: int | None = None,
        max_latency_ms: float | None = None,
        optimise: str = "latency_ms_p50",
        direction: Direction = "min",
    ) -> Trial | None:
        """The single best trial that satisfies hard constraints.

        This is the question an engineer actually asks: *given that I cannot lose more
        than 1% accuracy, what is the fastest thing you found?*
        """
        pool = self.successful()
        if min_accuracy is not None:
            pool = [
                t
                for t in pool
                if t.measurement and t.measurement.accuracy is not None
                and t.measurement.accuracy >= min_accuracy
            ]
        if max_size_bytes is not None:
            pool = [t for t in pool if t.measurement and t.measurement.size_bytes <= max_size_bytes]
        if max_latency_ms is not None:
            pool = [
                t
                for t in pool
                if t.measurement and t.measurement.latency_ms_p50 <= max_latency_ms
            ]
        if not pool:
            return None
        key = lambda t: t.objective(optimise) or 0.0  # noqa: E731
        return min(pool, key=key) if direction == "min" else max(pool, key=key)

    def speedup_of(self, trial: Trial) -> float | None:
        """Measured p50 speedup versus the baseline trial."""
        base = self.baseline
        if base is None or not trial.ok or base.measurement is None or trial.measurement is None:
            return None
        if trial.measurement.latency_ms_p50 <= 0:
            return None
        return base.measurement.latency_ms_p50 / trial.measurement.latency_ms_p50

    # ----- persistence ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "target_fingerprint": self.target_fingerprint,
            "config": self.config,
            "trials": [t.to_dict() for t in self.trials],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Ledger:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            run_id=d["run_id"],
            started_at=d.get("started_at", 0.0),
            target_fingerprint=d.get("target_fingerprint", {}),
            config=d.get("config", {}),
            trials=[Trial.from_dict(t) for t in d.get("trials", [])],
        )


def _dominates(a: Trial, b: Trial, objectives: Sequence[tuple[str, Direction]]) -> bool:
    at_least_as_good = True
    strictly_better = False
    for key, direction in objectives:
        av, bv = a.objective(key), b.objective(key)
        if av is None or bv is None:
            return False
        if direction == "min":
            if av > bv:
                at_least_as_good = False
                break
            if av < bv:
                strictly_better = True
        else:
            if av < bv:
                at_least_as_good = False
                break
            if av > bv:
                strictly_better = True
    return at_least_as_good and strictly_better
