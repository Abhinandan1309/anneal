"""Operator-level attribution: *why* is this graph slow?

A benchmark tells you a model takes 627ms. It does not tell you that 94% of that is one
operator type that fell off the optimised kernel path — and without that, "INT8 made it
slower" is a superstition rather than a finding.

onnxruntime can emit a Chrome-trace JSON with per-node kernel times. This module turns
that into attribution by op type and by node, and can diff two models so the regression
points at itself.

One caveat, stated loudly because it matters: profiling instrumentation adds per-node
overhead, so the *absolute* times here run higher than the numbers :mod:`anneal.core.measure`
reports. Use this for relative attribution — which op dominates — and use the benchmarker
for latency.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anneal.core.artifact import ModelArtifact, concrete_input_shape, describe_io
from anneal.core.targets import Target

#: onnxruntime names per-node timing events "<node>_kernel_time".
_KERNEL_SUFFIX = "_kernel_time"


@dataclass(frozen=True)
class OpStat:
    """Aggregate cost of one operator type."""

    op_type: str
    total_us: float
    count: int
    share: float

    @property
    def mean_us(self) -> float:
        return self.total_us / self.count if self.count else 0.0


@dataclass(frozen=True)
class NodeStat:
    """Cost of one individual node."""

    name: str
    op_type: str
    total_us: float
    count: int
    share: float


@dataclass(frozen=True)
class Profile:
    """Per-operator attribution for one model on one target."""

    label: str
    total_us: float
    runs: int
    by_op: tuple[OpStat, ...]
    by_node: tuple[NodeStat, ...]
    providers_used: tuple[str, ...]
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_ms_per_run(self) -> float:
        """Instrumented time per run. Higher than the true latency; see module docstring."""
        return self.total_us / self.runs / 1000.0 if self.runs else 0.0

    def op(self, op_type: str) -> OpStat | None:
        return next((o for o in self.by_op if o.op_type == op_type), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "total_us": self.total_us,
            "runs": self.runs,
            "mean_ms_per_run": round(self.mean_ms_per_run, 3),
            "providers_used": list(self.providers_used),
            "by_op": [
                {
                    "op_type": o.op_type,
                    "total_us": o.total_us,
                    "count": o.count,
                    "share": round(o.share, 5),
                }
                for o in self.by_op
            ],
            "by_node": [
                {
                    "name": n.name,
                    "op_type": n.op_type,
                    "total_us": n.total_us,
                    "share": round(n.share, 5),
                }
                for n in self.by_node
            ],
            "notes": self.notes,
        }


def parse_profile_events(events: list[dict[str, Any]], label: str, runs: int) -> Profile:
    """Turn onnxruntime's trace events into attribution. Pure, so it is testable."""
    op_total: dict[str, float] = defaultdict(float)
    op_count: dict[str, int] = defaultdict(int)
    node_total: dict[str, float] = defaultdict(float)
    node_count: dict[str, int] = defaultdict(int)
    node_op: dict[str, str] = {}

    for event in events:
        if event.get("cat") != "Node":
            continue
        name = event.get("name", "")
        if not name.endswith(_KERNEL_SUFFIX):
            continue

        node = name[: -len(_KERNEL_SUFFIX)]
        duration = float(event.get("dur", 0.0))
        op_type = str((event.get("args") or {}).get("op_name", "Unknown"))

        op_total[op_type] += duration
        op_count[op_type] += 1
        node_total[node] += duration
        node_count[node] += 1
        node_op[node] = op_type

    total = sum(op_total.values())

    by_op = tuple(
        sorted(
            (
                OpStat(
                    op_type=op,
                    total_us=us,
                    count=op_count[op],
                    share=(us / total) if total else 0.0,
                )
                for op, us in op_total.items()
            ),
            key=lambda o: o.total_us,
            reverse=True,
        )
    )

    by_node = tuple(
        sorted(
            (
                NodeStat(
                    name=node,
                    op_type=node_op[node],
                    total_us=us,
                    count=node_count[node],
                    share=(us / total) if total else 0.0,
                )
                for node, us in node_total.items()
            ),
            key=lambda n: n.total_us,
            reverse=True,
        )
    )

    return Profile(
        label=label,
        total_us=total,
        runs=runs,
        by_op=by_op,
        by_node=by_node,
        providers_used=(),
    )


def profile_model(
    artifact: ModelArtifact,
    target: Target,
    *,
    runs: int = 20,
    warmup: int = 5,
    batch_size: int = 1,
    seed: int = 0,
    workdir: Path | None = None,
) -> Profile:
    """Run the model under onnxruntime's profiler and attribute time to operators."""
    import numpy as np
    import onnxruntime as ort

    target.ensure_available()

    workdir = Path(workdir) if workdir else artifact.path.parent
    workdir.mkdir(parents=True, exist_ok=True)

    opts = target.session_options()
    opts.enable_profiling = True
    # Keep the trace next to the candidate rather than in the process CWD.
    opts.profile_file_prefix = str(workdir / f"prof_{artifact.path.stem}")

    session = ort.InferenceSession(
        str(artifact.path), sess_options=opts, providers=list(target.providers)
    )

    io = describe_io(artifact.path)
    spec = io["inputs"][0]
    shape = concrete_input_shape(spec, batch_size)
    rng = np.random.default_rng(seed)
    feed = {spec["name"]: rng.standard_normal(shape, dtype=np.float32)}
    output_names = [o.name for o in session.get_outputs()]

    for _ in range(warmup):
        session.run(output_names, feed)
    for _ in range(runs):
        session.run(output_names, feed)

    providers_used = tuple(session.get_providers())
    trace_path = Path(session.end_profiling())
    del session

    try:
        events = json.loads(trace_path.read_text(encoding="utf-8"))
    finally:
        trace_path.unlink(missing_ok=True)

    # The warm-up runs are in the trace too. They are a small fraction of the total and
    # removing them would require matching events to iterations, which onnxruntime's
    # trace does not make reliable; the share-of-total attribution is unaffected either way.
    profile = parse_profile_events(events, artifact.label, runs + warmup)
    return Profile(
        label=profile.label,
        total_us=profile.total_us,
        runs=profile.runs,
        by_op=profile.by_op,
        by_node=profile.by_node,
        providers_used=providers_used,
        notes={"warmup_included": warmup, "instrumented": True},
    )


@dataclass(frozen=True)
class OpDelta:
    """How one operator type's cost changed between two models."""

    op_type: str
    before_us: float
    after_us: float

    @property
    def delta_us(self) -> float:
        return self.after_us - self.before_us

    @property
    def appeared(self) -> bool:
        return self.before_us == 0.0 and self.after_us > 0.0

    @property
    def vanished(self) -> bool:
        return self.after_us == 0.0 and self.before_us > 0.0


def diff_profiles(before: Profile, after: Profile) -> list[OpDelta]:
    """Per-op-type cost change, largest regression first.

    Normalised per run, so two profiles taken with different iteration counts still
    compare honestly.
    """
    def per_run(profile: Profile) -> dict[str, float]:
        divisor = profile.runs or 1
        return {o.op_type: o.total_us / divisor for o in profile.by_op}

    b, a = per_run(before), per_run(after)
    deltas = [
        OpDelta(op_type=op, before_us=b.get(op, 0.0), after_us=a.get(op, 0.0))
        for op in sorted(set(b) | set(a))
    ]
    return sorted(deltas, key=lambda d: d.delta_us, reverse=True)
