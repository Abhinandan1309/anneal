"""Audit an optimised model against its original — whoever produced it.

Anneal's search is one way to get an optimised model. Olive, Neural Compressor, NNCF,
TensorRT's builder and a colleague's script are others. They all end in the same place:
two ONNX files and a claim that the second is faster and "about as accurate". This module
checks that claim on the target it will run on, and it does not care which tool made it.

The accuracy comparison is **paired**. Both models see the same images, so what matters is
the images whose outcome differs between them:

* ``b`` — images the original got right and the candidate gets wrong (regressions)
* ``c`` — images the original got wrong and the candidate gets right (fixes)

McNemar's test on (b, c) asks whether that imbalance could be chance, and the paired
confidence interval on the accuracy delta says how large a loss the data cannot rule out.
Comparing two separately-computed accuracies throws away the pairing, and with it most of
the statistical power — which is how a real regression gets waved through as "noise".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from anneal.core.artifact import ModelArtifact
from anneal.core.measure import Benchmarker, EvalSet, Measurement, MeasurementError
from anneal.core.targets import Target


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts ``b`` and ``c``.

    Under the null hypothesis each discordant image is equally likely to go either way, so
    min(b, c) is Binomial(b + c, 1/2). The exact test is used rather than the chi-square
    approximation because discordant counts in a quantization audit are often small.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_delta_ci(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Accuracy delta (candidate − original) and its 95% CI, in percentage points.

    Uses the standard error of a difference of paired proportions:
    sqrt((b + c) − (c − b)² / n) / n.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    delta = (c - b) / n
    var = max(0.0, (b + c) - (c - b) ** 2 / n)
    se = math.sqrt(var) / n
    return delta * 100, (delta - z * se) * 100, (delta + z * se) * 100


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------


def predict(
    artifact: ModelArtifact, target: Target, evalset: EvalSet
) -> tuple[np.ndarray, np.ndarray]:
    """Per-image predictions and ground-truth labels, in eval-set order."""
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(
            str(artifact.path),
            sess_options=target.session_options(),
            providers=list(target.providers),
        )
    except Exception as exc:
        raise MeasurementError(f"could not load {artifact.path.name}: {exc}") from exc

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in evalset.batches():
        try:
            logits = session.run([output_name], {input_name: x})[0]
        except Exception as exc:
            raise MeasurementError(
                f"{artifact.path.name} failed on the eval set (input {x.shape}): {exc}"
            ) from exc
        preds.append(np.asarray(evalset.decode(np.asarray(logits, dtype=np.float32))))
        labels.append(np.asarray(y))
    del session
    if not preds:
        raise MeasurementError("the eval set yielded no batches")
    return np.concatenate(preds), np.concatenate(labels)


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------


@dataclass
class ClassDelta:
    label: int
    name: str
    n: int
    original_acc: float
    candidate_acc: float

    @property
    def delta_pp(self) -> float:
        return (self.candidate_acc - self.original_acc) * 100


@dataclass
class AuditResult:
    original: str
    candidate: str
    target: str
    n: int
    original_latency: Measurement
    candidate_latency: Measurement
    original_acc: float
    candidate_acc: float
    #: original right, candidate wrong
    regressions: int
    #: original wrong, candidate right
    fixes: int
    #: images whose predicted class changed at all, right or wrong
    changed: int
    p_value: float
    delta_pp: float
    ci_low_pp: float
    ci_high_pp: float
    classes: list[ClassDelta] = field(default_factory=list)
    profile_diff: list[dict[str, Any]] = field(default_factory=list)

    @property
    def speedup_p50(self) -> float:
        return self.original_latency.latency_ms_p50 / self.candidate_latency.latency_ms_p50

    @property
    def speedup_p99(self) -> float:
        return self.original_latency.latency_ms_p99 / self.candidate_latency.latency_ms_p99

    @property
    def size_ratio(self) -> float:
        return self.candidate_latency.size_bytes / self.original_latency.size_bytes

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def verdict(self, max_drop_pp: float = 1.0) -> list[str]:
        """Plain-language findings, most important first."""
        lines: list[str] = []

        if self.speedup_p50 < 1.0:
            lines.append(
                f"SLOWER, not faster: {1 / self.speedup_p50:.2f}x the original's median latency "
                f"on {self.target}."
            )
        elif self.speedup_p50 < 1.05:
            lines.append(f"No meaningful speedup on {self.target} ({self.speedup_p50:.2f}x).")
        else:
            lines.append(f"{self.speedup_p50:.2f}x faster at the median on {self.target}.")
        if self.speedup_p50 >= 1.05 and self.speedup_p99 < 0.95 * self.speedup_p50:
            lines.append(
                f"Tail latency improves less: {self.speedup_p99:.2f}x at p99 vs "
                f"{self.speedup_p50:.2f}x at p50."
            )

        if self.significant and self.delta_pp < 0:
            lines.append(
                f"Accuracy loss is real: {self.delta_pp:+.2f}pp "
                f"(95% CI {self.ci_low_pp:+.2f} to {self.ci_high_pp:+.2f}), McNemar p = "
                f"{self.p_value:.2g}. {self.regressions} images broke, {self.fixes} were fixed."
            )
        elif self.significant:
            lines.append(
                f"Accuracy improved significantly ({self.delta_pp:+.2f}pp, p = "
                f"{self.p_value:.2g}) — unusual for an optimisation; check the eval set."
            )
        else:
            lines.append(
                f"No detectable accuracy change at n={self.n} (p = {self.p_value:.2g}), but the "
                f"data cannot rule out a loss of up to {-self.ci_low_pp:.2f}pp."
            )
            if -self.ci_low_pp > max_drop_pp:
                lines.append(
                    f"That is wider than a {max_drop_pp:.1f}pp budget: audit on more images "
                    f"before signing this off."
                )

        changed_pct = self.changed / self.n * 100
        if changed_pct >= 1.0:
            lines.append(
                f"{changed_pct:.1f}% of predictions changed class — far more than the accuracy "
                f"delta alone suggests, because regressions and fixes partly cancel."
                if abs(self.delta_pp) < changed_pct / 2
                else f"{changed_pct:.1f}% of predictions changed class."
            )

        worst = [c for c in self.classes if c.delta_pp < 0]
        if worst and self.significant and self.delta_pp < 0:
            w = worst[0]
            lines.append(
                f"The loss is concentrated: '{w.name}' dropped {w.delta_pp:+.1f}pp "
                f"({w.original_acc * 100:.1f}% -> {w.candidate_acc * 100:.1f}%, n={w.n})."
            )
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "candidate": self.candidate,
            "target": self.target,
            "n": self.n,
            "latency": {
                "original_p50_ms": self.original_latency.latency_ms_p50,
                "original_p99_ms": self.original_latency.latency_ms_p99,
                "candidate_p50_ms": self.candidate_latency.latency_ms_p50,
                "candidate_p99_ms": self.candidate_latency.latency_ms_p99,
                "speedup_p50": self.speedup_p50,
                "speedup_p99": self.speedup_p99,
            },
            "size": {
                "original_bytes": self.original_latency.size_bytes,
                "candidate_bytes": self.candidate_latency.size_bytes,
                "ratio": self.size_ratio,
            },
            "accuracy": {
                "original": self.original_acc,
                "candidate": self.candidate_acc,
                "delta_pp": self.delta_pp,
                "ci95_pp": [self.ci_low_pp, self.ci_high_pp],
                "mcnemar_p": self.p_value,
                "regressions": self.regressions,
                "fixes": self.fixes,
                "changed": self.changed,
            },
            "classes": [
                {
                    "label": c.label,
                    "name": c.name,
                    "n": c.n,
                    "original_acc": c.original_acc,
                    "candidate_acc": c.candidate_acc,
                    "delta_pp": c.delta_pp,
                }
                for c in self.classes
            ],
            "profile_diff": self.profile_diff,
            "verdict": self.verdict(),
        }

    def markdown(self) -> str:
        o, c = self.original_latency, self.candidate_latency
        rows = [
            f"# Audit: `{self.candidate}` vs `{self.original}`",
            "",
            f"Target `{self.target}`, {self.n} evaluation images.",
            "",
            "## Verdict",
            "",
            *[f"- {line}" for line in self.verdict()],
            "",
            "## Measurements",
            "",
            "| | original | candidate | ratio |",
            "|---|---:|---:|---:|",
            f"| p50 latency | {o.latency_ms_p50:.2f} ms | {c.latency_ms_p50:.2f} ms | "
            f"{self.speedup_p50:.2f}x faster |",
            f"| p99 latency | {o.latency_ms_p99:.2f} ms | {c.latency_ms_p99:.2f} ms | "
            f"{self.speedup_p99:.2f}x faster |",
            f"| size | {o.size_mb:.2f} MB | {c.size_mb:.2f} MB | {self.size_ratio * 100:.0f}% |",
            f"| top-1 | {self.original_acc * 100:.2f}% | {self.candidate_acc * 100:.2f}% | "
            f"{self.delta_pp:+.2f}pp |",
            "",
            "## Paired accuracy test",
            "",
            f"- Images the original got right and the candidate got wrong: **{self.regressions}**",
            f"- Images the original got wrong and the candidate got right: **{self.fixes}**",
            f"- Images whose predicted class changed at all: **{self.changed}** "
            f"({self.changed / self.n * 100:.1f}%)",
            f"- Accuracy delta: **{self.delta_pp:+.2f}pp**, 95% CI "
            f"[{self.ci_low_pp:+.2f}, {self.ci_high_pp:+.2f}]",
            f"- Exact McNemar p-value: **{self.p_value:.3g}**",
            "",
        ]
        if self.classes:
            rows += [
                "## Per class",
                "",
                "| class | n | original | candidate | Δpp |",
                "|---|---:|---:|---:|---:|",
                *[
                    f"| {k.name} | {k.n} | {k.original_acc * 100:.1f}% | "
                    f"{k.candidate_acc * 100:.1f}% | {k.delta_pp:+.1f} |"
                    for k in self.classes
                ],
                "",
            ]
        if self.profile_diff:
            rows += [
                "## Where the time moved (per run, instrumented)",
                "",
                "| operator | original ms | candidate ms | Δ ms |",
                "|---|---:|---:|---:|",
                *[
                    f"| {d['op_type']} | {d['before_ms']:.2f} | {d['after_ms']:.2f} | "
                    f"{d['delta_ms']:+.2f} |"
                    for d in self.profile_diff
                ],
                "",
            ]
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def audit(
    original: ModelArtifact,
    candidate: ModelArtifact,
    target: Target,
    evalset: EvalSet,
    *,
    warmup: int = 10,
    runs: int = 50,
    class_names: dict[int, str] | None = None,
    profile: bool = False,
) -> AuditResult:
    """Measure both models on ``target`` and compare them image by image."""
    bench = Benchmarker(target, warmup=warmup, runs=runs)
    lat_o = bench.measure(original)
    lat_c = bench.measure(candidate)

    pred_o, labels = predict(original, target, evalset)
    pred_c, labels_c = predict(candidate, target, evalset)
    if not np.array_equal(labels, labels_c):  # pragma: no cover - eval sets are deterministic
        raise MeasurementError("eval set was not deterministic between the two passes")

    right_o = pred_o == labels
    right_c = pred_c == labels
    b = int(np.sum(right_o & ~right_c))
    c = int(np.sum(~right_o & right_c))
    n = int(labels.shape[0])
    delta, lo, hi = paired_delta_ci(b, c, n)

    names = class_names or {}
    classes = []
    for label in np.unique(labels):
        mask = labels == label
        classes.append(
            ClassDelta(
                label=int(label),
                name=names.get(int(label), str(int(label))),
                n=int(mask.sum()),
                original_acc=float(right_o[mask].mean()),
                candidate_acc=float(right_c[mask].mean()),
            )
        )
    classes.sort(key=lambda k: k.delta_pp)

    profile_rows: list[dict[str, Any]] = []
    if profile:
        from anneal.core.profile import diff_profiles, profile_model

        diff = diff_profiles(profile_model(original, target), profile_model(candidate, target))
        profile_rows = [
            {
                "op_type": d.op_type,
                "before_ms": d.before_us / 1000,
                "after_ms": d.after_us / 1000,
                "delta_ms": d.delta_us / 1000,
            }
            for d in diff
            if abs(d.delta_us) >= 50
        ]

    return AuditResult(
        original=original.path.name,
        candidate=candidate.path.name,
        target=target.name,
        n=n,
        original_latency=lat_o,
        candidate_latency=lat_c,
        original_acc=float(right_o.mean()),
        candidate_acc=float(right_c.mean()),
        regressions=b,
        fixes=c,
        changed=int(np.sum(pred_o != pred_c)),
        p_value=mcnemar_exact(b, c),
        delta_pp=delta,
        ci_low_pp=lo,
        ci_high_pp=hi,
        classes=classes,
        profile_diff=profile_rows,
    )
