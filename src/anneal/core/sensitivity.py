"""Measured layer sensitivity — and an honest test of the cheap proxy.

:func:`anneal.core.transforms.rank_layer_sensitivity` ranks layers by the relative L2
error INT8 introduces into their weights. It costs nothing and it is a *guess*: it ignores
activation ranges and how error propagates through the rest of the network.

This module does the expensive version. It quantizes exactly one layer at a time, runs the
eval set, and records how many predictions actually changed. Then it asks the question the
proxy's existence depends on: **does the cheap ranking predict the measured one?** If the
rank correlation is poor, the proxy is decoration and the selective-quantization transform
is choosing layers for no reason. Better to measure that and know.

Prediction *changes* are the signal here rather than accuracy drop. They are paired
per-image, so a 3% change is resolvable on a few hundred images, whereas two independent
accuracy aggregates on the same data are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from anneal.core.artifact import ModelArtifact, TransformRecord
from anneal.core.measure import Benchmarker, EvalSet, MeasurementError
from anneal.core.targets import Target
from anneal.core.transforms import (
    TransformContext,
    rank_layer_sensitivity,
)


@dataclass(frozen=True)
class LayerSensitivity:
    """One layer's predicted and measured quantization damage."""

    node: str
    op_type: str
    proxy_error: float
    #: Fraction of eval predictions that changed when only this layer was quantized.
    changed_fraction: float | None = None
    accuracy_drop_pp: float | None = None
    error: str | None = None

    @property
    def measured(self) -> bool:
        return self.changed_fraction is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "op_type": self.op_type,
            "proxy_error": self.proxy_error,
            "changed_fraction": self.changed_fraction,
            "accuracy_drop_pp": self.accuracy_drop_pp,
            "error": self.error,
        }


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks with ties averaged, which is what Spearman requires."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation. Returns None when it is undefined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    rx, ry = _average_ranks(xs), _average_ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n

    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = sum((a - mx) ** 2 for a in rx) ** 0.5
    den_y = sum((b - my) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def quantize_single_node(
    artifact: ModelArtifact, node: str, ctx: TransformContext, *, per_channel: bool = True
) -> ModelArtifact:
    """Quantize exactly one node's weights, leaving the rest of the graph in FP32.

    Weight-only dynamic quantization is deliberate: it isolates the same thing the proxy
    models, so the comparison between them is apples to apples. Static quantization would
    also perturb activations and muddy the attribution.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    record = TransformRecord("quantize_one", {"node": node})
    out = ctx.path_for(artifact, record)
    if not out.exists():
        quantize_dynamic(
            model_input=str(artifact.path),
            model_output=str(out),
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            nodes_to_quantize=[node],
        )
    return artifact.derive(record, out)


def measured_sensitivity(
    artifact: ModelArtifact,
    target: Target,
    evalset: EvalSet,
    ctx: TransformContext,
    *,
    per_channel: bool = True,
    limit: int | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[LayerSensitivity]:
    """Quantize each layer alone and measure how much the model's behaviour changes.

    Costs one eval pass per quantizable layer, which is the entire point — it is the
    expensive ground truth the cheap proxy is trying to approximate.
    """
    proxy = rank_layer_sensitivity(artifact.path, per_channel=per_channel)
    if limit is not None:
        proxy = proxy[:limit]

    # Latency is irrelevant here; keep the timed portion minimal.
    bench = Benchmarker(target, warmup=1, runs=2)

    baseline = bench.measure(artifact, evalset=evalset, record_baseline=True)
    baseline_acc = baseline.accuracy

    results: list[LayerSensitivity] = []
    for i, (node, proxy_error, op_type) in enumerate(proxy, start=1):
        if on_progress:
            on_progress(i, len(proxy), node)
        try:
            candidate = quantize_single_node(artifact, node, ctx, per_channel=per_channel)
            m = bench.measure(candidate, evalset=evalset)
        except (MeasurementError, Exception) as exc:  # vendor code, wide exception surface
            results.append(
                LayerSensitivity(
                    node=node,
                    op_type=op_type,
                    proxy_error=proxy_error,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        changed = None if m.top1_agreement is None else 1.0 - m.top1_agreement
        drop = (
            None
            if (baseline_acc is None or m.accuracy is None)
            else (baseline_acc - m.accuracy) * 100
        )
        results.append(
            LayerSensitivity(
                node=node,
                op_type=op_type,
                proxy_error=proxy_error,
                changed_fraction=changed,
                accuracy_drop_pp=drop,
            )
        )

    return results


def proxy_agreement(results: Sequence[LayerSensitivity]) -> dict[str, Any]:
    """Does the cheap proxy rank layers the way measurement does?"""
    usable = [r for r in results if r.measured]
    if len(usable) < 3:
        return {
            "n": len(usable),
            "spearman": None,
            "verdict": "too few measured layers to say anything",
        }

    rho = spearman(
        [r.proxy_error for r in usable],
        [r.changed_fraction or 0.0 for r in usable],
    )

    if rho is None:
        verdict = "undefined (no variance in one of the rankings)"
    elif rho >= 0.7:
        verdict = "strong — the proxy is a good stand-in for the measured sweep"
    elif rho >= 0.4:
        verdict = "moderate — the proxy is useful as a prior, not as an answer"
    elif rho >= 0.1:
        verdict = "weak — the proxy barely beats picking layers at random"
    else:
        verdict = (
            "none — the proxy does not predict measured damage on this model; "
            "selective quantization should use the measured sweep instead"
        )

    top_proxy = {r.node for r in sorted(usable, key=lambda r: r.proxy_error, reverse=True)[:5]}
    top_measured = {
        r.node for r in sorted(usable, key=lambda r: r.changed_fraction or 0.0, reverse=True)[:5]
    }

    return {
        "n": len(usable),
        "spearman": rho,
        "verdict": verdict,
        "top5_overlap": len(top_proxy & top_measured),
        "top5_proxy": sorted(top_proxy),
        "top5_measured": sorted(top_measured),
    }
