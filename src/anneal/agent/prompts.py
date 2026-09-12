"""Prompt construction for the LLM policy.

The state handed to the model is deliberately *small and numeric*. An optimisation agent
fed a wall of text will pattern-match on the text; fed a table of measured deltas, it has
to reason about the measurements. Each turn is stateless — the full ledger is re-rendered
rather than accumulated as conversation — which keeps the context bounded and makes every
individual decision auditable in isolation.
"""

from __future__ import annotations

from anneal.core.ledger import Ledger, Trial
from anneal.core.transforms import TransformSpec

SYSTEM = """\
You are the search policy inside Anneal, a tool that optimises neural networks for a \
specific hardware target. Each turn you propose ONE transform to apply; the harness \
applies it, benchmarks it on real hardware, and shows you the measured result next turn.

Rules of engagement:

1. Every number you are shown was measured, not estimated. Trust the measurements over \
your priors about what "should" be faster. If INT8 quantization made the model slower on \
this target, that is a real fact about this target's kernels, not an error.
2. You are searching for a Pareto frontier across latency, accuracy and size — not a \
single winner. A proposal that trades 0.4% accuracy for 2x speed AND a proposal that \
keeps accuracy exactly and gains 15% are both valuable. Cover the space.
3. Do not repeat a recipe already in the ledger. The harness will reject it.
4. Prefer informative proposals. Early on, probe different families of transform. Later, \
exploit whatever the measurements showed works, and tune its parameters.
5. Watch top1_agreement, not just accuracy. A model whose accuracy is unchanged but whose \
agreement with the baseline has dropped to 0.7 is behaving very differently and got lucky \
on this eval set; treat it as a risk.
6. Stop when further proposals are unlikely to extend the frontier. Stopping early with a \
clean frontier is a better outcome than burning budget on noise.

Be specific in your rationale: name the measurement that motivated the proposal.\
"""


def render_transform_catalog(transforms: dict[str, TransformSpec]) -> str:
    lines = ["Available transforms:"]
    for spec in transforms.values():
        lines.append(f"\n- {spec.name}: {spec.summary}")
        for pname, meta in spec.params.items():
            enum = f" (one of {meta['enum']})" if "enum" in meta else ""
            lines.append(f"    - {pname}: {meta['type']}{enum} — {meta['description']}")
    return "\n".join(lines)


def _fmt(value: float | None, digits: int = 2, dash: str = "n/a") -> str:
    return dash if value is None else f"{value:.{digits}f}"


def render_trial_row(trial: Trial, ledger: Ledger) -> str:
    if not trial.ok or trial.measurement is None:
        return f"  [{trial.index}] {trial.label}\n        FAILED: {trial.error}"

    m = trial.measurement
    speedup = ledger.speedup_of(trial)
    base = ledger.baseline
    acc_delta = None
    if base and base.measurement and base.measurement.accuracy is not None and m.accuracy is not None:
        acc_delta = (m.accuracy - base.measurement.accuracy) * 100

    parts = [
        f"p50={m.latency_ms_p50:.2f}ms",
        f"p99={m.latency_ms_p99:.2f}ms",
        f"size={m.size_mb:.2f}MB",
    ]
    if m.accuracy is not None:
        parts.append(f"acc={m.accuracy * 100:.2f}%")
    if acc_delta is not None:
        parts.append(f"acc_delta={acc_delta:+.2f}pp")
    if speedup is not None:
        parts.append(f"speedup={speedup:.2f}x")
    if m.top1_agreement is not None:
        parts.append(f"agreement={m.top1_agreement:.3f}")

    return f"  [{trial.index}] {trial.label}\n        " + "  ".join(parts)


def render_state(
    ledger: Ledger,
    transforms: dict[str, TransformSpec],
    budget_remaining: int,
    constraints_text: str,
) -> str:
    target = ledger.target_fingerprint
    header = [
        f"TARGET: {target.get('target', 'unknown')} "
        f"({', '.join(target.get('providers', []))}, "
        f"{target.get('intra_op_threads', '?')} intra-op threads)",
        f"MACHINE: {target.get('processor') or target.get('machine', 'unknown')}",
        f"BUDGET: {budget_remaining} trial(s) remaining",
        f"CONSTRAINTS: {constraints_text}",
        "",
    ]

    body = ["LEDGER (every row below is a real measurement):"]
    if not ledger.trials:
        body.append("  (empty — nothing measured yet)")
    for trial in ledger.trials:
        body.append(render_trial_row(trial, ledger))

    front = ledger.pareto_front()
    if front:
        body.append("")
        body.append("CURRENT PARETO FRONTIER: " + ", ".join(f"[{t.index}] {t.label}" for t in front))

    sensitivity = _sensitivity_hint(ledger)
    if sensitivity:
        body.append("")
        body.append(sensitivity)

    return "\n".join(header + body + ["", render_transform_catalog(transforms)])


def _sensitivity_hint(ledger: Ledger) -> str:
    """Surface the layer-sensitivity analysis if any trial computed one."""
    for trial in reversed(ledger.trials):
        top5 = trial.artifact.meta.get("sensitivity_top5")
        if top5:
            rows = ", ".join(f"{d['node']} ({d['rel_err']:.4f})" for d in top5)
            return (
                "LAYER SENSITIVITY (relative weight-quantization error, most sensitive "
                f"first): {rows}"
            )
    return ""


def build_tools(transforms: dict[str, TransformSpec]) -> list[dict]:
    """Two tools: propose one transform, or stop."""
    return [
        {
            "name": "propose_transform",
            "description": (
                "Apply one transform to a model in the ledger and measure the result. "
                "The harness benchmarks it on real hardware and reports back."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "transform": {
                        "type": "string",
                        "enum": sorted(transforms),
                        "description": "Which transform to apply.",
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "Parameters for the transform. Omit a parameter to accept its "
                            "default. See the catalog for each transform's parameters."
                        ),
                    },
                    "base_trial": {
                        "type": "integer",
                        "description": (
                            "Index of the ledger trial whose model to build on, enabling "
                            "chained recipes (e.g. graph_optimize then quantize). Use -1 "
                            "for the original baseline model."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "Why this proposal, citing the specific measurement that "
                            "motivated it. One or two sentences."
                        ),
                    },
                },
                "required": ["transform", "base_trial", "rationale"],
            },
        },
        {
            "name": "stop",
            "description": (
                "End the search. Use when the frontier is unlikely to improve further."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the search is finished.",
                    }
                },
                "required": ["reason"],
            },
        },
    ]
