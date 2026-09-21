"""Turning a ledger into something a human (or a hiring manager) can read.

Three renderers: a live Rich table for the terminal, a Markdown report for the repo, and
a dependency-free SVG scatter of the Pareto frontier. No matplotlib — a 40-line SVG
writer keeps the install light and the output diffable in git.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from anneal.core.ledger import Ledger, Trial


def _pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


#: Short names for terminal display. The ledger and Markdown report keep the full,
#: unambiguous recipe; a table column is not the place to read forty characters of
#: keyword arguments.
_SHORT_NAMES = {
    "graph_optimize": "fuse",
    "quantize_dynamic_int8": "dyn-int8",
    "quantize_dynamic_sensitive": "sel-int8",
    "quantize_static_int8": "static-int8",
    "cast_fp16": "fp16",
}


def compact_label(trial: Trial) -> str:
    """A recipe rendered short enough to fit in a terminal column."""
    if not trial.artifact.lineage:
        return "baseline (fp32)"

    parts = []
    for record in trial.artifact.lineage:
        name = _SHORT_NAMES.get(record.name, record.name)
        flags = []
        p = record.params
        if "level" in p:
            flags.append(str(p["level"]))
        if p.get("per_channel") is True:
            flags.append("pc")
        elif p.get("per_channel") is False:
            flags.append("pt")
        if p.get("reduce_range"):
            flags.append("rr")
        if p.get("calibrate_method"):
            flags.append(str(p["calibrate_method"]))
        if "skip_top_k" in p:
            flags.append(f"k={p['skip_top_k']}")
        if p.get("skip_first_last"):
            flags.append("+stem/head")
        if p.get("ranking") in ("measured", "proxy"):
            flags.append("meas" if p["ranking"] == "measured" else "proxy")
        if p.get("weight_type") == "uint8":
            flags.append("u8")
        parts.append(f"{name}[{','.join(flags)}]" if flags else name)
    return " > ".join(parts)


def wilson_halfwidth_pp(accuracy: float | None, n: int, z: float = 1.96) -> float | None:
    """Half-width of the 95% Wilson score interval, in percentage points.

    This matters more than it looks. On a 256-image eval set a 2.7pp accuracy difference
    is roughly one standard error — which is to say, not evidence of anything. Reporting
    top-1 to two decimal places without also reporting how much of that is sampling noise
    invites exactly the wrong conclusion. Wilson rather than the normal approximation
    because it stays sane near p=0 and p=1.
    """
    if accuracy is None or n <= 0:
        return None
    z2 = z * z
    denom = 1 + z2 / n
    half = z * ((accuracy * (1 - accuracy) / n + z2 / (4 * n * n)) ** 0.5) / denom
    return half * 100


def _acc_delta_pp(ledger: Ledger, trial: Trial) -> float | None:
    base = ledger.baseline
    if (
        base is None
        or base.measurement is None
        or trial.measurement is None
        or base.measurement.accuracy is None
        or trial.measurement.accuracy is None
    ):
        return None
    return (trial.measurement.accuracy - base.measurement.accuracy) * 100


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def console_table(ledger: Ledger, *, only_frontier: bool = False):
    from rich.table import Table

    front_ids = {t.index for t in ledger.pareto_front()}
    rows = [t for t in ledger.trials if not only_frontier or t.index in front_ids]

    table = Table(show_lines=False, header_style="bold", expand=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("recipe", overflow="ellipsis", no_wrap=True, max_width=34)
    table.add_column("p50 ms", justify="right")
    table.add_column("p99 ms", justify="right")
    table.add_column("speedup", justify="right")
    table.add_column("size MB", justify="right")
    table.add_column("top-1", justify="right")
    # Deliberately ASCII: Windows consoles still default to cp1252, which cannot encode
    # the Greek delta, and a crash in the results table would throw away a whole run.
    table.add_column("acc pp", justify="right")
    table.add_column("agree", justify="right")

    for trial in rows:
        if not trial.ok or trial.measurement is None:
            table.add_row(
                str(trial.index),
                f"[red]{compact_label(trial)}[/red]",
                *(["[red]failed[/red]"] + ["—"] * 7),
            )
            continue

        m = trial.measurement
        speedup = ledger.speedup_of(trial)
        delta = _acc_delta_pp(ledger, trial)
        on_front = trial.index in front_ids

        name = compact_label(trial)
        if on_front:
            name = f"[bold green]{name}[/bold green]"

        speed_txt = "—" if speedup is None else f"{speedup:.2f}x"
        if speedup is not None and speedup > 1.05:
            speed_txt = f"[green]{speed_txt}[/green]"
        elif speedup is not None and speedup < 0.95:
            speed_txt = f"[red]{speed_txt}[/red]"

        delta_txt = "—"
        if delta is not None:
            colour = "green" if delta >= -0.001 else ("yellow" if delta > -1.0 else "red")
            delta_txt = f"[{colour}]{delta:+.2f}[/{colour}]"

        table.add_row(
            str(trial.index),
            name,
            f"{m.latency_ms_p50:.2f}",
            f"{m.latency_ms_p99:.2f}",
            speed_txt,
            f"{m.size_mb:.2f}",
            _pct(m.accuracy),
            delta_txt,
            "—" if m.top1_agreement is None else f"{m.top1_agreement:.3f}",
        )
    return table


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def markdown_report(ledger: Ledger, *, svg_path: str | None = None) -> str:
    fp = ledger.target_fingerprint
    cfg = ledger.config
    base = ledger.baseline

    lines: list[str] = [
        f"# Anneal run `{ledger.run_id}`",
        "",
        "## Setup",
        "",
        "| | |",
        "|---|---|",
        f"| Model | `{cfg.get('baseline_model', '?')}` |",
        f"| Target | `{fp.get('target', '?')}` ({', '.join(fp.get('providers', []))}) |",
        f"| Threads | {fp.get('intra_op_threads', '?')} intra-op, {fp.get('inter_op_threads', '?')} inter-op |",
        f"| Machine | {fp.get('processor') or fp.get('machine', '?')} |",
        f"| Platform | {fp.get('platform', '?')} |",
        f"| onnxruntime | {fp.get('onnxruntime', '?')} |",
        f"| Policy | `{cfg.get('policy', '?')}` |",
        f"| Eval set | `{cfg.get('evalset', 'none')}` ({cfg.get('evalset_size', 0)} images) |",
        f"| Latency protocol | {cfg.get('warmup', '?')} warm-up discarded, {cfg.get('runs', '?')} timed runs, batch {cfg.get('batch_size', '?')} |",
        f"| Budget | {cfg.get('budget', '?')} trials |",
        "",
    ]

    env = cfg.get("environment") or {}
    if env:
        drift = env.get("baseline_drift")
        trustworthy = env.get("latency_trustworthy")
        if trustworthy is False:
            lines += [
                "> **Latencies in this run are not trustworthy.** The machine's performance "
                "state was degraded or changed while it ran:",
                ">",
                *[f"> - {w}" for w in env.get("warnings", [])],
                *(
                    [f"> - the baseline measured {env['baseline_p50_start_ms']:.2f}ms at the "
                     f"start and {env['baseline_p50_end_ms']:.2f}ms at the end "
                     f"({drift * 100:.1f}% drift)"]
                    if drift is not None
                    else []
                ),
                ">",
                "> Accuracy numbers are unaffected. Re-run on a stable machine before comparing speeds.",
                "",
            ]
        elif drift is not None:
            lines += [
                f"**Measurement stability:** the baseline re-measured at the end of the run "
                f"drifted {drift * 100:.1f}% (tolerance 10%), and no power or throttling "
                f"problems were detected.",
                "",
            ]

    if base is not None and base.measurement is not None:
        half = wilson_halfwidth_pp(base.measurement.accuracy, base.measurement.n_eval)
        if half is not None:
            lines += [
                f"**Accuracy resolution: ±{half:.1f}pp** (95% Wilson interval at "
                f"n={base.measurement.n_eval}). Two trials whose top-1 differs by less "
                f"than roughly this much are tied, not ranked. When the difference is "
                f"within noise, `agreement` — the fraction of images where a candidate "
                f"predicts the same class as the baseline — is the sharper signal, "
                f"because it is paired per-image rather than an aggregate.",
                "",
            ]

    if cfg.get("evalset_synthetic"):
        lines += [
            "> **Note:** this run used the synthetic eval set. Accuracy numbers below are "
            "meaningless by construction and are shown only to demonstrate the plumbing.",
            "",
        ]

    if svg_path:
        lines += ["## Frontier", "", f"![Pareto frontier]({svg_path})", ""]

    lines += ["## All trials", "", _md_table(ledger, ledger.trials), ""]

    front = ledger.pareto_front()
    if front:
        lines += [
            "## Pareto frontier",
            "",
            "Non-dominated across (latency p50, size, top-1). No single row is 'best' — "
            "pick by whichever constraint binds you.",
            "",
            _md_table(ledger, front),
            "",
        ]

    pick = ledger.best_under_constraints(
        min_accuracy=_floor_from_config(ledger),
    )
    if pick is not None and pick.measurement is not None:
        speedup = ledger.speedup_of(pick)
        delta = _acc_delta_pp(ledger, pick)
        size_ratio = None
        if base and base.measurement and base.measurement.size_bytes:
            size_ratio = pick.measurement.size_bytes / base.measurement.size_bytes
        lines += [
            "## Recommended pick",
            "",
            f"**`{pick.label}`** — trial {pick.index}.",
            "",
            f"- {speedup:.2f}x faster (p50)" if speedup else "- speedup: n/a",
            f"- {delta:+.2f}pp top-1" if delta is not None else "- accuracy: n/a",
            f"- {size_ratio * 100:.0f}% of baseline size" if size_ratio else "- size: n/a",
            "",
            f"> {pick.rationale}" if pick.rationale else "",
            "",
        ]

    warnings = [
        (t, t.artifact.meta["portability_warning"])
        for t in ledger.trials
        if t.artifact.meta.get("portability_warning")
    ]
    if warnings:
        lines += [
            "## Portability caveats",
            "",
            "A fast number on this machine is not automatically a fast number on the "
            "deployment target. These artifacts carry hardware-specific assumptions:",
            "",
        ]
        for trial, warning in warnings:
            lines.append(f"- **Trial {trial.index}** (`{trial.label}`): {warning}")
        lines.append("")

    failures = [t for t in ledger.trials if not t.ok]
    if failures:
        lines += [
            "## Failed trials",
            "",
            "Recorded rather than hidden — a transform that does not apply is a real "
            "property of this model and target.",
            "",
        ]
        for t in failures:
            lines.append(f"- `{t.label}` — {t.error}")
        lines.append("")

    lines += [
        "## Reproducing",
        "",
        "Every row above is a recipe. To rebuild one, apply its transform chain to the "
        "baseline model with the same parameters; the ledger JSON alongside this report "
        "records the exact parameters, the machine fingerprint and the measurement "
        "protocol used.",
        "",
    ]

    return "\n".join(line for line in lines if line is not None)


def _floor_from_config(ledger: Ledger) -> float | None:
    constraints = ledger.config.get("constraints") or {}
    base = ledger.baseline
    floors = []
    if constraints.get("min_accuracy") is not None:
        floors.append(float(constraints["min_accuracy"]))
    drop = constraints.get("max_accuracy_drop_pp")
    if drop is not None and base and base.measurement and base.measurement.accuracy is not None:
        floors.append(base.measurement.accuracy - float(drop) / 100.0)
    return max(floors) if floors else None


def _md_table(ledger: Ledger, trials: Sequence[Trial]) -> str:
    head = (
        "| # | recipe | p50 ms | p99 ms | speedup | size MB | top-1 | Δpp | agreement |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for t in trials:
        if not t.ok or t.measurement is None:
            rows.append(f"| {t.index} | `{t.label}` | failed | | | | | | |")
            continue
        m = t.measurement
        speedup = ledger.speedup_of(t)
        delta = _acc_delta_pp(ledger, t)
        rows.append(
            f"| {t.index} | `{t.label}` | {m.latency_ms_p50:.2f} | {m.latency_ms_p99:.2f} | "
            f"{'—' if speedup is None else f'{speedup:.2f}x'} | {m.size_mb:.2f} | "
            f"{_pct(m.accuracy)} | {'—' if delta is None else f'{delta:+.2f}'} | "
            f"{'—' if m.top1_agreement is None else f'{m.top1_agreement:.3f}'} |"
        )
    return "\n".join([head, *rows])


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Point:
    x: float
    y: float
    label: str
    on_front: bool
    is_baseline: bool


def frontier_svg(ledger: Ledger, *, width: int = 760, height: int = 440) -> str:
    """Scatter of latency vs accuracy, with the Pareto frontier traced."""
    scored = ledger.scored()
    if not scored:
        return _empty_svg(width, height, "no scored trials")

    front_ids = {t.index for t in ledger.pareto_front()}
    base = ledger.baseline

    points = [
        _Point(
            x=t.measurement.latency_ms_p50,  # type: ignore[union-attr]
            y=t.measurement.accuracy * 100,  # type: ignore[union-attr,operator]
            label=str(t.index),
            on_front=t.index in front_ids,
            is_baseline=(base is not None and t.index == base.index),
        )
        for t in scored
    ]

    pad_l, pad_r, pad_t, pad_b = 68, 24, 34, 52
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = (x_max - x_min) * 0.12 or max(x_max * 0.1, 0.1)
    y_pad = (y_max - y_min) * 0.18 or 0.5
    x_min, x_max = max(0.0, x_min - x_pad), x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def sx(v: float) -> float:
        return pad_l + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v: float) -> float:
        return pad_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="ui-sans-serif,system-ui,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    # Gridlines + axis ticks
    for i in range(5):
        gy = pad_t + plot_h * i / 4
        value = y_max - (y_max - y_min) * i / 4
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" '
            f'stroke="#e8eaed" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 10}" y="{gy + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="#5f6368">{value:.1f}%</text>'
        )
    for i in range(5):
        gx = pad_l + plot_w * i / 4
        value = x_min + (x_max - x_min) * i / 4
        parts.append(
            f'<text x="{gx:.1f}" y="{pad_t + plot_h + 20}" text-anchor="middle" '
            f'font-size="11" fill="#5f6368">{value:.1f}</text>'
        )

    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
        f'y2="{pad_t + plot_h}" stroke="#9aa0a6" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" '
        f'stroke="#9aa0a6" stroke-width="1"/>'
    )

    # Frontier polyline
    front_pts = sorted([p for p in points if p.on_front], key=lambda p: p.x)
    if len(front_pts) > 1:
        d = " ".join(f"{sx(p.x):.1f},{sy(p.y):.1f}" for p in front_pts)
        parts.append(
            f'<polyline points="{d}" fill="none" stroke="#1a73e8" stroke-width="2" '
            f'stroke-dasharray="5,4" opacity="0.8"/>'
        )

    for p in points:
        cx, cy = sx(p.x), sy(p.y)
        if p.is_baseline:
            fill, stroke, r = "#ffffff", "#202124", 7
        elif p.on_front:
            fill, stroke, r = "#1a73e8", "#0b57d0", 6
        else:
            fill, stroke, r = "#dadce0", "#9aa0a6", 5
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{cy - r - 5:.1f}" text-anchor="middle" font-size="10" '
            f'fill="#3c4043">{p.label}</text>'
        )

    parts.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 12}" text-anchor="middle" '
        f'font-size="12" fill="#202124">p50 latency (ms) — lower is better</text>'
    )
    parts.append(
        f'<text x="16" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" font-size="12" '
        f'fill="#202124" transform="rotate(-90 16 {pad_t + plot_h / 2:.1f})">'
        f'top-1 accuracy — higher is better</text>'
    )

    target = ledger.target_fingerprint.get("target", "?")
    parts.append(
        f'<text x="{pad_l}" y="20" font-size="12" fill="#5f6368">'
        f'target: {target} · hollow = baseline · blue = Pareto frontier</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _empty_svg(width: int, height: int, message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}"><rect width="{width}" height="{height}" '
        f'fill="#ffffff"/><text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
        f'font-size="14" fill="#5f6368">{message}</text></svg>'
    )


def write_report(ledger: Ledger, outdir: Path) -> dict[str, Path]:
    """Write ledger.json, report.md and frontier.svg into ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ledger_path = ledger.save(outdir / "ledger.json")
    svg_path = outdir / "frontier.svg"
    svg_path.write_text(frontier_svg(ledger), encoding="utf-8")
    md_path = outdir / "report.md"
    md_path.write_text(markdown_report(ledger, svg_path="frontier.svg"), encoding="utf-8")

    return {"ledger": ledger_path, "report": md_path, "svg": svg_path}
