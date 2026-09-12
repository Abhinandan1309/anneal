"""Command line interface."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from anneal import __version__

DEFAULT_CACHE = Path.home() / ".anneal_cache"


def _console():
    from rich.console import Console

    return Console()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from anneal.agent.loop import OptimizationRun, RunConfig
    from anneal.agent.policy import Constraints, build_policy
    from anneal.core.dataset import load_evalset
    from anneal.core.targets import TargetUnavailable, default_target, get_target
    from anneal.core.artifact import model_batch_dim
    from anneal.core.measure import MeasurementError
    from anneal.models import resolve_model
    from anneal.report import console_table, write_report

    console = _console()

    outdir = Path(args.out) if args.out else Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)

    try:
        target = get_target(args.target) if args.target else default_target()
        target.ensure_available()
    except (KeyError, TargetUnavailable) as exc:
        console.print(f"[red]target error:[/red] {exc}")
        return 2

    console.rule(f"[bold]anneal[/bold] · {target.name}")
    console.print(f"[dim]{target.description}[/dim]\n")

    console.print(f"Loading model [cyan]{args.model}[/cyan] …")
    try:
        baseline = resolve_model(
            args.model, outdir / "models", batch_size=args.batch_size, image_size=args.image_size
        )
    except Exception as exc:
        console.print(f"[red]could not load model:[/red] {exc}")
        return 2

    # A graph with a fixed batch axis dictates batching; honour it rather than letting
    # onnxruntime fail on a shape mismatch halfway through the run.
    fixed_batch = model_batch_dim(baseline.path)
    eval_batch = args.eval_batch
    if fixed_batch is not None:
        if eval_batch != fixed_batch:
            console.print(
                f"[yellow]model has a fixed batch axis of {fixed_batch}; evaluating at "
                f"batch {fixed_batch} rather than --eval-batch {eval_batch}[/yellow]"
            )
            eval_batch = fixed_batch
        if args.batch_size != fixed_batch:
            console.print(
                f"[yellow]benchmarking at batch {fixed_batch} rather than --batch-size "
                f"{args.batch_size}, which the graph cannot accept[/yellow]"
            )

    evalset = None
    if args.eval != "none":
        console.print(f"Loading eval set [cyan]{args.eval}[/cyan] …")
        try:
            evalset = load_evalset(
                args.eval, cache_dir=cache, batch_size=eval_batch, limit=args.eval_limit
            )
        except Exception as exc:
            console.print(f"[red]could not load eval set:[/red] {exc}")
            return 2
        console.print(f"  {len(evalset)} images")
        if getattr(evalset, "synthetic", False):
            console.print(
                "  [yellow]synthetic data — accuracy numbers from this run are not "
                "meaningful[/yellow]"
            )

    try:
        policy = build_policy(args.policy, model=args.llm_model)
    except (ImportError, RuntimeError, ValueError) as exc:
        console.print(f"[red]policy error:[/red] {exc}")
        return 2

    config = RunConfig(
        workdir=outdir,
        budget=args.budget,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        calib_samples=args.calib_samples,
        seed=args.seed,
        constraints=Constraints(
            max_accuracy_drop_pp=args.max_accuracy_drop,
            min_accuracy=args.min_accuracy,
            max_size_bytes=int(args.max_size_mb * 1e6) if args.max_size_mb else None,
            max_latency_ms=args.max_latency_ms,
        ),
    )

    console.print(
        f"\nPolicy [cyan]{policy.name}[/cyan] · budget [cyan]{args.budget}[/cyan] trials · "
        f"{args.warmup} warm-up + {args.runs} timed runs each\n"
    )

    def on_event(kind: str, payload: dict[str, Any]) -> None:
        if kind == "baseline_start":
            console.print("[dim]measuring baseline …[/dim]")
        elif kind == "proposal":
            console.print(
                f"[bold]#{payload['index']}[/bold] [cyan]{payload['transform']}[/cyan]"
                f"{payload['params'] or ''}"
                + (f" [dim]on #{payload['base_index']}[/dim]" if payload["base_index"] >= 0 else "")
            )
            if payload["rationale"]:
                console.print(f"    [dim italic]{payload['rationale']}[/dim italic]")
        elif kind == "trial":
            trial = payload["trial"]
            ledger = payload["ledger"]
            if not trial.ok:
                console.print(f"    [red]failed[/red] {trial.error}")
                return
            m = trial.measurement
            speedup = ledger.speedup_of(trial)
            bits = [f"p50 {m.latency_ms_p50:.2f}ms", f"{m.size_mb:.1f}MB"]
            if speedup is not None:
                bits.append(f"[{'green' if speedup > 1 else 'red'}]{speedup:.2f}x[/]")
            if m.accuracy is not None:
                bits.append(f"top-1 {m.accuracy * 100:.2f}%")
            console.print("    " + "  ".join(bits))
        elif kind == "duplicate":
            console.print("    [yellow]already tried that recipe — skipping[/yellow]")
        elif kind == "stopped":
            console.print(f"\n[dim]stopped: {payload['reason']}[/dim]")

    run = OptimizationRun(
        baseline, target, policy, config, evalset=evalset, on_event=on_event
    )

    try:
        ledger = run.run()
    except (RuntimeError, MeasurementError) as exc:
        console.print(f"[red]run failed:[/red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted — writing partial results[/yellow]")
        ledger = run.ledger

    console.print()
    console.rule("[bold]results[/bold]")
    console.print(console_table(ledger))

    front = ledger.pareto_front()
    if front:
        console.print(f"\n[bold]Pareto frontier:[/bold] {len(front)} of {len(ledger.trials)} trials")

    pick = ledger.best_under_constraints(
        min_accuracy=None if args.max_accuracy_drop is None else _floor(ledger, args)
    )
    if pick is not None:
        from anneal.report import compact_label

        speedup = ledger.speedup_of(pick)
        console.print(
            f"[bold green]pick:[/bold green] {compact_label(pick)}"
            + (f"  ({speedup:.2f}x faster)" if speedup else "")
        )

    paths = write_report(ledger, outdir)
    console.print(f"\n[dim]report:   {paths['report']}[/dim]")
    console.print(f"[dim]ledger:   {paths['ledger']}[/dim]")
    console.print(f"[dim]frontier: {paths['svg']}[/dim]")
    return 0


def _floor(ledger, args) -> float | None:
    base = ledger.baseline
    if base is None or base.measurement is None or base.measurement.accuracy is None:
        return args.min_accuracy
    floor = base.measurement.accuracy - args.max_accuracy_drop / 100.0
    if args.min_accuracy is not None:
        floor = max(floor, args.min_accuracy)
    return floor


# ---------------------------------------------------------------------------
# introspection
# ---------------------------------------------------------------------------


def cmd_targets(args: argparse.Namespace) -> int:
    from rich.table import Table

    from anneal.core.targets import list_targets

    console = _console()
    table = Table(title="targets", header_style="bold")
    table.add_column("name")
    table.add_column("status")
    table.add_column("description", overflow="fold")

    for target in list_targets():
        if target.is_available():
            status = "[green]available[/green]"
        elif target.adapter_hint:
            status = "[yellow]adapter needed[/yellow]"
        else:
            status = "[dim]runtime missing[/dim]"
        table.add_row(target.name, status, target.description)

    console.print(table)
    console.print(
        "\n[dim]Targets marked 'adapter needed' are declared so the extension point is "
        "visible; they require a vendor toolchain Anneal does not bundle.[/dim]"
    )
    return 0


def cmd_transforms(args: argparse.Namespace) -> int:
    from anneal.core.transforms import REGISTRY

    console = _console()
    for spec in REGISTRY.values():
        ok, why = spec.available()
        mark = "[green]•[/green]" if ok else "[yellow]•[/yellow]"
        console.print(f"{mark} [bold cyan]{spec.name}[/bold cyan]")
        console.print(f"  {spec.summary}")
        if not ok:
            console.print(f"  [yellow]unavailable: {why}[/yellow]")
        for pname, meta in spec.params.items():
            enum = f" {meta['enum']}" if "enum" in meta else ""
            console.print(f"    [dim]{pname}: {meta['type']}{enum} — {meta['description']}[/dim]")
        console.print()
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    from rich.table import Table

    from anneal.core.transforms import rank_layer_sensitivity

    console = _console()
    path = Path(args.model)
    if not path.is_file():
        console.print(f"[red]no such file:[/red] {path}")
        return 2

    ranked = rank_layer_sensitivity(path, per_channel=not args.per_tensor)
    if not ranked:
        console.print("[yellow]no quantizable Conv/Gemm/MatMul nodes found[/yellow]")
        return 1

    table = Table(
        title=f"INT8 weight-quantization sensitivity — {path.name}", header_style="bold"
    )
    table.add_column("rank", justify="right", style="dim")
    table.add_column("node", overflow="fold")
    table.add_column("op")
    table.add_column("rel. L2 error", justify="right")

    for i, (name, err, op) in enumerate(ranked[: args.top], start=1):
        table.add_row(str(i), name, op, f"{err:.5f}")

    console.print(table)
    console.print(
        f"\n[dim]{len(ranked)} quantizable nodes. This is a cheap weight-space proxy for "
        f"quantization damage, used to pick which layers to spare — not a measurement of "
        f"end-to-end accuracy loss.[/dim]"
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Re-score the finalists on a much larger eval set.

    The search deliberately uses a small eval set — it runs once per trial, so a big one
    would dominate wall time. But a small eval set cannot resolve small accuracy
    differences, and the frontier is exactly where those differences decide things. This
    mirrors what an engineer actually does: search cheap, then confirm the shortlist
    properly before anyone ships it.
    """
    from rich.table import Table

    from anneal.core.dataset import load_evalset
    from anneal.core.ledger import Ledger
    from anneal.core.measure import Benchmarker, MeasurementError
    from anneal.core.targets import get_target
    from anneal.report import compact_label, wilson_halfwidth_pp

    console = _console()
    path = Path(args.ledger)
    if not path.is_file():
        console.print(f"[red]no such ledger:[/red] {path}")
        return 2

    ledger = Ledger.load(path)
    base = ledger.baseline
    if base is None or base.measurement is None:
        console.print("[red]ledger has no successful baseline to compare against[/red]")
        return 2

    target = get_target(ledger.target_fingerprint.get("target", "cpu-1t"))
    try:
        target.ensure_available()
    except Exception as exc:
        console.print(f"[red]cannot validate on {target.name}:[/red] {exc}")
        return 2

    # Validate the frontier plus the baseline, minus anything too slow to be a candidate.
    shortlist = {base.index: base}
    for trial in ledger.pareto_front():
        speedup = ledger.speedup_of(trial)
        if args.skip_slower_than and speedup and speedup < 1 / args.skip_slower_than:
            console.print(
                f"[dim]skipping #{trial.index} ({compact_label(trial)}) — "
                f"{1 / speedup:.1f}x slower than baseline, not a candidate[/dim]"
            )
            continue
        shortlist[trial.index] = trial

    missing = [t for t in shortlist.values() if not t.artifact.path.is_file()]
    if missing:
        console.print(
            f"[red]{len(missing)} model file(s) from this ledger no longer exist[/red]; "
            f"validation needs the candidate .onnx files the run produced"
        )
        return 2

    console.print(f"Loading eval set [cyan]{args.eval}[/cyan] …")
    evalset = load_evalset(
        args.eval, cache_dir=Path(args.cache), batch_size=args.eval_batch, limit=args.eval_limit
    )
    n = len(evalset)
    console.print(
        f"  {n} images (search used {ledger.config.get('evalset_size', '?')}) · "
        f"resolution ±{wilson_halfwidth_pp(base.measurement.accuracy, n):.2f}pp\n"
    )

    # Latency is already known from the search; a couple of runs just warms the graph.
    bench = Benchmarker(target, warmup=2, runs=3, seed=ledger.config.get("seed", 0))

    results: list[tuple[int, str, float | None, float | None, float | None]] = []
    for index in sorted(shortlist):
        trial = shortlist[index]
        label = compact_label(trial)
        console.print(f"[dim]scoring #{index} {label} …[/dim]")
        try:
            m = bench.measure(trial.artifact, evalset=evalset, record_baseline=(index == base.index))
        except MeasurementError as exc:
            console.print(f"  [red]failed:[/red] {exc}")
            results.append((index, label, None, None, None))
            continue
        results.append((index, label, m.accuracy, m.top1_agreement, ledger.speedup_of(trial)))

    search_acc = {t.index: t.measurement.accuracy for t in shortlist.values() if t.measurement}
    validated_base = next((a for i, _, a, _, _ in results if i == base.index), None)

    table = Table(title=f"validated on {n} images", header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("recipe")
    table.add_column("speedup", justify="right")
    table.add_column(f"top-1 (n={ledger.config.get('evalset_size', '?')})", justify="right")
    table.add_column(f"top-1 (n={n})", justify="right")
    table.add_column("acc pp", justify="right")
    table.add_column("agree", justify="right")

    for index, label, acc, agreement, speedup in results:
        if acc is None:
            table.add_row(str(index), label, "—", "—", "[red]failed[/red]", "—", "—")
            continue
        delta = (
            f"{(acc - validated_base) * 100:+.2f}"
            if validated_base is not None and index != base.index
            else "—"
        )
        old = search_acc.get(index)
        table.add_row(
            str(index),
            label,
            "—" if speedup is None else f"{speedup:.2f}x",
            "—" if old is None else f"{old * 100:.2f}%",
            f"{acc * 100:.2f}%",
            delta,
            "—" if agreement is None else f"{agreement:.3f}",
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Differences smaller than ±{wilson_halfwidth_pp(validated_base or 0.5, n):.2f}pp "
        f"remain unresolved even at this eval size. 'agree' is paired per-image and is the "
        f"sharper signal for whether behaviour actually changed.[/dim]"
    )

    out = path.parent / "validation.json"
    import json

    out.write_text(
        json.dumps(
            {
                "eval": args.eval,
                "n_eval": n,
                "resolution_pp": wilson_halfwidth_pp(validated_base or 0.5, n),
                "results": [
                    {
                        "index": i,
                        "recipe": lbl,
                        "accuracy": a,
                        "top1_agreement": ag,
                        "speedup": sp,
                    }
                    for i, lbl, a, ag, sp in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[dim]validation: {out}[/dim]")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from anneal.core.ledger import Ledger
    from anneal.report import console_table, write_report

    console = _console()
    path = Path(args.ledger)
    if not path.is_file():
        console.print(f"[red]no such ledger:[/red] {path}")
        return 2

    ledger = Ledger.load(path)
    console.print(console_table(ledger, only_frontier=args.frontier_only))
    paths = write_report(ledger, path.parent)
    console.print(f"\n[dim]rewrote {paths['report']}[/dim]")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anneal",
        description="An agent that optimises neural networks for the hardware they'll "
        "actually run on.",
    )
    parser.add_argument("--version", action="version", version=f"anneal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an optimisation search")
    run.add_argument(
        "--model",
        default="torchvision:resnet18",
        help="'torchvision:<name>' or a path to an .onnx file",
    )
    run.add_argument("--target", default=None, help="target name (see `anneal targets`)")
    run.add_argument(
        "--eval",
        default="imagenette",
        help="'imagenette[:160|:320]', 'synthetic', a dataset path, or 'none'",
    )
    run.add_argument("--eval-limit", type=int, default=256, help="max eval images")
    run.add_argument("--eval-batch", type=int, default=16, help="eval batch size")
    run.add_argument("--policy", default="heuristic", choices=["heuristic", "claude"])
    run.add_argument("--llm-model", default="claude-sonnet-5", help="model id for --policy claude")
    run.add_argument("--budget", type=int, default=10, help="trials after the baseline")
    run.add_argument("--batch-size", type=int, default=1, help="inference batch size")
    run.add_argument("--image-size", type=int, default=224)
    run.add_argument("--warmup", type=int, default=10, help="discarded warm-up iterations")
    run.add_argument("--runs", type=int, default=60, help="timed iterations per trial")
    run.add_argument("--calib-samples", type=int, default=64)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=1.0,
        metavar="PP",
        help="accuracy budget in percentage points vs baseline",
    )
    run.add_argument("--min-accuracy", type=float, default=None, help="absolute floor, 0-1")
    run.add_argument("--max-size-mb", type=float, default=None)
    run.add_argument("--max-latency-ms", type=float, default=None)
    run.add_argument("--out", default=None, help="output directory")
    run.add_argument("--cache", default=str(DEFAULT_CACHE), help="dataset cache directory")
    run.set_defaults(func=cmd_run)

    targets = sub.add_parser("targets", help="list execution targets")
    targets.set_defaults(func=cmd_targets)

    transforms = sub.add_parser("transforms", help="list the transform action space")
    transforms.set_defaults(func=cmd_transforms)

    sens = sub.add_parser(
        "sensitivity", help="rank layers by INT8 weight-quantization error"
    )
    sens.add_argument("model", help="path to an .onnx file")
    sens.add_argument("--top", type=int, default=20)
    sens.add_argument("--per-tensor", action="store_true", help="per-tensor instead of per-channel")
    sens.set_defaults(func=cmd_sensitivity)

    val = sub.add_parser(
        "validate", help="re-score a run's frontier on a much larger eval set"
    )
    val.add_argument("ledger", help="path to ledger.json")
    val.add_argument("--eval", default="imagenette")
    val.add_argument("--eval-limit", type=int, default=None, help="default: the whole set")
    val.add_argument("--eval-batch", type=int, default=32)
    val.add_argument(
        "--skip-slower-than",
        type=float,
        default=2.0,
        metavar="X",
        help="don't spend eval time on candidates more than X times slower than baseline",
    )
    val.add_argument("--cache", default=str(DEFAULT_CACHE))
    val.set_defaults(func=cmd_validate)

    rep = sub.add_parser("report", help="re-render a report from a ledger")
    rep.add_argument("ledger", help="path to ledger.json")
    rep.add_argument("--frontier-only", action="store_true")
    rep.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
