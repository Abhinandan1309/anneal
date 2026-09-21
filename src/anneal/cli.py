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
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.targets import TargetUnavailable, default_target, get_target
    from anneal.core.artifact import model_batch_dim, sample_shape
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

    from anneal.core.environment import snapshot, warnings_for

    for warning in warnings_for(snapshot()):
        console.print(f"[bold red]environment:[/bold red] {warning}")

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
    calibset = None
    if args.eval != "none":
        console.print(f"Loading eval set [cyan]{args.eval}[/cyan] …")
        try:
            evalset = load_evalset(
                args.eval,
                cache_dir=cache,
                batch_size=eval_batch,
                limit=args.eval_limit,
                sample_shape=sample_shape(baseline.path),
            )
        except Exception as exc:
            console.print(f"[red]could not load eval set:[/red] {exc}")
            return 2
        console.print(f"  {len(evalset)} images")
        calibset = load_calibset(
            args.eval,
            cache_dir=cache,
            batch_size=eval_batch,
            limit=args.calib_samples,
            sample_shape=sample_shape(baseline.path),
        )
        if calibset is None:
            console.print(
                "  [yellow]no separate calibration split found; static quantization will "
                "calibrate on eval images, which flatters its accuracy[/yellow]"
            )
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

    measured_ranking = None
    if args.sensitivity:
        from anneal.core.transforms import load_measured_ranking

        try:
            measured_ranking = load_measured_ranking(Path(args.sensitivity))
        except (OSError, ValueError, KeyError) as exc:
            console.print(f"[red]could not read sensitivity sweep:[/red] {exc}")
            return 2
        console.print(
            f"Using measured layer ranking from [cyan]{args.sensitivity}[/cyan] "
            f"({len(measured_ranking)} layers)"
        )

    config = RunConfig(
        workdir=outdir,
        measured_ranking=measured_ranking,
        measured_ranking_source=str(args.sensitivity) if args.sensitivity else None,
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
        elif kind == "drift":
            moved = payload["drift"]
            bad = moved > 0.10 or bool(payload["warnings"])
            colour = "red" if bad else "green"
            console.print(
                f"[{colour}]baseline re-measured: {payload['start_ms']:.2f}ms -> "
                f"{payload['end_ms']:.2f}ms ({moved * 100:.1f}% drift)[/{colour}]"
            )
            if bad:
                console.print(
                    "[bold red]latencies in this run are not trustworthy: the machine's "
                    "performance state changed or was degraded while it ran[/bold red]"
                )

    run = OptimizationRun(
        baseline, target, policy, config, evalset=evalset, calibset=calibset, on_event=on_event
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

    if not args.measured:
        table = Table(
            title=f"INT8 weight-quantization sensitivity (proxy) - {path.name}",
            header_style="bold",
        )
        table.add_column("rank", justify="right", style="dim")
        table.add_column("node", overflow="fold")
        table.add_column("op")
        table.add_column("rel. L2 error", justify="right")

        for i, (name, err, op) in enumerate(ranked[: args.top], start=1):
            table.add_row(str(i), name, op, f"{err:.5f}")

        console.print(table)
        console.print(
            f"\n[dim]{len(ranked)} quantizable nodes. This is a cheap weight-space proxy "
            f"for quantization damage, not a measurement. Pass --measured to quantize each "
            f"layer alone and find out how well the proxy actually predicts it.[/dim]"
        )
        return 0

    # ----- measured sweep -------------------------------------------------
    import json

    from anneal.core.dataset import load_evalset
    from anneal.core.sensitivity import measured_sensitivity, proxy_agreement
    from anneal.core.targets import TargetUnavailable, default_target, get_target
    from anneal.core.transforms import TransformContext
    from anneal.models import load_onnx

    try:
        target = get_target(args.target) if args.target else default_target()
        target.ensure_available()
    except (KeyError, TargetUnavailable) as exc:
        console.print(f"[red]target error:[/red] {exc}")
        return 2

    console.print(f"Loading eval set [cyan]{args.eval}[/cyan] …")
    from anneal.core.artifact import sample_shape

    evalset = load_evalset(
        args.eval,
        cache_dir=Path(args.cache),
        batch_size=args.eval_batch,
        limit=args.eval_limit,
        sample_shape=sample_shape(path),
    )
    n_layers = min(len(ranked), args.top)
    console.print(
        f"  {len(evalset)} images · quantizing {n_layers} layer(s) one at a time "
        f"({n_layers} eval passes)\n"
    )

    workdir = Path(args.out) if args.out else path.parent / "sensitivity"
    ctx = TransformContext(workdir=workdir, evalset=evalset)

    def progress(i: int, total: int, node: str) -> None:
        console.print(f"[dim]  [{i}/{total}] {node}[/dim]")

    results = measured_sensitivity(
        load_onnx(path),
        target,
        evalset,
        ctx,
        per_channel=not args.per_tensor,
        limit=args.top,
        on_progress=progress,
    )

    by_measured = sorted(
        results, key=lambda r: (r.changed_fraction if r.measured else -1), reverse=True
    )

    table = Table(
        title=f"measured vs predicted quantization damage - {path.name}", header_style="bold"
    )
    table.add_column("node", overflow="fold")
    table.add_column("op")
    table.add_column("proxy err", justify="right")
    table.add_column("proxy rank", justify="right")
    table.add_column("preds changed", justify="right")
    table.add_column("acc pp", justify="right")

    proxy_rank = {r.node: i for i, r in enumerate(sorted(results, key=lambda r: -r.proxy_error), 1)}
    for r in by_measured:
        if r.error:
            table.add_row(r.node, r.op_type, f"{r.proxy_error:.5f}", "—", "[red]failed[/red]", "—")
            continue
        table.add_row(
            r.node,
            r.op_type,
            f"{r.proxy_error:.5f}",
            str(proxy_rank.get(r.node, "—")),
            f"{(r.changed_fraction or 0) * 100:.1f}%",
            "—" if r.accuracy_drop_pp is None else f"{-r.accuracy_drop_pp:+.2f}",
        )

    console.print()
    console.print(table)

    verdict = proxy_agreement(results)
    rho = verdict["spearman"]
    console.print()
    if rho is None:
        console.print(f"[yellow]Proxy vs measured: {verdict['verdict']}[/yellow]")
    else:
        colour = "green" if rho >= 0.7 else ("yellow" if rho >= 0.4 else "red")
        console.print(
            f"[bold]Proxy vs measured:[/bold] Spearman rho = "
            f"[{colour}]{rho:+.3f}[/{colour}] over {verdict['n']} layers — {verdict['verdict']}"
        )
        console.print(
            f"[dim]Top-5 overlap: {verdict['top5_overlap']}/5 layers appear in both "
            f"rankings.[/dim]"
        )

    out_json = workdir / "sensitivity.json"
    workdir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "model": str(path),
                "target": target.name,
                "n_eval": len(evalset),
                "layers": [r.to_dict() for r in results],
                "proxy_agreement": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[dim]written: {out_json}[/dim]")
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
    from anneal.core.artifact import sample_shape

    evalset = load_evalset(
        args.eval,
        cache_dir=Path(args.cache),
        batch_size=args.eval_batch,
        limit=args.eval_limit,
        sample_shape=sample_shape(base.artifact.path),
    )
    n = len(evalset)
    mode = "cached" if getattr(evalset, "caching", True) else "streamed (too large to cache)"
    console.print(
        f"  {n} images (search used {ledger.config.get('evalset_size', '?')}) · "
        f"resolution ±{wilson_halfwidth_pp(base.measurement.accuracy, n):.2f}pp · {mode}\n"
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


def cmd_profile(args: argparse.Namespace) -> int:
    """Attribute a model's runtime to operator types and individual nodes."""
    from rich.table import Table

    from anneal.core.profile import diff_profiles, profile_model
    from anneal.core.targets import TargetUnavailable, default_target, get_target
    from anneal.models import load_onnx

    console = _console()
    path = Path(args.model)
    if not path.is_file():
        console.print(f"[red]no such file:[/red] {path}")
        return 2

    try:
        target = get_target(args.target) if args.target else default_target()
        target.ensure_available()
    except (KeyError, TargetUnavailable) as exc:
        console.print(f"[red]target error:[/red] {exc}")
        return 2

    console.print(f"Profiling [cyan]{path.name}[/cyan] on [cyan]{target.name}[/cyan] …")
    profile = profile_model(
        load_onnx(path), target, runs=args.runs, batch_size=args.batch_size
    )

    table = Table(title=f"time by operator type - {path.name}", header_style="bold")
    table.add_column("op", overflow="fold")
    table.add_column("share", justify="right")
    table.add_column("total ms", justify="right")
    table.add_column("nodes", justify="right")
    table.add_column("mean us/call", justify="right")

    for stat in profile.by_op[: args.top]:
        bar = "#" * max(1, round(stat.share * 24)) if stat.share > 0.01 else ""
        table.add_row(
            stat.op_type,
            f"{stat.share * 100:5.1f}% {bar}",
            f"{stat.total_us / 1000:.1f}",
            str(stat.count),
            f"{stat.mean_us:.1f}",
        )
    console.print(table)

    if args.nodes:
        node_table = Table(title="slowest individual nodes", header_style="bold")
        node_table.add_column("node", overflow="fold")
        node_table.add_column("op")
        node_table.add_column("share", justify="right")
        node_table.add_column("total ms", justify="right")
        for stat in profile.by_node[: args.top]:
            node_table.add_row(
                stat.name, stat.op_type, f"{stat.share * 100:.1f}%", f"{stat.total_us / 1000:.1f}"
            )
        console.print(node_table)

    if args.against:
        other_path = Path(args.against)
        if not other_path.is_file():
            console.print(f"[red]no such file:[/red] {other_path}")
            return 2
        console.print(f"\nProfiling [cyan]{other_path.name}[/cyan] for comparison …")
        other = profile_model(
            load_onnx(other_path), target, runs=args.runs, batch_size=args.batch_size
        )

        diff_table = Table(
            title=f"{path.name}  ->  {other_path.name}  (per run)", header_style="bold"
        )
        diff_table.add_column("op", overflow="fold")
        diff_table.add_column("before ms", justify="right")
        diff_table.add_column("after ms", justify="right")
        diff_table.add_column("delta ms", justify="right")

        for delta in diff_profiles(profile, other):
            if abs(delta.delta_us) < 50:  # ignore sub-0.05ms noise
                continue
            marker = ""
            if delta.appeared:
                marker = " [yellow](new)[/yellow]"
            elif delta.vanished:
                marker = " [dim](gone)[/dim]"
            colour = "red" if delta.delta_us > 0 else "green"
            diff_table.add_row(
                delta.op_type + marker,
                f"{delta.before_us / 1000:.2f}",
                f"{delta.after_us / 1000:.2f}",
                f"[{colour}]{delta.delta_us / 1000:+.2f}[/{colour}]",
            )
        console.print()
        console.print(diff_table)

    console.print(
        "\n[dim]Profiling instrumentation inflates absolute times; use these shares for "
        "attribution and `anneal run` for latency.[/dim]"
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Re-measure a run's frontier on a different target.

    This is the project's central claim, made falsifiable: if an optimisation recipe's
    ranking were a property of the model, this table would be boring.
    """
    from rich.table import Table

    from anneal.core.ledger import Ledger
    from anneal.core.measure import Benchmarker, MeasurementError
    from anneal.core.targets import TargetUnavailable, get_target
    from anneal.report import compact_label

    console = _console()
    path = Path(args.ledger)
    if not path.is_file():
        console.print(f"[red]no such ledger:[/red] {path}")
        return 2

    ledger = Ledger.load(path)
    base = ledger.baseline
    if base is None:
        console.print("[red]ledger has no successful baseline[/red]")
        return 2

    origin = ledger.target_fingerprint.get("target", "?")
    try:
        target = get_target(args.target)
        target.ensure_available()
    except (KeyError, TargetUnavailable) as exc:
        console.print(f"[red]target error:[/red] {exc}")
        return 2

    shortlist = [base] + [t for t in ledger.pareto_front() if t.index != base.index]
    missing = [t for t in shortlist if not t.artifact.path.is_file()]
    if missing:
        console.print(f"[red]{len(missing)} candidate .onnx file(s) are gone[/red]")
        return 2

    console.print(f"Re-measuring {len(shortlist)} models: [cyan]{origin}[/cyan] -> [cyan]{target.name}[/cyan]\n")
    bench = Benchmarker(target, warmup=args.warmup, runs=args.runs)

    rows: list[tuple[Any, float | None]] = []
    for trial in shortlist:
        try:
            m = bench.measure(trial.artifact)
            rows.append((trial, m.latency_ms_p50))
        except MeasurementError as exc:
            console.print(f"[red]#{trial.index} failed:[/red] {exc}")
            rows.append((trial, None))

    new_base = next((lat for t, lat in rows if t.index == base.index), None)

    table = Table(title=f"{origin} vs {target.name}", header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("recipe")
    table.add_column(f"{origin} p50", justify="right")
    table.add_column(f"{origin} x", justify="right")
    table.add_column(f"{target.name} p50", justify="right")
    table.add_column(f"{target.name} x", justify="right")
    table.add_column("verdict", justify="left")

    for trial, latency in rows:
        old_speedup = ledger.speedup_of(trial)
        new_speedup = (new_base / latency) if (new_base and latency) else None

        verdict = "—"
        if old_speedup is not None and new_speedup is not None:
            if (old_speedup > 1.02) != (new_speedup > 1.02):
                verdict = "[bold red]flips[/bold red]"
            elif abs(new_speedup - old_speedup) / max(old_speedup, 1e-9) > 0.15:
                verdict = "[yellow]shifts[/yellow]"
            else:
                verdict = "[green]holds[/green]"

        table.add_row(
            str(trial.index),
            compact_label(trial),
            f"{trial.measurement.latency_ms_p50:.2f}" if trial.measurement else "—",
            f"{old_speedup:.2f}x" if old_speedup else "—",
            f"{latency:.2f}" if latency else "[red]failed[/red]",
            f"{new_speedup:.2f}x" if new_speedup else "—",
            verdict,
        )

    console.print(table)
    console.print(
        f"\n[dim]Accuracy is a property of the graph and does not change with target, so "
        f"only latency is re-measured here. Recipes marked 'flips' reversed sign: they "
        f"helped on {origin} and hurt on {target.name}, or the reverse.[/dim]"
    )
    return 0


RECIPE_TEMPLATE = '''"""Reproduce trial {index} from Anneal run {run_id}.

    {label}

Measured on {target} ({machine}), onnxruntime {ort}:

{measurements}

This script is self-contained: it rebuilds the model from the baseline rather than
trusting a binary someone emailed you. Run it and you get the same graph, or you find
out that something in your toolchain differs from the one that produced these numbers —
which is the more useful outcome of the two.
"""

from pathlib import Path

from anneal.core.artifact import ModelArtifact
from anneal.core.transforms import TransformContext, apply_transform
{dataset_import}
BASELINE = Path({baseline!r})
OUTPUT = Path({output!r})


def build() -> Path:
    artifact = ModelArtifact(path=BASELINE)
    ctx = TransformContext(
        workdir=OUTPUT.parent / "_work",{evalset_arg}
    )

{steps}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(artifact.path.read_bytes())
    print(f"wrote {{OUTPUT}} ({{OUTPUT.stat().st_size / 1e6:.1f}} MB)")
    return OUTPUT


if __name__ == "__main__":
    build()
'''


def cmd_export(args: argparse.Namespace) -> int:
    """Emit a standalone script that rebuilds one trial's model from the baseline."""
    from anneal.core.ledger import Ledger
    from anneal.report import compact_label

    console = _console()
    path = Path(args.ledger)
    if not path.is_file():
        console.print(f"[red]no such ledger:[/red] {path}")
        return 2

    ledger = Ledger.load(path)
    base = ledger.baseline
    if base is None:
        console.print("[red]ledger has no baseline to rebuild from[/red]")
        return 2

    if args.trial is not None:
        trial = next((t for t in ledger.trials if t.index == args.trial), None)
        if trial is None:
            console.print(f"[red]no trial {args.trial} in this ledger[/red]")
            return 2
        if not trial.ok:
            console.print(f"[red]trial {args.trial} failed; there is nothing to reproduce[/red]")
            return 2
    else:
        trial = ledger.best_under_constraints(min_accuracy=_floor_from_ledger(ledger))
        if trial is None:
            console.print("[red]no trial satisfies this run's constraints[/red]")
            return 2
        console.print(f"[dim]no --trial given; exporting the recommended pick, #{trial.index}[/dim]")

    if not trial.artifact.lineage:
        console.print(
            "[yellow]that trial is the unmodified baseline — there is no recipe to "
            "export[/yellow]"
        )
        return 1

    needs_calibration = any(
        t.name in ("quantize_static_int8",) for t in trial.artifact.lineage
    )

    steps = []
    for i, record in enumerate(trial.artifact.lineage, start=1):
        steps.append(f"    # step {i}: {record.name}")
        steps.append(
            f"    artifact = apply_transform(\n"
            f"        {record.name!r},\n"
            f"        {record.params!r},\n"
            f"        artifact,\n"
            f"        ctx,\n"
            f"    )"
        )
        steps.append("")

    m = trial.measurement
    measurements = "\n".join(
        f"    {line}"
        for line in [
            f"p50 latency   {m.latency_ms_p50:.2f} ms   ({ledger.speedup_of(trial):.2f}x vs baseline)",
            f"p99 latency   {m.latency_ms_p99:.2f} ms",
            f"size          {m.size_mb:.2f} MB",
            (
                f"top-1         {m.accuracy * 100:.2f}%  on {m.n_eval} images"
                if m.accuracy is not None
                else "top-1         not scored"
            ),
            (
                f"agreement     {m.top1_agreement:.3f} of predictions match the baseline"
                if m.top1_agreement is not None
                else "agreement     not measured"
            ),
        ]
    )

    fp = ledger.target_fingerprint
    script = RECIPE_TEMPLATE.format(
        index=trial.index,
        run_id=ledger.run_id,
        label=trial.artifact.label,
        target=fp.get("target", "?"),
        machine=fp.get("processor") or fp.get("machine", "?"),
        ort=fp.get("onnxruntime", "?"),
        measurements=measurements,
        # POSIX separators so the generated script is not Windows-only.
        baseline=base.artifact.path.as_posix(),
        output=Path(args.model_out or f"anneal-trial{trial.index}.onnx").as_posix(),
        dataset_import=(
            "from anneal.core.dataset import load_calibset\n" if needs_calibration else ""
        ),
        evalset_arg=(
            "\n        # static quantization calibrates on real data from the deployment "
            "distribution,\n        # kept separate from anything you evaluate on.\n"
            '        calibset=load_calibset("imagenette", cache_dir=Path.home() / ".anneal_cache",\n'
            "                               batch_size=32, limit=64),"
            if needs_calibration
            else ""
        ),
        steps="\n".join(steps),
    )

    out = Path(args.out) if args.out else path.parent / f"reproduce_trial{trial.index}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")

    console.print(f"[bold green]exported[/bold green] {compact_label(trial)}")
    console.print(f"[dim]{out}[/dim]")
    return 0


def _floor_from_ledger(ledger) -> float | None:
    constraints = ledger.config.get("constraints") or {}
    base = ledger.baseline
    floors = []
    if constraints.get("min_accuracy") is not None:
        floors.append(float(constraints["min_accuracy"]))
    drop = constraints.get("max_accuracy_drop_pp")
    if drop is not None and base and base.measurement and base.measurement.accuracy is not None:
        floors.append(base.measurement.accuracy - float(drop) / 100.0)
    return max(floors) if floors else None


def cmd_audit(args: argparse.Namespace) -> int:
    """Check an optimised model against its original, whichever tool produced it."""
    import json

    from anneal.core.audit import audit
    from anneal.core.dataset import IMAGENETTE_CLASS_NAMES, load_evalset
    from anneal.core.measure import MeasurementError
    from anneal.core.targets import TargetUnavailable, default_target, get_target
    from anneal.models import load_onnx

    console = _console()
    try:
        original = load_onnx(Path(args.original))
        candidate = load_onnx(Path(args.candidate))
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    try:
        target = get_target(args.target) if args.target else default_target()
        target.ensure_available()
    except (KeyError, TargetUnavailable) as exc:
        console.print(f"[red]target error:[/red] {exc}")
        return 2

    console.print(f"Loading eval set [cyan]{args.eval}[/cyan] …")
    from anneal.core.artifact import sample_shape

    evalset = load_evalset(
        args.eval,
        cache_dir=Path(args.cache),
        batch_size=args.eval_batch,
        limit=args.eval_limit,
        sample_shape=sample_shape(original.path),
    )
    if getattr(evalset, "synthetic", False):
        console.print(
            "[yellow]synthetic eval set: the accuracy half of this audit is meaningless[/yellow]"
        )
    console.print(
        f"Auditing [cyan]{candidate.path.name}[/cyan] against [cyan]{original.path.name}[/cyan] "
        f"on [cyan]{target.name}[/cyan], {len(evalset)} images …\n"
    )

    try:
        result = audit(
            original,
            candidate,
            target,
            evalset,
            warmup=args.warmup,
            runs=args.runs,
            class_names=IMAGENETTE_CLASS_NAMES if args.eval.startswith("imagenette") else None,
            profile=args.profile,
            sequential=args.sequential,
            budget_pp=args.max_accuracy_drop,
            alpha=args.alpha,
        )
    except MeasurementError as exc:
        console.print(f"[red]audit failed:[/red] {exc}")
        return 1

    console.print("[bold]Verdict[/bold]")
    for line in result.verdict(max_drop_pp=args.max_accuracy_drop):
        console.print(f"  - {line}")
    console.print(
        f"\n[dim]regressions {result.regressions} · fixes {result.fixes} · changed "
        f"{result.changed}/{result.n} · McNemar p = {result.p_value:.3g}[/dim]"
    )

    out = Path(args.out) if args.out else candidate.path.parent / f"audit-{candidate.path.stem}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    (out / "audit.md").write_text(result.markdown(), encoding="utf-8")
    console.print(f"[dim]written: {out / 'audit.md'}[/dim]")

    # Non-zero exit when the audit finds a real problem, so this can gate a CI pipeline.
    # Exit codes, for gating CI: 0 pass, 3 a real problem, 4 could not decide.
    if result.speedup_p50 < 1.0:
        return 3
    if result.sequential is not None:
        return {"accept": 0, "reject": 3, "undecided": 4}[result.sequential.decision]
    failed = result.significant and result.delta_pp < -args.max_accuracy_drop
    return 3 if failed else 0


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
    run.add_argument(
        "--sensitivity",
        default=None,
        metavar="JSON",
        help="reuse a saved `anneal sensitivity --measured` result to rank layers for "
        "selective quantization, instead of sweeping again",
    )
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
    sens.add_argument(
        "--measured",
        action="store_true",
        help="quantize each layer alone and measure the real damage, then score the proxy "
        "against it (costs one eval pass per layer)",
    )
    sens.add_argument("--target", default=None)
    sens.add_argument("--eval", default="imagenette")
    sens.add_argument("--eval-limit", type=int, default=256)
    sens.add_argument("--eval-batch", type=int, default=32)
    sens.add_argument("--out", default=None, help="where to write candidates and results")
    sens.add_argument("--cache", default=str(DEFAULT_CACHE))
    sens.set_defaults(func=cmd_sensitivity)

    prof = sub.add_parser("profile", help="attribute runtime to operator types and nodes")
    prof.add_argument("model", help="path to an .onnx file")
    prof.add_argument("--against", default=None, help="second .onnx to diff against")
    prof.add_argument("--target", default=None)
    prof.add_argument("--runs", type=int, default=20)
    prof.add_argument("--batch-size", type=int, default=1)
    prof.add_argument("--top", type=int, default=12)
    prof.add_argument("--nodes", action="store_true", help="also list the slowest nodes")
    prof.set_defaults(func=cmd_profile)

    cmp_ = sub.add_parser(
        "compare", help="re-measure a run's frontier on a different target"
    )
    cmp_.add_argument("ledger", help="path to ledger.json")
    cmp_.add_argument("--target", required=True, help="the target to re-measure on")
    cmp_.add_argument("--warmup", type=int, default=10)
    cmp_.add_argument("--runs", type=int, default=50)
    cmp_.set_defaults(func=cmd_compare)

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

    aud = sub.add_parser(
        "audit",
        help="check any optimised model against its original (works on other tools' output)",
    )
    aud.add_argument("original", help="the unoptimised .onnx")
    aud.add_argument("candidate", help="the optimised .onnx, from any tool")
    aud.add_argument("--target", default=None)
    aud.add_argument("--eval", default="imagenette")
    aud.add_argument("--eval-limit", type=int, default=1024)
    aud.add_argument("--eval-batch", type=int, default=32)
    aud.add_argument("--warmup", type=int, default=10)
    aud.add_argument("--runs", type=int, default=50)
    aud.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=1.0,
        metavar="PP",
        help="exit non-zero if a statistically significant loss exceeds this",
    )
    aud.add_argument("--profile", action="store_true", help="also attribute the latency change to operators")
    aud.add_argument(
        "--sequential",
        action="store_true",
        help="stop evaluating as soon as an anytime-valid test decides whether the accuracy "
        "change is within --max-accuracy-drop (exit 0 accept, 3 reject, 4 undecided)",
    )
    aud.add_argument("--alpha", type=float, default=0.05, help="error rate per sequential decision")
    aud.add_argument("--out", default=None)
    aud.add_argument("--cache", default=str(DEFAULT_CACHE))
    aud.set_defaults(func=cmd_audit)

    exp = sub.add_parser(
        "export", help="emit a standalone script that rebuilds one trial's model"
    )
    exp.add_argument("ledger", help="path to ledger.json")
    exp.add_argument("--trial", type=int, default=None, help="default: the recommended pick")
    exp.add_argument("--out", default=None, help="where to write the script")
    exp.add_argument("--model-out", default=None, help="path the script will write the model to")
    exp.set_defaults(func=cmd_export)

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
