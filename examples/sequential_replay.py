"""Replay the sequential test on *real* per-image outcomes, over many random orders.

The simulation study draws synthetic outcomes at measured rates. This checks the same
thing on the real thing: each model's right/wrong result on all 3,925 Imagenette
validation images, computed once, then streamed to the sequential test in 1,000 random
orders. The full-set answer is the reference each early decision is judged against.

Outcomes are saved to sequential_study/replay_outcomes.npz so the replay can be rerun
without the models:

    python examples/sequential_replay.py                 # compute outcomes, then replay
    python examples/sequential_replay.py --from-outcomes  # replay only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "sequential_study"
BUDGET_PP = 1.0


def compute_outcomes() -> dict[str, np.ndarray]:
    from anneal.core.artifact import ModelArtifact, TransformRecord, sample_shape
    from anneal.core.audit import predict
    from anneal.core.dataset import load_evalset
    from anneal.core.targets import get_target
    from anneal.core.transforms import TransformContext

    base = ModelArtifact(path=HERE / "resnet18-cpu1t" / "models" / "resnet18-fp32.onnx")
    ctx = TransformContext(workdir=HERE / "static_ab_candidates")

    def static(**over):
        params = {"per_channel": True, "reduce_range": False, "calibrate_method": "minmax",
                  "calib_samples": 64, "activation_type": "uint8", **over}
        return ModelArtifact(path=ctx.path_for(base, TransformRecord("quantize_static_int8", params)))

    models = {
        "fp32": base,
        "static INT8, full-range per-channel": static(),
        "static INT8, reduce_range": static(reduce_range=True),
        "Olive default static INT8": ModelArtifact(
            path=HERE.parent / "scratch" / "olive_out" / "default" / "model.onnx"
        ),
    }
    evalset = load_evalset("imagenette", cache_dir=Path.home() / ".anneal_cache",
                           batch_size=32, sample_shape=sample_shape(base.path))
    target = get_target("cpu-4t")
    right = {}
    labels = None
    for name, model in models.items():
        if not model.path.exists():
            raise FileNotFoundError(f"{name}: {model.path} (run static_recipe_ab.py / Olive first)")
        pred, y = predict(model, target, evalset)
        labels = y if labels is None else labels
        right[name] = pred == y
        print(f"{name:40s} top-1 {right[name].mean() * 100:.2f}%  (n={len(y)})")
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "replay_outcomes.npz", **{k: v for k, v in right.items()})
    return right


def replay(right: dict[str, np.ndarray], reps: int, alpha: float, seed: int) -> list[dict]:
    from anneal.core.sequential import paired_scores, run_sequential

    rng = np.random.default_rng(seed)
    base = right["fp32"]
    n = len(base)
    rows = []
    for name, cand in right.items():
        if name == "fp32":
            continue
        scores = paired_scores(base, cand)
        full_delta = scores.mean() * 100
        truth = "accept" if full_delta >= -BUDGET_PP else "reject"
        decisions, used, fixed256 = [], [], []
        for _ in range(reps):
            order = rng.permutation(n)
            res = run_sequential(scores[order], budget_pp=BUDGET_PP, alpha=alpha, track_ci=False)
            decisions.append(res.decision)
            used.append(res.n)
            fixed256.append("accept" if scores[order[:256]].mean() * 100 >= -BUDGET_PP else "reject")
        used = np.array(used)
        rows.append({
            "candidate": name,
            "full_set_delta_pp": full_delta,
            "full_set_decision": truth,
            "regressions": int(np.sum(scores == -1)),
            "fixes": int(np.sum(scores == 1)),
            "sequential_agrees": float(np.mean([d == truth for d in decisions])),
            "sequential_contradicts": float(np.mean([d not in (truth, "undecided") for d in decisions])),
            "sequential_undecided": float(np.mean([d == "undecided" for d in decisions])),
            "median_n": float(np.median(used)),
            "p90_n": float(np.percentile(used, 90)),
            "fixed256_contradicts": float(np.mean([d != truth for d in fixed256])),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-outcomes", action="store_true")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    if args.from_outcomes:
        data = np.load(OUT / "replay_outcomes.npz")
        right = {k: data[k] for k in data.files}
    else:
        right = compute_outcomes()

    rows = replay(right, args.reps, args.alpha, args.seed)
    (OUT / "replay_results.json").write_text(
        json.dumps({"reps": args.reps, "alpha": args.alpha, "budget_pp": BUDGET_PP,
                    "n_full": int(len(right["fp32"])), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# Sequential test replayed on real outcomes ({args.reps} random orders each)",
        "",
        f"ResNet-18 on all {len(right['fp32'])} Imagenette validation images. Budget "
        f"{BUDGET_PP:.1f}pp, alpha = {args.alpha}. The reference is the full-set decision.",
        "",
        "| candidate | full-set change | reference | sequential agrees | contradicts | undecided "
        "| median n | p90 n | fixed-256 contradicts |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['candidate']} | {r['full_set_delta_pp']:+.2f}pp | {r['full_set_decision']} | "
            f"{r['sequential_agrees'] * 100:.1f}% | {r['sequential_contradicts'] * 100:.1f}% | "
            f"{r['sequential_undecided'] * 100:.1f}% | {r['median_n']:.0f} | {r['p90_n']:.0f} | "
            f"{r['fixed256_contradicts'] * 100:.1f}% |"
        )
    (OUT / "replay_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
