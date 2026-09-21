"""How many images does an accept/reject decision really need?

Compares three ways of deciding whether an optimised model stays within a 1pp accuracy
budget, on simulated evaluations whose per-image fix/break rates are taken from real
measurements in this repository:

* **fixed-256**   — what Anneal's search did: accept if the point estimate on 256 images is
                    within budget. Cheap, no error control.
* **fixed-3925**  — the same rule on the whole Imagenette validation set. Expensive.
* **sequential**  — the anytime-valid betting test in anneal.core.sequential, stopping as
                    soon as it can decide, with each decision's error at most alpha.

Truth is known in simulation, so a decision is an error when it contradicts it. Scenarios
whose true change sits exactly on the budget have no right answer and are reported apart.

    python examples/sequential_study.py --reps 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anneal.core.sequential import fixed_n_for_power, run_sequential

HERE = Path(__file__).parent
N_FULL = 3925
BUDGET_PP = 1.0

# (name, P(candidate fixes an image), P(candidate breaks an image), source)
SCENARIOS = [
    ("static INT8, full-range per-channel", 35 / 1024, 78 / 1024,
     "ResNet-18 recipe A/B, n=1024 (-4.20pp)"),
    ("static INT8, reduce_range", 11 / 1024, 7 / 1024,
     "ResNet-18 recipe A/B, n=1024 (+0.39pp)"),
    ("Olive default static INT8", 80 / 3925, 67 / 3925,
     "anneal audit of Olive's model, n=3925 (+0.33pp)"),
    ("near budget, acceptable", 0.045, 0.050, "synthetic: -0.5pp, 9.5% discordant"),
    ("near budget, unacceptable", 0.0425, 0.0575, "synthetic: -1.5pp, 10% discordant"),
    ("exactly on budget", 0.045, 0.055, "synthetic: -1.0pp, 10% discordant"),
]


def draw(rng, n, p_fix, p_break):
    u = rng.random(n)
    return np.where(u < p_fix, 1, np.where(u < p_fix + p_break, -1, 0)).astype(int)


def fixed_rule(scores: np.ndarray) -> str:
    return "accept" if scores.mean() * 100 >= -BUDGET_PP else "reject"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for name, p_fix, p_break, source in SCENARIOS:
        delta_pp = (p_fix - p_break) * 100
        truth = (
            "accept" if delta_pp > -BUDGET_PP + 1e-9
            else "reject" if delta_pp < -BUDGET_PP - 1e-9
            else "boundary"
        )
        seq_n, seq_dec, f256, ffull = [], [], [], []
        for _ in range(args.reps):
            scores = draw(rng, N_FULL, p_fix, p_break)
            res = run_sequential(scores, budget_pp=BUDGET_PP, alpha=args.alpha, track_ci=False)
            seq_n.append(res.n)
            seq_dec.append(res.decision)
            f256.append(fixed_rule(scores[:256]))
            ffull.append(fixed_rule(scores))

        def error_rate(decisions):
            if truth == "boundary":
                return None
            return float(np.mean([d not in (truth, "undecided") for d in decisions]))

        seq_n = np.array(seq_n)
        row = {
            "scenario": name,
            "source": source,
            "true_delta_pp": delta_pp,
            "discordance": p_fix + p_break,
            "truth": truth,
            "sequential": {
                "error_rate": error_rate(seq_dec),
                "undecided_rate": float(np.mean([d == "undecided" for d in seq_dec])),
                "accept_rate": float(np.mean([d == "accept" for d in seq_dec])),
                "mean_n": float(seq_n.mean()),
                "median_n": float(np.median(seq_n)),
                "p90_n": float(np.percentile(seq_n, 90)),
            },
            "fixed_256": {"error_rate": error_rate(f256),
                          "accept_rate": float(np.mean([d == "accept" for d in f256]))},
            "fixed_3925": {"error_rate": error_rate(ffull),
                           "accept_rate": float(np.mean([d == "accept" for d in ffull]))},
            "fixed_n_for_90pct_power": (
                None if truth == "boundary"
                else fixed_n_for_power(delta_pp / 100, p_fix + p_break, BUDGET_PP / 100, args.alpha)
            ),
        }
        rows.append(row)

    out_dir = HERE / "sequential_study"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"reps": args.reps, "alpha": args.alpha, "budget_pp": BUDGET_PP,
                    "n_full": N_FULL, "seed": args.seed, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    def pct(x):
        return "—" if x is None else f"{x * 100:.1f}%"

    lines = [
        f"# Sequential acceptance: simulation study ({args.reps} runs per scenario)",
        "",
        f"Budget {BUDGET_PP:.1f}pp, alpha = {args.alpha}, full set {N_FULL} images.",
        "",
        "| scenario | true Δ | truth | fixed-256 error | sequential error | undecided | "
        "median n | p90 n | fixed-n for 90% power |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["sequential"]
        fixed_power = r["fixed_n_for_90pct_power"]
        lines.append(
            f"| {r['scenario']} | {r['true_delta_pp']:+.2f}pp | {r['truth']} | "
            f"{pct(r['fixed_256']['error_rate'])} | {pct(s['error_rate'])} | "
            f"{pct(s['undecided_rate'])} | {s['median_n']:.0f} | {s['p90_n']:.0f} | "
            f"{'—' if fixed_power is None else f'{fixed_power:,}'} |"
        )
    lines += [
        "",
        "On the boundary scenario there is no correct answer; its accept rate under each rule "
        "shows how each behaves when the evidence cannot decide:",
        "",
    ]
    for r in rows:
        if r["truth"] == "boundary":
            lines.append(
                f"- fixed-256 accepts {pct(r['fixed_256']['accept_rate'])}, fixed-3925 accepts "
                f"{pct(r['fixed_3925']['accept_rate'])}, sequential accepts "
                f"{pct(r['sequential']['accept_rate'])} and stays undecided "
                f"{pct(r['sequential']['undecided_rate'])}."
            )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
