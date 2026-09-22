"""Score the saturation analyser against the hardware lab.

For each model and recipe, the analyser predicts saturation on x86 CPUs without VNNI. The
lab measures accuracy on those CPUs *and* on CPUs whose INT8 arithmetic is 32-bit (VNNI x86,
ARM dot-product). The difference between the two — the accuracy lost only where 16-bit
saturation is possible — is what the analyser is actually claiming to predict. Loss that
happens everywhere is ordinary quantization damage and is not its business.

    python examples/hardware_lab/scorecard.py results/*.json > scorecard.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

from summarize import SHORT, int8_path, studies_of

#: An x86-only excess loss beyond this (percentage points) counts as "saturation observed".
OBSERVED_PP = 1.5


def main(paths: list[str]) -> None:
    sys.stdout.reconfigure(errors="replace")
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(paths)]
    by_model: dict[str, dict[str, list[dict]]] = {}
    for r in results:
        path = int8_path(r["cpu"])
        group = "saturating" if path == "x86-avx2-16bit" else "32-bit" if path in ("x86-vnni", "arm-dotprod") else None
        if group is None:
            continue
        for model, study in studies_of(r).items():
            by_model.setdefault(model, {"saturating": [], "32-bit": []})[group].append(study)

    print("# Saturation analyser vs. the hardware lab\n")
    print("*x86-only loss* = mean accuracy change on x86 CPUs without VNNI minus the mean on "
          "CPUs with 32-bit INT8 accumulation (VNNI x86, ARM). Negative means the recipe loses "
          f"accuracy only where saturation is possible. Beyond {OBSERVED_PP}pp counts as observed.\n")
    print("| model | recipe | on 32-bit CPUs | on saturating x86 | x86-only loss | predicted | verdict |")
    print("|---|---|---:|---:|---:|---|---|")
    tally = {"hit": 0, "miss": 0, "false alarm": 0, "correct all-clear": 0, "not modelled": 0}
    for model in sorted(by_model):
        groups = by_model[model]
        if not groups["saturating"] or not groups["32-bit"]:
            continue
        variants = [row["variant"] for row in groups["saturating"][0]["rows"]]
        for i, variant in enumerate(variants):
            sat = mean(s["rows"][i]["delta_pp"] for s in groups["saturating"])
            ref = mean(s["rows"][i]["delta_pp"] for s in groups["32-bit"])
            excess = sat - ref
            row = groups["saturating"][0]["rows"][i]
            p = row.get("predicted_saturation")
            if p is None:
                continue
            predicted = bool(p["layers_saturating"]) and p["worst_layer_accumulator_rate"] > 0.02
            pred_txt = (
                "impossible" if not p["saturation_possible"]
                else f"{p['layers_saturating']} layer(s), worst {p['worst_layer_accumulator_rate'] * 100:.0f}%"
                if p["layers_saturating"] else "none"
            )
            observed = excess < -OBSERVED_PP
            signed = row["params"].get("activation_type") == "int8"
            if signed:
                verdict = "not modelled (S8S8)"
                tally["not modelled"] += 1
            elif predicted and observed:
                verdict = "hit"
                tally["hit"] += 1
            elif predicted and not observed:
                verdict = "false alarm"
                tally["false alarm"] += 1
            elif observed:
                verdict = "miss"
                tally["miss"] += 1
            else:
                verdict = "correct all-clear"
                tally["correct all-clear"] += 1
            print(f"| {model} | {SHORT.get(variant, variant)} | {ref:+.1f}pp | {sat:+.1f}pp | "
                  f"{excess:+.1f}pp | {pred_txt} | {verdict} |")
    print("\n**Tally (U8S8 recipes):** " + ", ".join(f"{k}: {v}" for k, v in tally.items()))
    print("\nS8S8 recipes are listed but not scored: the lab shows the S8S8 kernel on non-VNNI "
          "x86 does not consistently use the saturating path the analyser assumes.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main(sys.argv[1:])
