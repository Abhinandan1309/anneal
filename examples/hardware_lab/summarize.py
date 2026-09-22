"""Combine hardware-lab results from several machines into one table per model.

    python examples/hardware_lab/summarize.py results/*.json > results.md

Each cell is "accuracy change · speedup". For every recipe the saturation analyser's
prediction is shown once per model: it emulates non-VNNI x86 arithmetic, so it is the same
on every machine, and it should come true only on the machines marked "x86-avx2-16bit".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SHORT = {
    "U8S8 per-channel (Anneal's old default)": "U8S8 per-channel",
    "U8S8 per-channel + reduce_range": "U8S8 per-ch + reduce_range",
    "S8S8 per-channel": "S8S8 per-channel",
    "U8S8 per-tensor": "U8S8 per-tensor",
    "S8S8 per-tensor (Olive's default)": "S8S8 per-tensor",
}


def cell(row: dict) -> str:
    sig = "*" if row["mcnemar_p"] < 0.05 else ""
    return f"{row['delta_pp']:+.1f}pp{sig} · {row['speedup']:.2f}x"


def studies_of(result: dict) -> dict:
    """Multi-model results carry "studies"; the first lab run carried one "study"."""
    if "studies" in result:
        return result["studies"]
    return {result["study"].get("model", "resnet18").replace("-fp32.onnx", ""): result["study"]}


def int8_path(cpu: dict) -> str:
    if cpu.get("int8_path"):
        return cpu["int8_path"]
    if cpu.get("has_arm_dotprod"):
        return "arm-dotprod"
    if cpu.get("has_vnni"):
        return "x86-vnni"
    return "x86-avx2-16bit" if "avx2" in cpu.get("int8_features", []) else "unknown"


def main(paths: list[str]) -> None:
    sys.stdout.reconfigure(errors="replace")  # Windows consoles cannot print the markers
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(paths)]
    if not results:
        print("no results")
        return
    models = sorted({m for r in results for m in studies_of(r)})

    print("# Hardware lab: static INT8 recipes across CPUs\n")
    print("Calibrated on 64 Imagenette train images, scored against FP32 on the same validation "
          "images. Each cell: accuracy change (\\* = McNemar p < 0.05) · speedup.\n")
    for model in models:
        rows_by_machine = [(r["cpu"], studies_of(r)[model]) for r in results if model in studies_of(r)]
        if not rows_by_machine:
            continue
        variants = [row["variant"] for row in rows_by_machine[0][1]["rows"]]
        print(f"## {model}\n")
        head = ["machine", "CPU", "INT8 path", "drift"] + [SHORT.get(v, v) for v in variants]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for cpu, study in rows_by_machine:
            drift = study.get("latency_drift")
            warn = " ⚠" if (drift is not None and drift > 0.10) or study.get("environment_warnings") else ""
            cells = [
                f"{cpu.get('runner_os') or cpu.get('system')} {cpu.get('runner_arch') or cpu.get('machine')}",
                (cpu.get("brand") or cpu.get("processor") or "?").replace("|", "/"),
                int8_path(cpu),
                "—" if drift is None else f"{drift * 100:.1f}%{warn}",
            ] + [cell(row) for row in study["rows"]]
            print("| " + " | ".join(cells) + " |")
        predicted = next((s for _, s in rows_by_machine if "predicted_saturation" in s["rows"][0]), None)
        if predicted:
            preds = []
            for row in predicted["rows"]:
                p = row["predicted_saturation"]
                preds.append(
                    "none possible" if not p["saturation_possible"]
                    else f"{p['layers_saturating']} layer(s), worst {p['worst_layer_accumulator_rate'] * 100:.1f}%"
                    if p["layers_saturating"] else "none observed"
                )
            print("| **predicted saturation (x86-avx2-16bit only)** | | | | " + " | ".join(preds) + " |")
        print()
    print("Drift is the change in FP32 latency between the start and end of each job; ⚠ marks "
          "over 10% drift or a machine-state warning, and that machine's speedups are not "
          "reliable. Accuracy is unaffected by timing noise.")


if __name__ == "__main__":
    main(sys.argv[1:])
