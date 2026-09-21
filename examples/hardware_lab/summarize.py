"""Combine hardware-lab results from several machines into one table.

    python examples/hardware_lab/summarize.py results/*.json > results.md
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


def fmt_cell(row: dict) -> str:
    sig = "*" if row["mcnemar_p"] < 0.05 else ""
    return f"{row['delta_pp']:+.1f}pp{sig} · {row['speedup']:.2f}x"


def main(paths: list[str]) -> None:
    sys.stdout.reconfigure(errors="replace")  # Windows consoles cannot print the markers
    results =[json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(paths)]
    if not results:
        print("no results")
        return
    variants = [r["variant"] for r in results[0]["study"]["rows"]]

    print("# Hardware lab: static INT8 recipes across CPUs\n")
    print("ResNet-18, calibrated on 64 Imagenette train images, scored against FP32 on the same "
          "validation images. Each cell: accuracy change (\\* = McNemar p < 0.05) · speedup.\n")
    head = ["machine", "CPU", "VNNI", "ARM dotprod", "latency drift"] + [SHORT.get(v, v) for v in variants]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in results:
        cpu, study = r["cpu"], r["study"]
        drift = study.get("latency_drift")
        trust = "" if drift is None or drift <= 0.10 else " ⚠"
        cells = [
            f"{cpu.get('runner_os') or cpu.get('system')} {cpu.get('runner_arch') or cpu.get('machine')}",
            (cpu.get("brand") or cpu.get("processor") or "?").replace("|", "/"),
            "yes" if cpu.get("has_vnni") else "no",
            "yes" if cpu.get("has_arm_dotprod") else "no",
            "—" if drift is None else f"{drift * 100:.1f}%{trust}",
        ] + [fmt_cell(row) for row in study["rows"]]
        print("| " + " | ".join(cells) + " |")
    print("\nLatency drift is the change in FP32 latency between the start and end of the job; "
          "above 10% (⚠) that machine's speedups are not reliable. Accuracy is unaffected by "
          "timing noise.")


if __name__ == "__main__":
    main(sys.argv[1:])
