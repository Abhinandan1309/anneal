"""Re-time the advised recipes on this CPU (cpu-1t), interleaved over rounds.

Each round times FP32 and every candidate once, so slow drift in the machine's performance
state shows up as round-to-round spread instead of biasing one candidate.

    python examples/advise/retime.py examples/advise/efficientnet_b0-x86-avx2-16bit.json
"""
import json
import sys
from pathlib import Path
from statistics import median

from anneal.core.artifact import ModelArtifact
from anneal.core.environment import snapshot, warnings_for
from anneal.core.measure import Benchmarker
from anneal.core.targets import get_target

ROUNDS = 3
report = Path(sys.argv[1])
data = json.loads(report.read_text(encoding="utf-8"))
model = Path("examples/models") / data["model"]
rows = [r for r in data["verification"]["rows"] if r.get("path")]
bench = Benchmarker(get_target("cpu-1t"), warmup=20, runs=100)
env = warnings_for(snapshot())
fp32, times = [], {r["label"]: [] for r in rows}
for _ in range(ROUNDS):
    fp32.append(bench.measure(ModelArtifact(path=model)).latency_ms_p50)
    for r in rows:
        times[r["label"]].append(bench.measure(ModelArtifact(path=Path(r["path"]))).latency_ms_p50)
env += [w for w in warnings_for(snapshot()) if w not in env]
base = median(fp32)
spread = (max(fp32) - min(fp32)) / base
print(f"{data['model']}: FP32 p50 {base:.2f} ms (round-to-round spread {spread * 100:.1f}%) {env or ''}")
out = {"fp32_ms": fp32, "environment_warnings": env, "rows": {}}
for r in rows:
    t = median(times[r["label"]])
    out["rows"][r["label"]] = {"p50_ms": times[r["label"]], "speedup": base / t, "delta_pp": r["delta_pp"]}
    print(f"  {r['label']:52s} {t:7.2f} ms  {base / t:4.2f}x   accuracy {r['delta_pp']:+.2f}pp")
report.with_name(report.stem + "-timing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
