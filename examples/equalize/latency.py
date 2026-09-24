"""cpu-1t latency of the equalisation recipes, FP32 timed before and after (A-B-A)."""
import json, sys
from pathlib import Path
from anneal.core.artifact import ModelArtifact, sample_shape
from anneal.core.dataset import load_calibset
from anneal.core.environment import drift, snapshot, warnings_for
from anneal.core.measure import Benchmarker
from anneal.core.targets import get_target
from anneal.core.transforms import TransformContext, apply_transform

name = sys.argv[1]
ROOT = Path(__file__).resolve().parents[2]
model = ROOT / "examples" / "models" / f"{name}-fp32.onnx"
calib = load_calibset("imagenette", cache_dir=Path.home() / ".anneal_cache", batch_size=32, limit=64, sample_shape=sample_shape(model))
ctx = TransformContext(workdir=ROOT / "scratch" / "eq" / f"latency-{name}", calibset=calib)
P = {"calibrate_method": "percentile_asym"}
variants = {
    "baseline (minmax)": {"per_channel": True},
    "reduce_range": {"per_channel": True, "reduce_range": True},
    "P + stem": {"per_channel": True, **P, "float_stem": True},
    "EQ + P + stem": {"per_channel": True, "equalize": True, **P, "float_stem": True},
    "EQ + P + stem + reduce_range": {"per_channel": True, "equalize": True, **P, "float_stem": True, "reduce_range": True},
    "EQ + P + stem + float_gates": {"per_channel": True, "equalize": True, **P, "float_stem": True, "float_gates": True},
}
arts = {k: apply_transform("quantize_static_int8", {"activation_type": "uint8", **v}, ModelArtifact(path=model), ctx) for k, v in variants.items()}
bench = Benchmarker(get_target("cpu-1t"), warmup=20, runs=100)
base = ModelArtifact(path=model)
t0 = bench.measure(base).latency_ms_p50
res = {"env_warnings": warnings_for(snapshot()), "fp32_ms": t0}
print(f"{name}: FP32 {t0:.2f} ms")
for k, a in arts.items():
    m = bench.measure(a)
    res[k] = {"p50_ms": m.latency_ms_p50, "p99_ms": m.latency_ms_p99, "speedup": t0 / m.latency_ms_p50}
    print(f"  {k:30s} {m.latency_ms_p50:7.2f} ms  {t0 / m.latency_ms_p50:4.2f}x")
t1 = bench.measure(base).latency_ms_p50
res["fp32_end_ms"] = t1; res["drift"] = drift(t0, t1)
print(f"FP32 re-timed {t1:.2f} ms (drift {res['drift'] * 100:.1f}%)")
(ctx.workdir / "latency.json").write_text(json.dumps(res, indent=1))
