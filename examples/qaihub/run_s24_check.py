"""Two checks of the Galaxy S24 column of the benchmark grid, on the device.

1. Toolchain version: are the numbers a property of one QAIRT release? The grid was compiled with
   AI Hub's default, QAIRT 2.50. Each model is quantized once, then compiled and run with every
   QAIRT version AI Hub offers, on the grid's own 1,024 Imagenette images.
2. Holdout: the recipes were chosen on Imagenette. The same frozen recipes are scored on
   Imagewoof (ten ImageNet dog breeds; no study ever scored these images), with QAIRT 2.50.

Every delta is paired against true FP32 (onnxruntime, CPU) on the same images, with a 95% CI and
an exact McNemar p-value. Calibration is unchanged: 64 Imagenette train images.

    python run_s24_check.py --models efficientnet_b0,efficientnet_b1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = Path.home() / ".anneal_cache"
sys.path.insert(0, str(HERE))
from run_qaihub import static_copy, target_of  # noqa: E402

#: The grid's Anneal recipe per model on the S24 (docs/benchmark_grid.md). "hub" = equalised
#: (grid 1/s) model quantized by Qualcomm's quantizer; a dict = Anneal's own QDQ with these params.
QDQ = {"per_channel": True, "activation_type": "uint8", "calib_samples": 64, "calib_stride": 1,
       "equalize": True, "equalize_residual": True, "equalize_se": True, "calibrate_method": "percentile_asym"}
RECIPE = {
    "efficientnet_b0": "hub",
    "efficientnet_b1": {**QDQ, "equalize_grid_inverse": True},
    "mobilenet_v3_small": dict(QDQ),
    "mobilenet_v3_large": "hub",
    "mobilenet_v2": "hub",
    "lcnet_100": "hub",
    "mobilevit_s": "hub",
    "resnet50": "hub",
}
VERSIONS = ["2.45", "2.49", "2.50"]
GRID_VERSION = "2.50"


def images(spec: str, n: int, shape):
    from anneal.core.dataset import load_evalset

    ev = load_evalset(spec, cache_dir=CACHE, batch_size=16, limit=n, sample_shape=shape)
    xs, ys = [], []
    for xb, yb in ev.batches():
        xs += [xb[i:i + 1] for i in range(len(xb))]
        ys += list(yb)
    return xs, np.array(ys)


def main() -> None:
    import onnxruntime as ort
    import qai_hub as hub

    from anneal.core.artifact import ModelArtifact, sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset
    from anneal.core.equalize import equalise
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(RECIPE))
    ap.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    ap.add_argument("--versions", default=",".join(VERSIONS))
    ap.add_argument("--grid-images", type=int, default=1024)
    ap.add_argument("--holdout-images", type=int, default=1000)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    versions = args.versions.split(",")
    device = hub.Device(args.device)
    uploaded = {}
    fw = {f.api_version: f.full_version for f in hub.get_frameworks() if f.name == "QAIRT"}

    for model in args.models.split(","):
        out = HERE / "results" / f"{model}-s24-check.json"
        if out.exists():
            print(f"skip {model}: {out.name} exists", flush=True)
            continue
        src = ROOT / "examples" / "models" / f"{model}-fp32.onnx"
        work = ROOT / "scratch" / "qaihub_check" / model
        work.mkdir(parents=True, exist_ok=True)
        shape = sample_shape(src)
        calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64, sample_shape=shape)
        calib_imgs = [x[i:i + 1] for x in calib.calibration_batches(64) for i in range(len(x))][:64]

        # quantize once; every QAIRT version compiles the same quantized model
        recipe = RECIPE[model]
        q_jobs = {"vendor int8": hub.submit_quantize_job(str(static_copy(src, work)), {"input": calib_imgs},
                                                         name=f"anneal-check-{model}-int8-q")}
        quantized = {}
        if recipe == "hub":
            eqg = work / "eq-grid.onnx"
            equalise(src, eqg, [np.concatenate(calib_imgs[i:i + 8]) for i in range(0, 64, 8)], grid_inverse=True)
            q_jobs["anneal"] = hub.submit_quantize_job(str(static_copy(eqg, work)), {"input": calib_imgs},
                                                       name=f"anneal-check-{model}-anneal-q")
        else:
            art = apply_transform("quantize_static_int8", dict(recipe), ModelArtifact(path=src),
                                  TransformContext(workdir=work / "qdq", calibset=calib))
            quantized["anneal"] = str(static_copy(Path(art.path), work / "qdq"))
        for k, j in q_jobs.items():
            if (m := target_of(j, k)) is not None:
                quantized[k] = m

        # one set in memory at a time (~600 MB each); uploads are reused by models of the same shape
        cpu = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        sets, fp32, datasets = {}, {}, {}
        for s, spec, n in (("grid", "imagenette", args.grid_images), ("holdout", "imagewoof", args.holdout_images)):
            xs, sets[s] = images(spec, n, shape)
            fp32[s] = np.array([int(np.argmax(cpu.run(None, {"input": x})[0])) for x in xs])
            if (s, shape) not in uploaded:
                uploaded[(s, shape)] = hub.upload_dataset({"input": xs}, name=f"anneal-{spec}-{len(xs)}")
            datasets[s] = uploaded[(s, shape)]
            del xs

        # (variant, version) -> compile; grid images on every version, holdout on the grid's version
        c_jobs = {(k, v): hub.submit_compile_job(m, device=device, input_specs={"input": (1, *shape)},
                                                 options=f"--target_runtime qnn_dlc --qairt_version {v}",
                                                 name=f"anneal-check-{model}-{k}-{v}")
                  for k, m in quantized.items() for v in versions}
        runs = {}
        for (k, v), j in c_jobs.items():
            t = target_of(j, f"{k} {v}")
            if t is None:
                continue
            for s in ("grid", "holdout") if v == GRID_VERSION else ("grid",):
                runs[(k, v, s)] = hub.submit_inference_job(t, device=device, inputs=datasets[s],
                                                           options=f"--qairt_version {v}",
                                                           name=f"anneal-check-{model}-{k}-{v}-{s}")
        result = {"model": model, "device": args.device, "recipe": recipe, "qairt": {v: fw.get(v) for v in versions},
                  "calibration": "64 Imagenette train images",
                  "sets": {"grid": f"Imagenette val, first {len(sets['grid'])} (the grid's images)",
                           "holdout": f"Imagewoof val, first {len(sets['holdout'])} (never scored before)"},
                  "true_fp32_accuracy": {s: float((fp32[s] == ys).mean()) for s, ys in sets.items()}, "rows": []}
        for (k, v, s), j in runs.items():
            row = {"variant": k, "qairt": v, "set": s, "job": j.job_id}
            try:
                o = j.download_output_data()
                p = np.concatenate([np.asarray(a).reshape(1, -1) for a in next(iter(o.values()))]).argmax(1)
                ys, ref = sets[s], fp32[s] == sets[s]
                right = p == ys
                b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right))
                d, lo, hi = paired_delta_ci(b, c, len(ys))
                row.update(accuracy=float(right.mean()), delta_pp=d, ci95_pp=[lo, hi], mcnemar_p=mcnemar_exact(b, c))
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)[:300]
            result["rows"].append(row)
            print(f"  {model:20s} {k:12s} QAIRT {v} {s:8s} {row.get('delta_pp', float('nan')):+6.1f}pp "
                  f"{row.get('error', '')[:80]}", flush=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
