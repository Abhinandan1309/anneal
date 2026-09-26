"""Does Anneal's equalisation hold under Intel OpenVINO (NNCF post-training INT8)?

NNCF's default preset (PERFORMANCE) quantizes activations symmetrically per tensor and weights per
channel; the MIXED preset makes activations asymmetric. Both are scored on the exported model and
on Anneal's equalised one, on ImageNet validation images, paired against FP32 run by OpenVINO.
Runs locally on the x86 CPU (OpenVINO's CPU plugin executes INT8 with VNNI/AVX2 kernels, so these
are the numbers a user gets, not an emulation).

    python examples/openvino/run_openvino.py --models efficientnet_b0 --limit 5000 --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".anneal_cache"
# "ovf": NNCF's overflow fix on every layer, not only the first (its default). CPUs without VNNI
# (AVX2 only, e.g. Zen 2) compute U8S8 products with 16-bit intermediate sums that can saturate.
VARIANTS = ["nncf int8", "nncf int8 + eq", "nncf int8 ovf", "nncf int8 ovf + eq", "nncf mixed ovf",
            "nncf mixed ovf + eq"]


def main() -> None:
    import nncf
    import openvino as ov
    from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters, OverflowFix

    from anneal.core.artifact import sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import equalise

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="efficientnet_b0")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--dataset", default="imagenet")
    ap.add_argument("--calib", type=int, default=300, help="calibration images (NNCF's default subset size)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    core = ov.Core()
    report = {"runtime": f"OpenVINO {ov.__version__}, NNCF {nncf.__version__}", "device": "CPU",
              "cpu": core.get_property("CPU", "FULL_DEVICE_NAME"), "n": args.limit, "models": {}}
    work = ROOT / "scratch" / "openvino"
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        src = ROOT / "examples" / "models" / f"{name}-fp32.onnx"
        shape = sample_shape(src)
        calib = load_calibset(args.dataset, cache_dir=CACHE, batch_size=1, limit=args.calib, sample_shape=shape)
        calib_imgs = list(calib.calibration_batches(args.calib))
        eq = work / name / f"{name}-equalised.onnx"
        sites = len(equalise(src, eq, calib_imgs[:64]).sites)
        print(f"{name}: equalised {sites} sites", flush=True)
        fp32 = {"plain": ov.convert_model(str(src)), "eq": ov.convert_model(str(eq))}
        ev = load_evalset(args.dataset, cache_dir=CACHE, batch_size=32, limit=args.limit, sample_shape=shape)

        models = {"fp32": fp32["plain"]}
        for label in [v.strip() for v in args.variants.split(",") if v.strip()]:
            base = fp32["eq" if label.endswith("+ eq") else "plain"]
            preset = nncf.QuantizationPreset.MIXED if "mixed" in label else nncf.QuantizationPreset.PERFORMANCE
            advanced = AdvancedQuantizationParameters(overflow_fix=OverflowFix.ENABLE if "ovf" in label
                                                      else OverflowFix.FIRST_LAYER)
            t = time.time()
            models[label] = nncf.quantize(base, nncf.Dataset(calib_imgs), preset=preset, subset_size=len(calib_imgs),
                                          advanced_parameters=advanced)
            print(f"  {label}: quantized in {time.time() - t:.0f}s", flush=True)
        compiled = {k: core.compile_model(m, "CPU") for k, m in models.items()}

        preds, ys = {k: [] for k in compiled}, []
        for x, y in ev.batches():
            for k, c in compiled.items():
                preds[k].append(np.asarray(ev.decode(c(x)[0])))
            ys.append(y)
        y = np.concatenate(ys)
        ref = np.concatenate(preds["fp32"]) == y
        out = {"fp32": {"accuracy": float(ref.mean())}, "equalised_sites": sites,
               "fp32_correct": "".join("1" if r else "0" for r in ref)}
        for k in compiled:
            if k == "fp32":
                continue
            right = np.concatenate(preds[k]) == y
            b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right))
            d, lo, hi = paired_delta_ci(b, c, len(y))
            out[k] = {"accuracy": float(right.mean()), "delta_pp": d, "ci95_pp": [lo, hi],
                      "mcnemar_p": mcnemar_exact(b, c), "correct": "".join("1" if r else "0" for r in right)}
            print(f"  {k:18s} {d:+7.2f}pp vs FP32 [{lo:+.2f},{hi:+.2f}]", flush=True)
        report["models"][name] = out
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        del compiled, models, fp32


if __name__ == "__main__":
    main()
