"""Why the advised ReLU-CNN recipe lost to plain percentile on ResNet-50 (ImageNet).

On ImageNet (10,000 scored images) the advice for the ``cnn`` family, asymmetric percentile
(99.99) + float stem, scored -0.54pp against FP32, while symmetric percentile at 99.999 scored
-0.10pp: paired, the advice was 0.44pp worse (p=0.003). The two differ in three ways at once:
the percentile (99.99 vs 99.999), asymmetric vs symmetric ranges, and the float stem. This
ablation separates them, on the same images as the ImageNet run, both scored as 32-bit
hardware would (graph optimisations off) and fused on this laptop (x86 without VNNI, where the
stem saturates).

Results are paired against the stored ImageNet predictions (the FP32 predictions must match).

    python resnet50_ablation.py --part emulated   # then --part fused, then --part report

Run in parts: all nine ResNet-50 sessions at once needed over 9 GB on a 16 GB laptop.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples" / "imagenet"))

from run_imagenet import model_path, session, stats  # noqa: E402

from anneal.core.artifact import ModelArtifact, sample_shape  # noqa: E402
from anneal.core.audit import mcnemar_exact, paired_delta_ci  # noqa: E402
from anneal.core.dataset import load_calibset, load_evalset  # noqa: E402
from anneal.core.transforms import TransformContext, apply_transform  # noqa: E402

BASE = {"per_channel": True, "activation_type": "uint8", "calib_samples": 64}
SYM = {"calibrate_method": "percentile", "calib_percentile": 99.999}
ASYM_9999 = {"calibrate_method": "percentile_asym", "calib_percentile": 99.99}
ASYM_99999 = {"calibrate_method": "percentile_asym", "calib_percentile": 99.999}
STEM = {"float_stem": True}
#: label -> (params, fused)
RECIPES = {
    "asym 99.99": ({**BASE, **ASYM_9999}, False),
    "asym 99.999": ({**BASE, **ASYM_99999}, False),
    "asym 99.999 + stem": ({**BASE, **ASYM_99999, **STEM}, False),
    "sym 99.999 + stem": ({**BASE, **SYM, **STEM}, False),
    "minmax + stem": ({**BASE, "calibrate_method": "minmax", **STEM}, False),
    "sym 99.999 + stem (fused)": ({**BASE, **SYM, **STEM}, True),
    "sym 99.999 + reduce_range (fused)": ({**BASE, **SYM, "reduce_range": True}, True),
    "minmax + stem (fused)": ({**BASE, "calibrate_method": "minmax", **STEM}, True),
    "asym 99.999 + stem (fused)": ({**BASE, **ASYM_99999, **STEM}, True),
}
STORED = {
    "minmax": "minmax",
    "sym 99.999": "percentile",
    "asym 99.99 + stem (advice)": "anneal (32-bit): percentile + float stem",
    "asym 99.99 + stem + reduce_range (fused, advice)": "anneal (x86-avx2-16bit, fused): percentile + float stem + reduce_range",
    "minmax (fused)": "minmax (fused, this laptop)",
}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["emulated", "fused", "report"], required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    if args.part == "report":
        report()
    else:
        score([k for k, (_, fused) in RECIPES.items() if fused == (args.part == "fused")], args.part)


WORK = ROOT / "scratch" / "imagenet" / "resnet50-ablation"


def score(labels_to_run: list[str], part: str) -> None:
    name = "resnet50"
    path = model_path(name)
    shape = sample_shape(path)
    cache = Path.home() / ".anneal_cache"
    calib = load_calibset("imagenet", cache_dir=cache, batch_size=8, limit=64, sample_shape=shape)
    ev = load_evalset("imagenet", cache_dir=cache, batch_size=32, limit=10_000, sample_shape=shape)
    ctx = TransformContext(workdir=WORK, calibset=calib)

    import gc

    # Build every model before opening any session: histogram calibration holds all
    # activations of the calibration images, and open sessions on top of that exhausted RAM.
    built = {}
    for label in labels_to_run:
        built[label] = apply_transform("quantize_static_int8", dict(RECIPES[label][0]), ModelArtifact(path=path), ctx).path
        gc.collect()
        print(f"  built {label}", flush=True)
    sessions = {"fp32": session(path, True, 4)}
    for label in labels_to_run:
        sessions[label] = session(built[label], RECIPES[label][1], 4)
    inp = sessions["fp32"].get_inputs()[0].name
    preds = {k: [] for k in sessions}
    labels, t0, n = [], time.time(), 0
    for x, y in ev.batches():
        for k, s in sessions.items():
            preds[k].append(np.asarray(ev.decode(s.run(None, {inp: x})[0])))
        labels.append(y)
        n += len(y)
        if n % 1024 < len(y):
            print(f"  {n}/{len(ev)} images ({time.time() - t0:.0f}s)", flush=True)
    y = np.concatenate(labels)
    p = {k: np.concatenate(v) for k, v in preds.items()}
    np.savez_compressed(WORK / f"preds-{part}.npz", labels=y, **p)
    print(f"saved {part} predictions", flush=True)


def report() -> None:
    name = "resnet50"
    parts = [dict(np.load(WORK / f"preds-{part}.npz")) for part in ("emulated", "fused")]
    y = parts[0]["labels"]
    p = {}
    for part in parts:
        if not np.array_equal(part.pop("labels"), y) or ("fp32" in p and not np.array_equal(p["fp32"], part["fp32"])):
            raise RuntimeError("the two parts did not score the same images identically")
        p.update(part)

    stored = np.load(ROOT / "examples" / "imagenet" / "results" / f"{name}-predictions.npz")
    if not (np.array_equal(stored["labels"], y) and np.array_equal(stored["fp32"], p["fp32"])):
        raise RuntimeError("the ablation does not reproduce the stored ImageNet FP32 predictions")
    for label, key in STORED.items():
        p[label] = stored[key]

    fp_right = p["fp32"] == y
    best = p["sym 99.999"] == y
    rows = {}
    for k in [*STORED, *RECIPES]:
        right = p[k] == y
        row = stats(fp_right, right)
        b, c = int(np.sum(best & ~right)), int(np.sum(~best & right))
        d, lo, hi = paired_delta_ci(b, c, len(y))
        row["vs_sym_99999"] = {"delta_pp": d, "ci95_pp": [lo, hi], "mcnemar_p": mcnemar_exact(b, c)}
        row["source"] = "stored ImageNet run" if k in STORED else "this ablation"
        if k in RECIPES:
            row["params"], row["mode"] = RECIPES[k][0], "fused" if RECIPES[k][1] else "emulated"
        rows[k] = row
        print(f"  {k:48s} {row['delta_pp']:+6.2f}pp vs FP32 | {d:+6.2f}pp vs sym 99.999 "
              f"[{lo:+.2f},{hi:+.2f}] p={row['vs_sym_99999']['mcnemar_p']:.2g}", flush=True)
    out = HERE / "resnet50_ablation.json"
    out.write_text(json.dumps({"model": name, "n": int(len(y)), "fp32_accuracy": float(fp_right.mean()),
                               "rows": rows}, indent=1), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
