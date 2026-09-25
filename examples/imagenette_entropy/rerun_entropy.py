"""Re-score the Imagenette entropy results after the entropy-calibration fix.

onnxruntime's ``quantize_static`` cannot pass histogram sizes to its entropy calibrator, which
then runs with 128 bins folded to 128: the KL search has nothing to search and returns the
min/max range. Every "entropy" result produced before the fix was therefore min/max under
another name (identical predictions on ImageNet, identical accuracy in the agent ledgers).
``anneal.core.transforms`` now injects TensorRT's 2048 bins folded to 128.

Each case is rebuilt with its original parameters, calibration and sample size, and scored
the way the original was (onnxruntime with graph optimisations, this laptop). ``minmax`` is
re-scored alongside as a reproduction check: it must match the old number.

    python rerun_entropy.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from anneal.core.artifact import ModelArtifact
from anneal.core.audit import mcnemar_exact, paired_delta_ci
from anneal.core.dataset import load_calibset, load_evalset
from anneal.core.transforms import TransformContext, apply_transform

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".anneal_cache"

#: (label, model, n scored, base params, where the old result lives, old minmax acc, old entropy acc)
CASES = [
    ("resnet18 agent run", ROOT / "examples/resnet18-cpu1t/models/resnet18-fp32.onnx", 256,
     {"per_channel": True, "reduce_range": False, "calib_samples": 64},
     "examples/resnet18-cpu1t/ledger.json trials 3 and 9 (README table rows 3 and 9)", 0.640625, 0.640625),
    ("mobilenet_v3_large agent run", ROOT / "examples/models/mobilenet_v3_large-fp32.onnx", 256,
     {"per_channel": True, "reduce_range": False, "calib_samples": 64, "activation_type": "uint8"},
     "examples/mobilenetv3-cpu1t/ledger.json trials 3 and 10", 0.421875, 0.421875),
    ("efficientnet_b0 recipe sweep", ROOT / "examples/models/efficientnet_b0-fp32.onnx", 512,
     {"per_channel": True, "calib_samples": 64},
     "examples/equalize/recipe_sweep.json efficientnet_b0/baseline and efficientnet_b0/entropy", 0.224609375, 0.396484375),
]


def predict(path: Path, ev) -> np.ndarray:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return np.concatenate([s.run(None, {name: x})[0].argmax(1) for x, _ in ev.batches()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    report = {"fix": "entropy calibration now uses 2048 histogram bins folded to 128 (TensorRT's KL setting); "
                     "before, onnxruntime's 128/128 default made it identical to minmax", "cases": []}
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64)
    for label, model, n, base, where, old_minmax, old_entropy in CASES:
        ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=16, limit=n)
        y = np.concatenate([yy for _, yy in ev.batches()])
        work = ROOT / "scratch" / "imagenette_entropy" / model.stem
        preds = {"fp32": predict(model, ev)}
        for method in ("minmax", "entropy"):
            art = apply_transform("quantize_static_int8", {**base, "calibrate_method": method},
                                  ModelArtifact(path=model), TransformContext(workdir=work / method, calibset=calib))
            preds[method] = predict(Path(art.path), ev)
        ref = preds["fp32"] == y
        rows = {}
        for k, p in preds.items():
            right = p == y
            row = {"accuracy": float(right.mean())}
            if k != "fp32":
                b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right))
                d, lo, hi = paired_delta_ci(b, c, len(y))
                row.update(delta_pp=d, ci95_pp=[lo, hi], mcnemar_p=mcnemar_exact(b, c),
                           agreement=float(np.mean(p == preds["fp32"])))
            rows[k] = row
        case = {"case": label, "model": str(model.relative_to(ROOT)), "n": int(len(y)), "params": base,
                "old_result": where, "old_minmax_accuracy": old_minmax, "old_entropy_accuracy": old_entropy,
                "minmax_reproduces_old": abs(rows["minmax"]["accuracy"] - old_minmax) < 1e-9,
                "entropy_vs_minmax_agreement": float(np.mean(preds["entropy"] == preds["minmax"])),
                "rows": rows}
        report["cases"].append(case)
        print(f"{label} (n={len(y)}): fp32 {rows['fp32']['accuracy']*100:.1f}%  "
              f"minmax {rows['minmax']['accuracy']*100:.1f}% (old {old_minmax*100:.1f}%)  "
              f"entropy {rows['entropy']['accuracy']*100:.1f}% (old {old_entropy*100:.1f}%)", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
