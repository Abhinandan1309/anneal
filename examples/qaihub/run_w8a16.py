"""The vendor's own alternative: 16-bit activations (W8A16) instead of equalisation, on the S24.

A reviewer's first question: why not quantize activations to 16 bits, which gives a starved
channel 256x more levels? This runs Qualcomm's quantizer in W8A16 on the model as exported, and on
the equalised model, and scores accuracy and latency on the device against W8A8 and FP32, paired.

    python run_w8a16.py --runtime qnn_dlc
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = Path.home() / ".anneal_cache"
sys.path.insert(0, str(HERE))
from run_qaihub import static_copy, target_of  # noqa: E402


def main() -> None:
    import qai_hub as hub

    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import equalise

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    ap.add_argument("--runtime", default="qnn_dlc", choices=["tflite", "qnn_dlc", "onnx"])
    ap.add_argument("--images", type=int, default=1024)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    work = ROOT / "scratch" / "qaihub" / f"{args.model}-w8a16"
    work.mkdir(parents=True, exist_ok=True)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64)
    calib_imgs = [x[i:i + 1] for x in calib.calibration_batches(64) for i in range(len(x))][:64]
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=16, limit=args.images)
    eval_imgs, labels = [], []
    for x, y in ev.batches():
        eval_imgs += [x[i:i + 1] for i in range(len(x))]
        labels += list(y)
    labels = np.array(labels)
    eq = work / f"{args.model}-equalised.onnx"
    equalise(src, eq, [np.concatenate(calib_imgs[i:i + 8]) for i in range(0, 64, 8)])
    src_b1, eq_b1 = static_copy(src, work), static_copy(eq, work)

    I8, I16 = hub.QuantizeDtype.INT8, hub.QuantizeDtype.INT16
    cal = {"input": calib_imgs}
    q_jobs = {
        "W8A8": hub.submit_quantize_job(str(src_b1), cal, I8, I8, name=f"anneal-{args.model}-w8a8-q"),
        "W8A16": hub.submit_quantize_job(str(src_b1), cal, I8, I16, name=f"anneal-{args.model}-w8a16-q"),
        "W8A8 + equalised": hub.submit_quantize_job(str(eq_b1), cal, I8, I8, name=f"anneal-{args.model}-w8a8-eq-q"),
        "W8A16 + equalised": hub.submit_quantize_job(str(eq_b1), cal, I8, I16, name=f"anneal-{args.model}-w8a16-eq-q"),
    }
    models = {"fp32": str(src_b1)}
    for k, j in q_jobs.items():
        m = target_of(j, k)
        if m is not None:
            models[k] = m
    device = hub.Device(args.device)
    c_jobs = {k: hub.submit_compile_job(m, device=device, input_specs={"input": (1, 3, 224, 224)},
                                        options=f"--target_runtime {args.runtime}", name=f"anneal-{args.model}-{k}")
              for k, m in models.items()}
    targets = {k: t for k, j in c_jobs.items() if (t := target_of(j, k)) is not None}
    dataset = hub.upload_dataset({"input": eval_imgs}, name=f"anneal-imagenette-{args.images}")
    i_jobs = {k: hub.submit_inference_job(t, device=device, inputs=dataset, name=f"anneal-{args.model}-{k}-inf")
              for k, t in targets.items()}
    p_jobs = {k: hub.submit_profile_job(t, device=device, name=f"anneal-{args.model}-{k}-prof") for k, t in targets.items()}

    rows, preds = {}, {}
    for k in targets:
        row = {"inference_job": i_jobs[k].job_id, "profile_job": p_jobs[k].job_id}
        try:
            out = i_jobs[k].download_output_data()
            preds[k] = np.concatenate([np.asarray(a).reshape(1, -1) for a in next(iter(out.values()))]).argmax(1)
            row["accuracy"] = float((preds[k] == labels).mean())
        except Exception as exc:
            row["inference_error"] = f"{i_jobs[k].get_status().message} {exc}"
        try:
            prof = p_jobs[k].download_profile()
            row["latency_ms"] = prof["execution_summary"]["estimated_inference_time"] / 1000.0
            row["compute_units"] = dict(Counter(u.get("compute_unit") for u in prof.get("execution_detail", [])))
        except Exception as exc:
            row["profile_error"] = f"{p_jobs[k].get_status().message} {exc}"
        rows[k] = row
    if "fp32" in preds:
        ref = preds["fp32"] == labels
        for k, p in preds.items():
            if k == "fp32":
                continue
            right = p == labels
            b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right))
            d, lo, hi = paired_delta_ci(b, c, len(labels))
            rows[k].update(delta_pp=d, ci95_pp=[lo, hi], mcnemar_p=mcnemar_exact(b, c))
    slug = args.device.lower().replace(" ", "-").replace("(", "").replace(")", "")
    path = HERE / "results" / f"{args.model}-w8a16-{slug}-{args.runtime}-n{len(labels)}.json"
    path.write_text(json.dumps({"model": args.model, "device": args.device, "runtime": args.runtime,
                                "n": int(len(labels)), "variants": rows}, indent=2), encoding="utf-8")
    print(f"\n{args.model}: W8A16 vs equalisation on {args.device} ({args.runtime}), {len(labels)} images")
    for k, r in rows.items():
        acc = f"{100 * r['accuracy']:5.1f}%" if "accuracy" in r else "  -  "
        d = f"{r['delta_pp']:+6.1f}pp p={r['mcnemar_p']:.2g}" if "delta_pp" in r else ""
        lat = f"{r['latency_ms']:.3f} ms" if "latency_ms" in r else r.get("profile_error", "")[:60]
        print(f"  {k:20s} {acc} {d:20s} {lat}", flush=True)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
