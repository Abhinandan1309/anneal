"""Selective equalisation on a real NPU: accuracy and latency against the number of sites equalised.

Full equalisation of EfficientNet-B0 recovers ~90% of the INT8 loss on the Galaxy S24, but its 16
gate multiplies cost ~30% of INT8 speed. ``rank_sites`` predicts each site's benefit; this
equalises only the top k (k = 0 is plain `hub int8`, k = 16 is full equalisation), quantizes each
with Qualcomm's quantizer and scores accuracy and latency on the device, paired against its FP32.

    python run_topk.py --ks 1,2,4,8,16 --runtime qnn_dlc
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
    from anneal.core.equalize import equalise, rank_sites

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    ap.add_argument("--runtime", default="qnn_dlc", choices=["tflite", "qnn_dlc", "onnx"])
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--images", type=int, default=1024)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    work = ROOT / "scratch" / "qaihub" / f"{args.model}-topk"
    work.mkdir(parents=True, exist_ok=True)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64)
    calib_imgs = [x[i:i + 1] for x in calib.calibration_batches(64) for i in range(len(x))][:64]
    batches = [np.concatenate(calib_imgs[i:i + 8]) for i in range(0, 64, 8)]
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=16, limit=args.images)
    eval_imgs, labels = [], []
    for x, y in ev.batches():
        eval_imgs += [x[i:i + 1] for i in range(len(x))]
        labels += list(y)
    labels = np.array(labels)

    ranking = rank_sites(src, batches)
    print("ranking (site, predicted gain):", flush=True)
    for site, gain in ranking:
        print(f"  {site:50s} {gain:.3f}", flush=True)
    ks = [int(k) for k in args.ks.split(",")]
    models = {"fp32": static_copy(src, work)}
    for k in ks:
        dst = work / f"{args.model}-eq-top{k}.onnx"
        equalise(src, dst, batches, top_k=k)
        models[f"top{k}"] = static_copy(dst, work)

    device = hub.Device(args.device)
    calibration = {"input": calib_imgs}
    q_jobs = {label: hub.submit_quantize_job(str(p), calibration, name=f"anneal-{args.model}-{label}-q")
              for label, p in models.items() if label != "fp32"}
    q_jobs["top0"] = hub.submit_quantize_job(str(models["fp32"]), calibration, name=f"anneal-{args.model}-top0-q")
    to_compile = {"fp32": str(models["fp32"])}
    for label, job in q_jobs.items():
        m = target_of(job, label)
        if m is not None:
            to_compile[label] = m
    c_jobs = {label: hub.submit_compile_job(m, device=device, input_specs={"input": (1, 3, 224, 224)},
                                            options=f"--target_runtime {args.runtime}", name=f"anneal-{args.model}-{label}")
              for label, m in to_compile.items()}
    targets = {label: t for label, j in c_jobs.items() if (t := target_of(j, label)) is not None}
    dataset = hub.upload_dataset({"input": eval_imgs}, name=f"anneal-imagenette-{args.images}")
    i_jobs = {label: hub.submit_inference_job(t, device=device, inputs=dataset, name=f"anneal-{args.model}-{label}-inf")
              for label, t in targets.items()}
    p_jobs = {label: hub.submit_profile_job(t, device=device, name=f"anneal-{args.model}-{label}-prof")
              for label, t in targets.items()}

    preds, rows = {}, {}
    for label in targets:
        row = {"inference_job": i_jobs[label].job_id, "profile_job": p_jobs[label].job_id}
        try:
            out = i_jobs[label].download_output_data()
            preds[label] = np.concatenate([np.asarray(a).reshape(1, -1) for a in next(iter(out.values()))]).argmax(1)
        except Exception as exc:
            row["inference_error"] = f"{i_jobs[label].get_status().message} {exc}"
        try:
            prof = p_jobs[label].download_profile()
            row["latency_ms"] = prof["execution_summary"]["estimated_inference_time"] / 1000.0
            row["compute_units"] = dict(Counter(u.get("compute_unit") for u in prof.get("execution_detail", [])))
        except Exception as exc:
            row["profile_error"] = f"{p_jobs[label].get_status().message} {exc}"
        rows[label] = row
    ref = preds.get("fp32")
    for label, p in preds.items():
        right = p == labels
        rows[label]["accuracy"] = float(right.mean())
        if ref is not None and label != "fp32":
            ref_right = ref == labels
            b, c = int(np.sum(ref_right & ~right)), int(np.sum(~ref_right & right))
            d, lo, hi = paired_delta_ci(b, c, len(labels))
            rows[label].update(delta_pp=d, ci95_pp=[lo, hi], mcnemar_p=mcnemar_exact(b, c))
    result = {"model": args.model, "device": args.device, "runtime": args.runtime, "n": int(len(labels)),
              "ranking": [[s, g] for s, g in ranking], "variants": rows}
    slug = args.device.lower().replace(" ", "-").replace("(", "").replace(")", "")
    path = HERE / "results" / f"{args.model}-topk-{slug}-{args.runtime}-n{len(labels)}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{args.model} top-k equalisation on {args.device} ({args.runtime}), {len(labels)} images")
    for label in ["fp32", "top0", *[f"top{k}" for k in ks]]:
        r = rows.get(label, {})
        acc = f"{r['accuracy'] * 100:5.1f}%" if "accuracy" in r else "  -  "
        d = f"{r['delta_pp']:+6.1f}pp p={r['mcnemar_p']:.2g}" if "delta_pp" in r else ""
        lat = f"{r['latency_ms']:.3f} ms" if r.get("latency_ms") else ""
        print(f"  {label:8s} {acc} {d:22s} {lat}", flush=True)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
