"""What is each equalised site worth on a real NPU? Leave-one-site-out on the Galaxy S24.

Equalising the top-k sites by *predicted* gain recovered accuracy only gradually (k=8: -6.4pp,
k=16: -0.7pp; run_topk.py), so the prediction does not rank sites by their value on the device.
This measures it: variant ``-site`` equalises every site except one. Its accuracy loss against full
equalisation is that site's marginal value; its latency saving is that site's cost. Quantized with
Qualcomm's quantizer and scored on the device, paired.

    python run_site_value.py --runtime qnn_dlc
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
    ap.add_argument("--images", type=int, default=1024)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    work = ROOT / "scratch" / "qaihub" / f"{args.model}-site-value"
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
    all_sites = [s for s, _ in ranking]
    models = {"fp32": static_copy(src, work)}
    full = work / f"{args.model}-eq-all.onnx"
    equalise(src, full, batches)
    models["all"] = static_copy(full, work)
    for i, site in enumerate(all_sites):
        dst = work / f"{args.model}-eq-minus{i}.onnx"
        equalise(src, dst, batches, sites=[s for s in all_sites if s != site])
        models[f"minus{i}"] = static_copy(dst, work)

    device = hub.Device(args.device)
    calibration = {"input": calib_imgs}
    q_jobs = {label: hub.submit_quantize_job(str(p), calibration, name=f"anneal-{args.model}-{label}-q")
              for label, p in models.items() if label != "fp32"}
    q_jobs["none"] = hub.submit_quantize_job(str(models["fp32"]), calibration, name=f"anneal-{args.model}-none-q")
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
    full_right = preds["all"] == labels if "all" in preds else None
    for i, site in enumerate(all_sites):
        r = rows.get(f"minus{i}", {})
        r["left_out_site"], r["predicted_gain"] = site, dict(ranking)[site]
        if full_right is not None and f"minus{i}" in preds:
            right = preds[f"minus{i}"] == labels
            b, c = int(np.sum(full_right & ~right)), int(np.sum(~full_right & right))
            d, lo, hi = paired_delta_ci(b, c, len(labels))
            r["vs_all"] = {"delta_pp": d, "ci95_pp": [lo, hi], "mcnemar_p": mcnemar_exact(b, c)}
        if "latency_ms" in r and "latency_ms" in rows.get("all", {}):
            r["latency_saving_ms"] = rows["all"]["latency_ms"] - r["latency_ms"]
    result = {"model": args.model, "device": args.device, "runtime": args.runtime, "n": int(len(labels)),
              "ranking": [[s, g] for s, g in ranking], "variants": rows}
    slug = args.device.lower().replace(" ", "-").replace("(", "").replace(")", "")
    path = HERE / "results" / f"{args.model}-site-value-{slug}-{args.runtime}-n{len(labels)}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{args.model} leave-one-site-out on {args.device} ({args.runtime}), {len(labels)} images")
    for label in ["fp32", "none", "all", *[f"minus{i}" for i in range(len(all_sites))]]:
        r = rows.get(label, {})
        acc = f"{r['accuracy'] * 100:5.1f}%" if "accuracy" in r else "  -  "
        d = (f"{r['vs_all']['delta_pp']:+6.1f}pp vs all" if "vs_all" in r
             else f"{r['delta_pp']:+6.1f}pp vs fp32" if "delta_pp" in r else "")
        lat = f"{r['latency_ms']:.3f} ms" if r.get("latency_ms") else ""
        site = f"{r['left_out_site'][:40]} (pred {r['predicted_gain']:.1f})" if "left_out_site" in r else ""
        print(f"  {label:8s} {acc} {d:18s} {lat:10s} {site}", flush=True)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
