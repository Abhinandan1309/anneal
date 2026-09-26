"""Do Anneal's INT8 findings hold on real edge silicon? Qualcomm AI Hub, real devices.

Everything else in this repository ran on server and laptop CPUs through onnxruntime. AI Hub
compiles a model for a real device (phones, automotive and IoT boards) and runs it there,
with Qualcomm's own toolchain: its quantizer, its TFLite or QNN runtime, its Hexagon NPU.

Per model and device, four variants, all scored on the device against its own FP32:

* ``fp32``                  the float model, compiled for the device (the reference)
* ``hub int8``              Qualcomm's quantizer (W8A8) on the model as exported: what a
                            developer following the standard AI Hub flow would ship
* ``hub int8 + equalised``  the same quantizer on Anneal's equalised model (exact in float)
* ``anneal recipe``         Anneal's own QDQ model (equalise + percentile + float stem),
                            built with onnxruntime and compiled as is

The images are Imagenette (public); ImageNet's licensed images are not uploaded.

    python run_qaihub.py --model efficientnet_b0 --device "Samsung Galaxy S24 (Family)" --runtime tflite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".anneal_cache"


VARIANTS = ["fp32", "hub int8", "hub int8 + equalised", "anneal recipe"]


def static_copy(path: Path, work: Path) -> Path:
    """A batch-1 copy with valid IO: AI Hub's quantizer needs static shapes, and its compiler
    (rightly) rejects graph inputs/outputs repeated in value_info, which onnxruntime's
    shape-inference pre-processing leaves behind."""
    import onnx

    model = onnx.load(str(path))
    for vi in list(model.graph.input) + list(model.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if len(dims):
            dims[0].ClearField("dim_param")
            dims[0].dim_value = 1
    io = {v.name for v in list(model.graph.input) + list(model.graph.output)}
    keep = [v for v in model.graph.value_info if v.name not in io]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)
    dst = work / f"{path.stem}-b1.onnx"
    onnx.save(model, str(dst))
    return dst


def target_of(job, label: str):
    """The job's output model, or None with the failure reason printed."""
    model = job.get_target_model()
    if model is None:
        print(f"  FAILED {label}: {job.get_status().message}", flush=True)
    return model


def main() -> None:
    import qai_hub as hub

    from anneal.core.artifact import ModelArtifact
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import equalise
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    ap.add_argument("--runtime", default="tflite", choices=["tflite", "qnn_dlc", "onnx"])
    ap.add_argument("--images", type=int, default=512)
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated subset of: " + ", ".join(VARIANTS))
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown or "fp32" not in variants:
        ap.error(f"--variants must include fp32 and be drawn from {VARIANTS}; got {variants}")

    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    work = ROOT / "scratch" / "qaihub" / args.model
    work.mkdir(parents=True, exist_ok=True)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64)
    calib_imgs = [x[i:i + 1] for x in calib.calibration_batches(64) for i in range(len(x))][:64]
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=16, limit=args.images)
    eval_imgs, labels = [], []
    for x, y in ev.batches():
        eval_imgs += [x[i:i + 1] for i in range(len(x))]
        labels += list(y)
    labels = np.array(labels)
    input_name = "input"

    eq = work / f"{args.model}-equalised.onnx"
    eq_result = equalise(src, eq, [np.concatenate(calib_imgs[i:i + 8]) for i in range(0, 64, 8)])
    print(f"{args.model}: {len(eq_result.sites)} equalised sites", flush=True)
    recipe = None if "anneal recipe" not in variants else apply_transform(
        "quantize_static_int8",
        {"per_channel": True, "activation_type": "uint8", "equalize": True,
         "calibrate_method": "percentile_asym", "float_stem": True},
        ModelArtifact(path=src), TransformContext(workdir=work, calibset=calib),
    )

    device = hub.Device(args.device)
    calibration = {input_name: calib_imgs}
    src_b1, eq_b1 = static_copy(src, work), static_copy(eq, work)
    to_quantize = {"hub int8": src_b1, "hub int8 + equalised": eq_b1}
    q_jobs = {label: hub.submit_quantize_job(str(path), calibration, name=f"anneal-{args.model}-{label}-q")
              for label, path in to_quantize.items() if label in variants}
    models = {"fp32": str(src_b1)}
    if recipe is not None:
        models["anneal recipe"] = str(static_copy(Path(recipe.path), work))
    for label, job in q_jobs.items():
        model = target_of(job, label)
        if model is not None:
            models[label] = model
            print(f"  quantized: {label}", flush=True)

    specs = {input_name: (1, 3, 224, 224)}
    c_jobs = {label: hub.submit_compile_job(m, device=device, input_specs=specs,
                                            options=f"--target_runtime {args.runtime}",
                                            name=f"anneal-{args.model}-{label}")
              for label, m in models.items()}
    targets = {}
    for label, job in c_jobs.items():
        model = target_of(job, label)  # a variant the toolchain cannot compile is a finding too
        if model is not None:
            targets[label] = model
            print(f"  compiled: {label}", flush=True)

    dataset = hub.upload_dataset({input_name: eval_imgs}, name=f"anneal-imagenette-{args.images}")
    i_jobs = {label: hub.submit_inference_job(t, device=device, inputs=dataset, name=f"anneal-{args.model}-{label}-inf")
              for label, t in targets.items()}
    p_jobs = {label: hub.submit_profile_job(t, device=device, name=f"anneal-{args.model}-{label}-prof")
              for label, t in targets.items()}

    preds, latency, units = {}, {}, {}
    for label in targets:
        try:
            out = i_jobs[label].download_output_data()
            if out is None:
                raise RuntimeError(i_jobs[label].get_status().message)
            logits = np.concatenate([np.asarray(a).reshape(1, -1) for a in next(iter(out.values()))])
            preds[label] = logits.argmax(1)
        except Exception as exc:
            print(f"  inference failed: {label}: {exc}", flush=True)
        try:
            prof = p_jobs[label].download_profile()
            latency[label] = prof["execution_summary"]["estimated_inference_time"] / 1000.0
            from collections import Counter

            units[label] = dict(Counter(u.get("compute_unit") for u in prof.get("execution_detail", [])))
        except Exception as exc:
            print(f"  profile failed: {label}: {exc}", flush=True)

    result = {"model": args.model, "device": args.device, "runtime": args.runtime, "n": int(len(labels)),
              "equalised_sites": len(eq_result.sites), "variants": {}}
    ref = preds.get("fp32")
    for label in variants:
        if label not in c_jobs:
            result["variants"][label] = {"compiled": False, "error": "quantize job failed"}
            continue
        row = {"compiled": label in targets, "latency_ms": latency.get(label), "compute_units": units.get(label),
               "compile_job": c_jobs[label].job_id}
        if label in preds:
            right = preds[label] == labels
            row["accuracy"] = float(right.mean())
            if ref is not None and label != "fp32":
                ref_right = ref == labels
                b, c = int(np.sum(ref_right & ~right)), int(np.sum(~ref_right & right))
                d, lo, hi = paired_delta_ci(b, c, len(labels))
                row.update(delta_pp=d, ci95_pp=[lo, hi], mcnemar_p=mcnemar_exact(b, c),
                           agreement=float(np.mean(preds[label] == ref)))
        result["variants"][label] = row
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.device.lower().replace(" ", "-").replace("(", "").replace(")", "")
    path = out_dir / f"{args.model}-{slug}-{args.runtime}-n{len(labels)}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{args.model} on {args.device} ({args.runtime}), {len(labels)} images", flush=True)
    for label, row in result["variants"].items():
        acc = f"{row['accuracy'] * 100:5.1f}%" if "accuracy" in row else "  -  "
        d = f"{row['delta_pp']:+6.1f}pp p={row['mcnemar_p']:.2g}" if "delta_pp" in row else ""
        lat = f"{row['latency_ms']:.2f} ms" if row.get("latency_ms") else ""
        print(f"  {label:24s} {acc} {d:22s} {lat:10s} {row.get('compute_units') or ''}", flush=True)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
