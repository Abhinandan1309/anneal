"""Why does Anneal's own QDQ recipe compile for the Galaxy S24 but fail to run on it?

The first S24 run compiled `anneal recipe` (equalise + asymmetric percentile + float stem, built
by onnxruntime as a QDQ model) and then failed on the device with no message. This bisects it:
each variant drops one ingredient, and each is compiled and run on the device (a short inference
on 64 images, then accuracy against the device's FP32 if it runs). A variant that fails to
compile or run is the result.

    python run_recipe_debug.py --runtime tflite
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
from run_qaihub import static_copy  # noqa: E402

BASE = {"per_channel": True, "activation_type": "uint8", "calib_samples": 16}
VARIANTS = {
    "full: eq + asym pct + stem": {**BASE, "equalize": True, "calibrate_method": "percentile_asym", "float_stem": True},
    "no stem: eq + asym pct": {**BASE, "equalize": True, "calibrate_method": "percentile_asym"},
    "no eq: asym pct + stem": {**BASE, "calibrate_method": "percentile_asym", "float_stem": True},
    "plain QDQ: minmax": {**BASE, "calibrate_method": "minmax"},
    "int8 activations: eq + asym pct": {**BASE, "activation_type": "int8", "equalize": True, "calibrate_method": "percentile_asym"},
}


def main() -> None:
    import qai_hub as hub

    from anneal.core.artifact import ModelArtifact
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    ap.add_argument("--runtime", default="tflite", choices=["tflite", "qnn_dlc", "onnx"])
    ap.add_argument("--images", type=int, default=256)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    work = ROOT / "scratch" / "qaihub" / f"{args.model}-recipe-debug"
    work.mkdir(parents=True, exist_ok=True)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=16)
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=16, limit=args.images)
    eval_imgs, labels = [], []
    for x, y in ev.batches():
        eval_imgs += [x[i:i + 1] for i in range(len(x))]
        labels += list(y)
    labels = np.array(labels)

    models = {"fp32": static_copy(src, work)}
    for label, params in VARIANTS.items():
        art = apply_transform("quantize_static_int8", dict(params), ModelArtifact(path=src),
                              TransformContext(workdir=work / label.split(":")[0].replace(" ", "_"), calibset=calib))
        models[label] = static_copy(Path(art.path), work / label.split(":")[0].replace(" ", "_"))
        print(f"  built {label}", flush=True)

    device = hub.Device(args.device)
    c_jobs = {k: hub.submit_compile_job(str(p), device=device, input_specs={"input": (1, 3, 224, 224)},
                                        options=f"--target_runtime {args.runtime}", name=f"anneal-debug-{k}")
              for k, p in models.items()}
    dataset = hub.upload_dataset({"input": eval_imgs}, name=f"anneal-imagenette-{args.images}")
    rows, preds, i_jobs = {}, {}, {}
    for k, j in c_jobs.items():
        st = j.wait()
        rows[k] = {"compile_job": j.job_id, "compiled": st.success, "compile_message": st.message}
        if st.success:
            i_jobs[k] = hub.submit_inference_job(j.get_target_model(), device=device, inputs=dataset,
                                                 name=f"anneal-debug-{k}-inf")
    for k, j in i_jobs.items():
        st = j.wait()
        rows[k].update(inference_job=j.job_id, ran=st.success, run_message=st.message)
        if st.success:
            out = j.download_output_data()
            preds[k] = np.concatenate([np.asarray(a).reshape(1, -1) for a in next(iter(out.values()))]).argmax(1)
            rows[k]["accuracy"] = float((preds[k] == labels).mean())
    if "fp32" in preds:
        for k, p in preds.items():
            rows[k]["delta_pp_vs_device_fp32"] = 100 * float((p == labels).mean() - (preds["fp32"] == labels).mean())
    result = {"model": args.model, "device": args.device, "runtime": args.runtime, "n": int(len(labels)),
              "variants": {k: {**rows[k], "params": VARIANTS.get(k)} for k in rows}}
    slug = args.device.lower().replace(" ", "-").replace("(", "").replace(")", "")
    path = HERE / "results" / f"{args.model}-recipe-debug-{slug}-{args.runtime}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for k, r in rows.items():
        state = "compile FAILED" if not r["compiled"] else ("run FAILED" if not r.get("ran") else f"{100 * r['accuracy']:.1f}%")
        print(f"  {k:36s} {state:16s} {(r.get('run_message') or r.get('compile_message') or '')[:120]}", flush=True)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
