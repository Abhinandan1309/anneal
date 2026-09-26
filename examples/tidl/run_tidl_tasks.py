"""Detection and segmentation on TI TDA4VM (TIDL host emulation): does 8-bit hold, and where not?

The classification harness (run_tidl.py) with the COCO tasks of examples/tasks/run_tasks.py:
``yolov8n`` (box mAP@[.5:.95], raw DFL + class logits out, decoded on the CPU) and
``lraspp_mobilenet_v3_large`` (mIoU over the 21 VOC classes). Calibration uses the task's
highest-id val images; scoring the first ``--images``, never the same ones. Paired against FP32
(onnxruntime CPU) with the task's own uncertainty (fold CI for mAP, bootstrap for mIoU).

Variants: ``tidl 16-bit``, ``tidl 8-bit``, ``tidl 8-bit + equalised`` (gated depthwise sites and
gated dense sites, i.e. Conv -> SiLU -> Conv as in YOLOv8).

    python examples/tidl/run_tidl_tasks.py --model yolov8n --images 300 --out result.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "tasks"))
sys.path.insert(0, str(ROOT / "examples" / "tidl"))

from run_tidl import COMMON, session  # noqa: E402

VARIANTS = {
    "tidl 16-bit": ("plain", {**COMMON, "tensor_bits": 16}),
    "tidl 8-bit": ("plain", COMMON),
    "tidl 8-bit + equalised": ("equalised", COMMON),
    # U-Net: the skip concatenations share one scale on TIDL (anneal.core.equalize_concat), and
    # conv-ReLU pairs meet per-tensor weights (anneal.core.cle)
    "tidl 8-bit + concat eq": ("concat_eq", COMMON),
    "tidl 8-bit + concat eq + cle": ("concat_eq_cle", COMMON),
}


class UNetFidelity:
    """COCO val2017 photos with a car, 192x288, pixels in [0, 1]; scored against FP32's own masks."""

    def __init__(self, api) -> None:
        sys.path.insert(0, str(ROOT / "examples" / "segmentation"))
        from unet_carvana import SIZE

        self.size = SIZE
        self.car = api.getCatIds(catNms=["car"])

    def image_ids(self, api) -> list[int]:
        return sorted(api.getImgIds(catIds=self.car))

    def preprocess(self, api, img_id):
        from unet_study import load

        return load(api, img_id, self.size), {}

    def per_image(self, outputs, meta, img_id) -> np.ndarray:
        return outputs[0][0].argmax(0) == 1


def main() -> None:
    import onnx
    import run_tasks as rt
    from export_models import export

    from anneal.core.equalize import equalise
    from anneal.core.equalize_dense import equalise_dense

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolov8n", "lraspp_mobilenet_v3_large", "unet_carvana"])
    ap.add_argument("--images", type=int, default=300)
    ap.add_argument("--calib-images", type=int, default=16)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    tools = os.environ.get("TIDL_TOOLS_PATH")
    if not tools:
        raise SystemExit("TIDL_TOOLS_PATH is not set: source edgeai-tidl-tools/scripts/setup/setup_env.sh J721E")

    api = rt.coco()
    if args.model == "unet_carvana":
        task = UNetFidelity(api)
    else:
        task = rt.Segmentation() if args.model.startswith("lraspp") else rt.YOLO(sorted(api.getCatIds()))
    all_ids = task.image_ids(api)
    ids, calib_ids = all_ids[:args.images], all_ids[-args.calib_images:]
    assert not set(ids) & set(calib_ids)

    work = Path("tidl-work") / args.model
    work.mkdir(parents=True, exist_ok=True)
    if args.model == "unet_carvana":
        from unet_carvana import export as export_unet

        exported = export_unet()
    else:
        exported = export(args.model)  # examples/models/<model>-fp32.onnx
    src = work / f"{args.model}-fp32.onnx"
    m = onnx.load(str(exported))
    for vi in list(m.graph.input) + list(m.graph.output):
        d = vi.type.tensor_type.shape.dim
        if len(d):
            d[0].ClearField("dim_param")
            d[0].dim_value = 1
    onnx.save(m, str(src))
    calib = [task.preprocess(api, i)[0] for i in calib_ids]
    eq = work / f"{args.model}-equalised.onnx"
    sites = len(equalise(src, eq, calib).sites)
    dense, _, _ = equalise_dense(eq, eq, calib)
    print(f"{args.model}: equalised {sites} depthwise sites, {len(dense)} dense sites", flush=True)
    from anneal.core.cle import cross_layer_equalise
    from anneal.core.equalize_concat import equalise_concat

    cat_eq, cat_eq_cle = work / f"{args.model}-concat-eq.onnx", work / f"{args.model}-concat-eq-cle.onnx"
    print(f"{args.model}: concat sites {len(equalise_concat(src, cat_eq, calib))}", flush=True)
    cross_layer_equalise(cat_eq, cat_eq_cle)
    models = {"plain": src, "equalised": eq, "concat_eq": cat_eq, "concat_eq_cle": cat_eq_cle}
    for p in models.values():
        onnx.shape_inference.infer_shapes_path(str(p), str(p))

    fp = session(src, ["CPUExecutionProvider"], None)
    inp = fp.get_inputs()[0].name
    sessions = {"fp32": fp}
    rows, timing = {}, {}
    for label in [v.strip() for v in args.variants.split(",") if v.strip()]:
        which, opts = VARIANTS[label]
        opts = {**opts, "advanced_options:calibration_frames": len(calib)}
        art = work / "artifacts" / label.replace(" ", "_").replace("+", "plus")
        shutil.rmtree(art, ignore_errors=True)
        art.mkdir(parents=True)
        t = time.time()
        try:
            comp = session(models[which], ["TIDLCompilationProvider", "CPUExecutionProvider"],
                           {**opts, "artifacts_folder": str(art), "tidl_tools_path": tools})
            for x in calib:
                comp.run(None, {inp: x})
            del comp
            sessions[label] = session(models[which], ["TIDLExecutionProvider", "CPUExecutionProvider"],
                                      {"artifacts_folder": str(art), "debug_level": 0})
            timing[label] = {"compile_s": time.time() - t}
            print(f"  {label}: compiled ({timing[label]['compile_s']:.0f}s)", flush=True)
        except Exception as exc:  # a variant the toolchain cannot compile is a result too
            rows[label] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            print(f"  {label}: FAILED {rows[label]['error']}", flush=True)

    per_image = {k: [] for k in sessions}
    t0 = time.time()
    for n, img_id in enumerate(ids, 1):
        x, meta = task.preprocess(api, img_id)
        for k, s in sessions.items():
            try:
                per_image[k].append(task.per_image(s.run(None, {inp: x}), meta, img_id))
            except Exception as exc:  # noqa: BLE001
                rows[k] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        if n % 25 == 0:
            print(f"  {n}/{len(ids)} images ({time.time() - t0:.0f}s)", flush=True)
    scored = [k for k in sessions if k != "fp32" and k not in rows]

    result = {"soc": "J721E (TDA4VM)", "model": args.model, "n": len(ids), "calibration_images": len(calib),
              "equalised_sites": sites, "equalised_dense_sites": len(dense), "variants": {}}
    if isinstance(task, UNetFidelity):
        result["metric"] = "car IoU against the FP32 model's mask (fidelity)"
        result["fp32"] = 1.0
        ref = per_image["fp32"]
        rng = np.random.default_rng(0)
        for k in scored:
            ious = np.array([float((a & b).sum() / max((a | b).sum(), 1)) for a, b in zip(per_image[k], ref) if b.any()])
            boot = [rng.choice(ious, len(ious)).mean() for _ in range(1000)]
            result["variants"][k] = {"metric": float(ious.mean()), "delta_pts": 100 * (float(ious.mean()) - 1),
                                     "ci95_pts": [100 * (float(np.percentile(boot, 2.5)) - 1),
                                                  100 * (float(np.percentile(boot, 97.5)) - 1)],
                                     "pixel_agreement": float(np.mean([(a == b).mean() for a, b in zip(per_image[k], ref)])),
                                     **timing.get(k, {})}
    elif isinstance(task, rt.Segmentation):
        conf = {k: np.stack(v) for k, v in per_image.items() if k == "fp32" or k in scored}
        classes = np.flatnonzero(conf["fp32"].sum((0, 2)) > 0)
        result["metric"] = "mIoU (21 VOC classes)"
        result["fp32"] = rt.miou(conf["fp32"].sum(0), classes)
        boots = np.random.default_rng(0).integers(0, len(ids), size=(1000, len(ids)))
        fp_boot = [rt.miou(conf["fp32"][b].sum(0), classes) for b in boots]
        for k in scored:
            m_ = rt.miou(conf[k].sum(0), classes)
            deltas = [rt.miou(conf[k][b].sum(0), classes) - f for b, f in zip(boots, fp_boot)]
            result["variants"][k] = {"metric": m_, "delta_pts": 100 * (m_ - result["fp32"]),
                                     "ci95_pts": [100 * float(np.percentile(deltas, 2.5)),
                                                  100 * float(np.percentile(deltas, 97.5))], **timing.get(k, {})}
    else:
        result["metric"] = "COCO box mAP@[.5:.95]"
        dets = {k: np.concatenate(v) for k, v in per_image.items() if k == "fp32" or k in scored}
        folds = [f.tolist() for f in np.array_split(np.array(ids), 10)]

        def fold_maps(d: np.ndarray) -> list[float]:
            return [rt.coco_map(api, d[np.isin(d[:, 0], f)], f) for f in folds]

        result["fp32"] = rt.coco_map(api, dets["fp32"], ids)
        fp_folds = fold_maps(dets["fp32"])
        for k in scored:
            m_ = rt.coco_map(api, dets[k], ids)
            d = np.array(fold_maps(dets[k])) - np.array(fp_folds)
            half = rt.T_975_DF9 * d.std(ddof=1) / np.sqrt(len(d))
            result["variants"][k] = {"metric": m_, "delta_pts": 100 * (m_ - result["fp32"]),
                                     "ci95_pts": [100 * float(d.mean() - half), 100 * float(d.mean() + half)],
                                     **timing.get(k, {})}
    result["variants"].update(rows)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{args.model}: FP32 {result['metric']} {100 * result['fp32']:.2f} on {len(ids)} images")
    for k, r in result["variants"].items():
        if "error" in r:
            print(f"  {k:26s} ERROR {r['error'][:150]}")
        else:
            print(f"  {k:26s} {r['delta_pts']:+6.2f} pts [{r['ci95_pts'][0]:+.2f},{r['ci95_pts'][1]:+.2f}]", flush=True)


if __name__ == "__main__":
    main()
