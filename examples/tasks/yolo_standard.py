"""YOLOv8n as Ultralytics exports it, (1, 84, 8400): pixel boxes and 0-1 scores in one tensor.

The failure the float_mixed_outputs safeguard exists for: one 8-bit scale on that output rounds
every score to zero. Scored on COCO val2017 (box mAP@[.5:.95]) with the safeguard off and on
(on is the default), paired against FP32, with the fold CI of run_tasks.py.

    python yolo_standard.py --images 300
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import run_tasks as rt  # noqa: E402

from anneal.core.advise import BASE  # noqa: E402
from anneal.core.artifact import ModelArtifact  # noqa: E402
from anneal.core.transforms import TransformContext, apply_transform  # noqa: E402


class YOLOStandard(rt.YOLO):
    def per_image(self, outputs, meta, img_id) -> np.ndarray:
        import torchvision

        torch = self.torch
        out = torch.from_numpy(outputs[0][0]).T  # (8400, 84): cx, cy, w, h in letterbox pixels, 80 scores
        xy, wh, scores = out[:, :2], out[:, 2:4], out[:, 4:]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)
        i, j = torch.where(scores > 0.001)
        boxes, conf, cls = boxes[i], scores[i, j], j
        if len(conf) > 30000:
            top = conf.topk(30000).indices
            boxes, conf, cls = boxes[top], conf[top], cls[top]
        keep = torchvision.ops.batched_nms(boxes, conf, cls, 0.7)[:300]
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        left, top = meta["pad"]
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / meta["r"]).clamp(0, meta["orig"][1])
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / meta["r"]).clamp(0, meta["orig"][0])
        b = boxes.numpy()
        return np.column_stack([np.full(len(b), img_id), b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1],
                                conf.numpy(), self.cat_ids[cls.numpy()]]).astype(np.float64)


def export_standard() -> Path:
    from ultralytics import YOLO

    work = ROOT / "scratch" / "tasks"
    work.mkdir(parents=True, exist_ok=True)
    dst = ROOT / "examples" / "models" / "yolov8n-ultralytics-fp32.onnx"
    if dst.exists():
        return dst
    here = os.getcwd()
    os.chdir(work)
    try:
        path = Path(YOLO("yolov8n.pt").export(format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False)).resolve()
    finally:
        os.chdir(here)
    Path(path).replace(dst)
    return dst


def main() -> None:
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=300)
    ap.add_argument("--calib-images", type=int, default=16)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    api = rt.coco()
    task = YOLOStandard(sorted(api.getCatIds()))
    all_ids = task.image_ids(api)
    ids, calib_ids = all_ids[:args.images], all_ids[-args.calib_images:]
    path = export_standard()
    ctx = TransformContext(workdir=ROOT / "scratch" / "tasks" / "yolov8n-ultralytics",
                           calibset=rt.Calib([task.preprocess(api, i)[0] for i in calib_ids]))
    base = {**BASE, "calib_samples": args.calib_images}
    advised = {**base, "calibrate_method": "percentile", "calib_percentile": 99.999, "float_stem": True}
    recipes = {
        "onnxruntime default, safeguard off": {**base, "calibrate_method": "minmax", "float_mixed_outputs": False},
        "onnxruntime default + safeguard": {**base, "calibrate_method": "minmax"},
        "anneal advised, safeguard off": {**advised, "float_mixed_outputs": False},
        "anneal advised + safeguard (default)": advised,
    }
    built, meta = {}, {}
    for k, params in recipes.items():
        art = apply_transform("quantize_static_int8", dict(params), ModelArtifact(path=path), ctx)
        built[k], meta[k] = str(art.path), art.meta.get("mixed_range_outputs")
        print(f"  built {k}: flagged {meta[k]}", flush=True)
        gc.collect()

    def session(p: str, emulated: bool):
        so = ort.SessionOptions()
        if emulated:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

    sessions = {"fp32": session(str(path), False), **{k: session(p, True) for k, p in built.items()}}
    inp = sessions["fp32"].get_inputs()[0].name
    dets = {k: [] for k in sessions}
    for img_id in ids:
        x, m = task.preprocess(api, img_id)
        for k, s in sessions.items():
            dets[k].append(task.per_image(s.run(None, {inp: x}), m, img_id))
    dets = {k: np.concatenate(v) for k, v in dets.items()}
    folds = [f.tolist() for f in np.array_split(np.array(ids), 10)]

    def fold_maps(d):
        return [rt.coco_map(api, d[np.isin(d[:, 0], f)], f) for f in folds]

    fp = rt.coco_map(api, dets["fp32"], ids)
    fp_folds = fold_maps(dets["fp32"])
    result = {"model": "yolov8n (Ultralytics export, 1x84x8400)", "n": len(ids), "fp32_map": fp, "recipes": {}}
    print(f"\nFP32 mAP {100 * fp:.2f} on {len(ids)} images")
    for k in built:
        m_ = rt.coco_map(api, dets[k], ids)
        d = np.array(fold_maps(dets[k])) - np.array(fp_folds)
        half = rt.T_975_DF9 * d.std(ddof=1) / np.sqrt(len(d))
        result["recipes"][k] = {"map": m_, "delta_pts": 100 * (m_ - fp),
                                "ci95_pts": [100 * float(d.mean() - half), 100 * float(d.mean() + half)],
                                "params": recipes[k], "flagged": meta[k]}
        print(f"  {k:40s} mAP {100 * m_:5.2f}  {100 * (m_ - fp):+6.2f} pts "
              f"[{100 * (d.mean() - half):+.2f},{100 * (d.mean() + half):+.2f}]", flush=True)
    (HERE / "results" / "yolov8n_ultralytics_export.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
