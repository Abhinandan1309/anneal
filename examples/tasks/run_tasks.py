"""Does the INT8 failure, and Anneal's fix, carry over from classification to the edge
segmentation and detection networks people actually deploy?

COCO val2017 (public). Calibration uses the 32 highest-id val images of each task; scoring uses
the first ``--images`` by id, never the same images. Every recipe is scored as 32-bit-accumulating
hardware would run it (onnxruntime, graph optimisations off), the way the edge NPUs in
examples/qaihub behave, and paired against the same model's FP32 on the same images.

* ``lraspp_mobilenet_v3_large``      segmentation, mIoU over the 21 VOC classes of COCO, on images
                                    containing a VOC object, squashed to 512x512. Not torchvision's
                                    protocol (all 5,000 images, short side 520), so FP32 sits below
                                    the published 57.9; the paired comparison is unaffected.
* ``ssdlite320_mobilenet_v3_large``  detection, COCO box mAP@[.5:.95], torchvision's preprocessing
* ``yolov8n``                        detection, COCO box mAP@[.5:.95], letterboxed 640, multi-label
                                    NMS as in Ultralytics' COCO validation

Uncertainty: segmentation deltas carry a paired bootstrap 95% CI over images (confusion
matrices resampled together). Detection deltas carry a fold-level 95% CI: mAP recomputed on each
tenth of the images, t-interval on the ten paired deltas (centred on their mean, which need not
equal the pooled delta, since mAP is not additive over images).

Calibration holds every activation of every calibration image in RAM (onnxruntime); at 512-640
pixels that is ~100-150 MB per image, hence 32 images and every model built before any
inference session opens.

    python run_tasks.py --model lraspp_mobilenet_v3_large --images 500
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COCO = Path.home() / ".anneal_cache" / "coco"
CALIB_IMAGES = 32

from anneal.core.advise import BASE, advise  # noqa: E402
from anneal.core.artifact import ModelArtifact  # noqa: E402
from anneal.core.measure import EvalSet  # noqa: E402
from anneal.core.transforms import TransformContext, apply_transform  # noqa: E402

#: torchvision's COCO -> VOC mapping (references/segmentation/coco_utils.py): index = VOC class.
VOC_CATS = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
T_975_DF9 = 2.2622  # Student t, 97.5th percentile, 9 degrees of freedom (10 folds)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def coco():
    import contextlib
    import io

    from pycocotools.coco import COCO as _COCO

    with contextlib.redirect_stdout(io.StringIO()):
        return _COCO(str(COCO / "annotations" / "instances_val2017.json"))


def load_rgb(api, img_id: int) -> np.ndarray:
    from PIL import Image

    info = api.loadImgs(img_id)[0]
    return np.asarray(Image.open(COCO / "val2017" / info["file_name"]).convert("RGB"))


def pil_resize(img: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((size[1], size[0]), Image.NEAREST if nearest else Image.BILINEAR))


def torch_resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """[0,1] float CHW, bilinear without antialiasing: what torchvision's detection transform and
    OpenCV's INTER_LINEAR (Ultralytics) do, unlike PIL's antialiased downscale."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)[0].numpy()


class Task:
    name: str

    def image_ids(self, api) -> list[int]:
        return sorted(api.getImgIds())

    def preprocess(self, api, img_id: int) -> tuple[np.ndarray, dict]:
        raise NotImplementedError


class Segmentation(Task):
    name, size = "lraspp_mobilenet_v3_large", (512, 512)

    def image_ids(self, api):
        cats = set(VOC_CATS[1:])
        return [i for i in sorted(api.getImgIds())
                if any(a["category_id"] in cats and not a["iscrowd"] for a in api.loadAnns(api.getAnnIds(imgIds=i)))]

    def target(self, api, img_id: int) -> np.ndarray:
        info = api.loadImgs(img_id)[0]
        masks, cats = [], []
        for a in api.loadAnns(api.getAnnIds(imgIds=img_id, iscrowd=None)):
            if a["category_id"] in VOC_CATS:
                masks.append(api.annToMask(a))
                cats.append(VOC_CATS.index(a["category_id"]))
        if not masks:
            return np.zeros((info["height"], info["width"]), np.uint8)
        m = np.stack(masks)
        t = (m * np.array(cats, np.uint8)[:, None, None]).max(0)
        t[m.sum(0) > 1] = 255  # torchvision: pixels covered by several objects are ignored
        return t

    def preprocess(self, api, img_id):
        x = pil_resize(load_rgb(api, img_id), self.size).astype(np.float32) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)
        return x, {"target": pil_resize(self.target(api, img_id), self.size, nearest=True)}

    def per_image(self, outputs: list[np.ndarray], meta: dict, img_id: int) -> np.ndarray:
        pred, t = outputs[0][0].argmax(0), meta["target"]
        keep = t != 255
        return np.bincount(21 * t[keep].astype(np.int64) + pred[keep], minlength=441).reshape(21, 21)


def miou(conf: np.ndarray, classes: np.ndarray) -> float:
    """Mean IoU over ``classes`` (fixed by the ground truth, so every model averages the same set)."""
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    return float(np.mean(inter[classes] / np.maximum(union[classes], 1)))


class SSDLite(Task):
    name, size = "ssdlite320_mobilenet_v3_large", (320, 320)

    def __init__(self) -> None:
        import torch
        from torchvision.models.detection.image_list import ImageList

        sys.path.insert(0, str(HERE))
        from export_models import ssdlite

        self.torch, self.net = torch, ssdlite()
        with torch.no_grad():
            feats = list(self.net.backbone(torch.zeros(1, 3, *self.size)).values())
            self.anchors = self.net.anchor_generator(ImageList(torch.zeros(1, 3, *self.size), [self.size]), feats)

    def preprocess(self, api, img_id):
        img = load_rgb(api, img_id)
        x = ((torch_resize(img, self.size) - 0.5) / 0.5)[None].astype(np.float32)  # SSDLite: mean 0.5, std 0.5
        return x, {"orig": img.shape[:2]}

    def per_image(self, outputs, meta, img_id) -> np.ndarray:
        torch = self.torch
        head = {"bbox_regression": torch.from_numpy(outputs[0]), "cls_logits": torch.from_numpy(outputs[1])}
        with torch.no_grad():
            det = self.net.postprocess_detections(head, self.anchors, [self.size])
            det = self.net.transform.postprocess(det, [self.size], [tuple(meta["orig"])])[0]
        b = det["boxes"].numpy()
        return np.column_stack([np.full(len(b), img_id), b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1],
                                det["scores"].numpy(), det["labels"].numpy()]).astype(np.float64)


class YOLO(Task):
    name, size = "yolov8n", (640, 640)

    def __init__(self, cat_ids: list[int]) -> None:
        import torch

        self.torch, self.cat_ids = torch, np.array(cat_ids)
        # Anchor centres and strides of YOLOv8's three levels (80x80 / 8, 40x40 / 16, 20x20 / 32).
        pts, strides = [], []
        for n, s in ((80, 8), (40, 16), (20, 32)):
            ys, xs = torch.meshgrid(torch.arange(n) + 0.5, torch.arange(n) + 0.5, indexing="ij")
            pts.append(torch.stack((xs, ys), -1).view(-1, 2))
            strides.append(torch.full((n * n, 1), float(s)))
        self.anchors, self.strides = torch.cat(pts), torch.cat(strides)
        self.bins = torch.arange(16, dtype=torch.float32)

    def preprocess(self, api, img_id):
        img = load_rgb(api, img_id)
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        nh, nw = round(h * r), round(w * r)
        canvas = np.full((3, 640, 640), 114 / 255.0, np.float32)
        top, left = (640 - nh) // 2, (640 - nw) // 2
        canvas[:, top:top + nh, left:left + nw] = torch_resize(img, (nh, nw)) if (nh, nw) != (h, w) else img.transpose(2, 0, 1) / 255.0
        return np.ascontiguousarray(canvas[None]), {"r": r, "pad": (left, top), "orig": (h, w)}

    def per_image(self, outputs, meta, img_id) -> np.ndarray:
        import torchvision

        torch = self.torch
        box_logits = torch.from_numpy(outputs[0][0])  # (64, 8400): 4 sides x 16 DFL bins
        dist = (box_logits.view(4, 16, -1).softmax(1) * self.bins[None, :, None]).sum(1).T  # (8400, 4) ltrb
        xy1 = (self.anchors - dist[:, :2]) * self.strides
        xy2 = (self.anchors + dist[:, 2:]) * self.strides
        boxes = torch.cat([xy1, xy2], 1)
        scores = torch.from_numpy(outputs[1][0]).T.sigmoid()  # (8400, 80)
        i, j = torch.where(scores > 0.001)  # multi-label, as Ultralytics' COCO val
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


class Calib(EvalSet):
    """Preprocessed images, one per batch (the exported graphs have a fixed batch of 1)."""

    def __init__(self, xs: list[np.ndarray]) -> None:
        self.xs = xs

    def batches(self):
        for x in self.xs:
            yield x, np.zeros(1, np.int64)

    def calibration_batches(self, limit: int):
        yield from self.xs[:limit]

    def __len__(self) -> int:
        return len(self.xs)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def coco_map(api, dets: np.ndarray, ids: list[int]) -> float:
    """COCO box mAP@[.5:.95] of (N, 7) [image_id, x, y, w, h, score, category_id] detections."""
    import contextlib
    import io

    from pycocotools.cocoeval import COCOeval

    if not len(dets):
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(api, api.loadRes(dets), "bbox")
        e.params.imgIds = list(ids)
        e.evaluate()
        e.accumulate()
        e.summarize()
    return float(e.stats[0])


def recipes_for(path: Path, calib_images: int) -> tuple[dict[str, dict], object]:
    adv = advise(path, "arm-dotprod")
    base = {**BASE, "calib_samples": calib_images}
    out = {
        "minmax (onnxruntime default)": {**base, "calibrate_method": "minmax"},
        "percentile 99.999": {**base, "calibrate_method": "percentile", "calib_percentile": 99.999},
        f"anneal advised [{adv.profile.family}]: {adv.recommended.label}": {**adv.recommended.params, "calib_samples": calib_images},
    }
    if adv.profile.family == "gated-depthwise":
        out["anneal without equalisation: percentile + float stem"] = {
            **base, "calibrate_method": "percentile_asym", "float_stem": True}
    else:  # SiLU into dense convs (YOLO): the experimental dense-consumer equalisation
        out["anneal advised + dense equalisation"] = {**adv.recommended.params, "calib_samples": calib_images,
                                                      "equalize_dense": True}
    return out, adv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lraspp_mobilenet_v3_large", "ssdlite320_mobilenet_v3_large", "yolov8n"])
    ap.add_argument("--images", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--calib-images", type=int, default=CALIB_IMAGES,
                    help="fewer for large inputs: calibration keeps every activation of every image in RAM")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    api = coco()
    task = {"lraspp_mobilenet_v3_large": Segmentation, "ssdlite320_mobilenet_v3_large": SSDLite}.get(args.model)
    task = task() if task else YOLO(sorted(api.getCatIds()))
    all_ids = task.image_ids(api)
    ids, calib_ids = all_ids[:args.images], all_ids[-args.calib_images:]
    assert not set(ids) & set(calib_ids)
    path = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    work = ROOT / "scratch" / "tasks" / args.model
    ctx = TransformContext(workdir=work, calibset=Calib([task.preprocess(api, i)[0] for i in calib_ids]))

    # Build every model first: calibration holds all activations, and open sessions on top of
    # that exhausted this laptop's RAM in an earlier run.
    recipes, adv = recipes_for(path, args.calib_images)
    built, failed = {}, {}
    for label, params in recipes.items():
        t = time.time()
        try:
            built[label] = str(apply_transform("quantize_static_int8", dict(params), ModelArtifact(path=path), ctx).path)
            print(f"  built {label} ({time.time() - t:.0f}s)", flush=True)
        except Exception as exc:  # a recipe that cannot be built is a result too
            failed[label] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED to build {label}: {failed[label]}", flush=True)
        gc.collect()

    def session(p: str, emulated: bool) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        if emulated:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

    sessions = {"fp32": session(str(path), False), **{k: session(p, True) for k, p in built.items()}}
    inp = sessions["fp32"].get_inputs()[0].name
    per_image: dict[str, list] = {k: [] for k in sessions}
    t0 = time.time()
    for n, img_id in enumerate(ids, 1):
        x, meta = task.preprocess(api, img_id)
        for k, s in sessions.items():
            per_image[k].append(task.per_image(s.run(None, {inp: x}), meta, img_id))
        if n % 50 == 0:
            print(f"  {n}/{len(ids)} images ({time.time() - t0:.0f}s)", flush=True)
    with open(work / "per_image.pkl", "wb") as f:  # kept before scoring, so a scoring bug loses nothing
        pickle.dump({"ids": ids, "per_image": per_image}, f)

    result = {"model": args.model, "n": len(ids),
              "calibration": f"{args.calib_images} highest-id COCO val2017 images of this task, disjoint from the scored ones",
              "mode": "emulated (32-bit accumulation)", "advice": adv.to_dict(), "failed_to_build": failed, "recipes": {}}
    if isinstance(task, Segmentation):
        conf = {k: np.stack(v) for k, v in per_image.items()}
        classes = np.flatnonzero(conf["fp32"].sum((0, 2)) > 0)
        result["metric"] = "mIoU (21 VOC classes)"
        result["fp32"] = miou(conf["fp32"].sum(0), classes)
        boots = np.random.default_rng(0).integers(0, len(ids), size=(1000, len(ids)))
        fp_boot = [miou(conf["fp32"][b].sum(0), classes) for b in boots]
        for k in built:
            m = miou(conf[k].sum(0), classes)
            deltas = [miou(conf[k][b].sum(0), classes) - f for b, f in zip(boots, fp_boot)]
            result["recipes"][k] = {"metric": m, "delta_pts": 100 * (m - result["fp32"]),
                                    "ci95_pts": [100 * float(np.percentile(deltas, 2.5)), 100 * float(np.percentile(deltas, 97.5))],
                                    "params": recipes[k], "model_path": built[k]}
    else:
        result["metric"] = "COCO box mAP@[.5:.95]"
        dets = {k: np.concatenate(v) for k, v in per_image.items()}
        folds = [f.tolist() for f in np.array_split(np.array(ids), 10)]

        def fold_maps(d: np.ndarray) -> list[float]:
            return [coco_map(api, d[np.isin(d[:, 0], f)], f) for f in folds]

        result["fp32"] = coco_map(api, dets["fp32"], ids)
        fp_folds = fold_maps(dets["fp32"])
        for k in built:
            m = coco_map(api, dets[k], ids)
            d = np.array(fold_maps(dets[k])) - np.array(fp_folds)
            half = T_975_DF9 * d.std(ddof=1) / np.sqrt(len(d))
            result["recipes"][k] = {"metric": m, "delta_pts": 100 * (m - result["fp32"]),
                                    "fold_mean_delta_pts": 100 * float(d.mean()),
                                    "ci95_pts": [100 * float(d.mean() - half), 100 * float(d.mean() + half)],
                                    "params": recipes[k], "model_path": built[k]}
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.model}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n{args.model}: FP32 {result['metric']} {100 * result['fp32']:.2f} on {len(ids)} images")
    for k, r in result["recipes"].items():
        print(f"  {k:70s} {r['delta_pts']:+6.2f} pts [{r['ci95_pts'][0]:+.2f},{r['ci95_pts'][1]:+.2f}]", flush=True)
    for k, e in failed.items():
        print(f"  {k:70s} FAILED TO BUILD: {e[:120]}", flush=True)


if __name__ == "__main__":
    main()
