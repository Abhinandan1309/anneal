"""Where does a classic U-Net lose accuracy at INT8, and what fixes it?

The U-Net of unet_carvana.py (conv-BN-ReLU, transposed-conv up, skip concatenations), on COCO
val2017 photos containing a car, squashed to 192x288. Scored by fidelity to the FP32 model: the
car-class IoU between each INT8 mask and the FP32 mask (no Carvana labels needed), averaged over
images where FP32 finds a car, and pixel agreement. Calibration images are disjoint from the
scored ones. Emulated 32-bit accumulation (onnxruntime, graph optimisations off).

Recipes cover onnxruntime's default, Anneal's CNN recipe, and TIDL-like constraints: weights per
tensor, symmetric int8 activations, and one scale shared by a Concat's inputs.

    python unet_study.py --images 200
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "examples" / "tasks"))

BASE = {"per_channel": True, "activation_type": "uint8"}
RECIPES = {
    "minmax (onnxruntime default)": {**BASE, "calibrate_method": "minmax"},
    "percentile 99.999 + float stem (anneal cnn)": {**BASE, "calibrate_method": "percentile",
                                                     "calib_percentile": 99.999, "float_stem": True},
    "minmax + shared concat": {**BASE, "calibrate_method": "minmax", "concat_shared_scale": True},
    "anneal cnn + shared concat": {**BASE, "calibrate_method": "percentile", "calib_percentile": 99.999,
                                   "float_stem": True, "concat_shared_scale": True},
    "TIDL-like: per-tensor w, sym int8 acts, shared concat": {
        "per_channel": False, "activation_type": "int8", "calibrate_method": "minmax", "concat_shared_scale": True},
    # the same, on the concat-equalised model (anneal.core.equalize_concat)
    "concat-equalised + minmax + shared concat": {**BASE, "calibrate_method": "minmax", "concat_shared_scale": True,
                                                  "_pre": "concat"},
    "concat-equalised + anneal cnn + shared concat": {**BASE, "calibrate_method": "percentile",
                                                      "calib_percentile": 99.999, "float_stem": True,
                                                      "concat_shared_scale": True, "_pre": "concat"},
    "concat-equalised + TIDL-like": {"per_channel": False, "activation_type": "int8", "calibrate_method": "minmax",
                                     "concat_shared_scale": True, "_pre": "concat"},
    "concat-equalised + TIDL-like + pow2 + symmetric": {
        "per_channel": False, "activation_type": "int8", "calibrate_method": "minmax", "concat_shared_scale": True,
        "activation_symmetric": True, "pow2_activation_scales": True, "_pre": "concat"},
    "TIDL-like + pow2 + symmetric": {
        "per_channel": False, "activation_type": "int8", "calibrate_method": "minmax", "concat_shared_scale": True,
        "activation_symmetric": True, "pow2_activation_scales": True},
    "TIDL-like + percentile": {
        "per_channel": False, "activation_type": "int8", "calibrate_method": "percentile",
        "calib_percentile": 99.999, "concat_shared_scale": True},
}


def load(api, img_id: int, size) -> np.ndarray:
    from PIL import Image

    info = api.loadImgs(img_id)[0]
    img = Image.open(Path.home() / ".anneal_cache" / "coco" / "val2017" / info["file_name"]).convert("RGB")
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None].copy()


def main() -> None:
    import onnxruntime as ort
    import run_tasks as rt
    from unet_carvana import SIZE, export

    from anneal.core.artifact import ModelArtifact
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=200)
    ap.add_argument("--calib-images", type=int, default=8)
    ap.add_argument("--only", help="recipe labels separated by ';'")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    api = rt.coco()
    car = api.getCatIds(catNms=["car"])
    ids = sorted(api.getImgIds(catIds=car))
    scored, calib_ids = ids[:args.images], ids[-args.calib_images:]
    path = export()
    ctx = TransformContext(workdir=ROOT / "scratch" / "unet", calibset=rt.Calib([load(api, i, SIZE) for i in calib_ids]))
    recipes = {k: v for k, v in RECIPES.items()
               if not args.only or any(o.strip() and o.strip() in k for o in args.only.split(";"))}
    from anneal.core.equalize_concat import equalise_concat

    calib_imgs = [load(api, i, SIZE) for i in calib_ids]
    eq_path = ROOT / "scratch" / "unet" / "unet-concat-equalised.onnx"
    if any(v.get("_pre") == "concat" for v in recipes.values()):
        for r in equalise_concat(path, eq_path, calib_imgs):
            print(f"  concat-equalised {r}", flush=True)
    built = {}
    for k, params in recipes.items():
        params = dict(params)
        source = eq_path if params.pop("_pre", None) == "concat" else path
        built[k] = str(apply_transform("quantize_static_int8", {**params, "calib_samples": args.calib_images},
                                       ModelArtifact(path=source), ctx).path)
        print(f"  built {k}", flush=True)
        gc.collect()

    def session(p: str, emulated: bool):
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        if emulated:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

    sessions = {"fp32": session(str(path), False), **{k: session(p, True) for k, p in built.items()}}
    inp = sessions["fp32"].get_inputs()[0].name
    ious = {k: [] for k in built}
    agree = {k: [] for k in built}
    fg_share = []
    for img_id in scored:
        x = load(api, img_id, SIZE)
        masks = {k: s.run(None, {inp: x})[0][0].argmax(0) for k, s in sessions.items()}
        ref = masks["fp32"] == 1
        fg_share.append(float(ref.mean()))
        for k in built:
            m = masks[k] == 1
            agree[k].append(float((m == ref).mean()))
            if ref.any():
                ious[k].append(float((m & ref).sum() / (m | ref).sum()))
    result = {"model": "U-Net (milesial, Carvana, transposed-conv up)", "input": list(SIZE), "n": len(scored),
              "fp32_car_pixel_share": float(np.mean(fg_share)), "images_with_car_in_fp32": len(ious[next(iter(built))]),
              "recipes": {}}
    print(f"\nFP32 finds a car in {result['images_with_car_in_fp32']}/{len(scored)} images "
          f"(mean car pixel share {100 * result['fp32_car_pixel_share']:.1f}%)")
    rng = np.random.default_rng(0)
    for k in built:
        v = np.array(ious[k])
        boot = [rng.choice(v, len(v)).mean() for _ in range(1000)]
        result["recipes"][k] = {"car_iou_vs_fp32": float(v.mean()), "ci95": [float(np.percentile(boot, 2.5)),
                                                                            float(np.percentile(boot, 97.5))],
                                "pixel_agreement": float(np.mean(agree[k])),
                                "params": {kk: vv for kk, vv in recipes[k].items()}}
        print(f"  {k:56s} car IoU vs FP32 {v.mean():.3f} [{np.percentile(boot, 2.5):.3f},{np.percentile(boot, 97.5):.3f}]"
              f"  pixel agreement {np.mean(agree[k]):.4f}", flush=True)
    (HERE / ("unet_study" + ("_" + args.tag if args.tag else "") + ".json")).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
