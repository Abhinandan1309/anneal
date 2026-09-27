"""LRASPP under the full TIDL-like rules (per-tensor weights + symmetric power-of-two int8
activations): candidate equalisation recipes, mIoU against ground truth, vs FP32."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples" / "tasks"))
sys.path.insert(0, str(HERE))
from lraspp_split import ACT  # noqa: E402


def main() -> None:
    import onnxruntime as ort
    from run_tasks import Calib, Segmentation, coco, miou

    from anneal.core.artifact import ModelArtifact
    from anneal.core.cle import cross_layer_equalise
    from anneal.core.equalize import equalise
    from anneal.core.transforms import TransformContext, apply_transform

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    api, task = coco(), Segmentation()
    ids = task.image_ids(api)
    cal = [task.preprocess(api, i)[0] for i in ids[-16:]]
    src = ROOT / "examples" / "models" / "lraspp_mobilenet_v3_large-fp32.onnx"
    w = ROOT / "scratch" / "lraspp_recipes"
    w.mkdir(parents=True, exist_ok=True)
    ctx = TransformContext(workdir=w, calibset=Calib(cal))
    recipes = {"eq se": dict(se=True), "eq se + mix": dict(se=True, mix=(0.5, 0.5)),
               "eq se + res + mix": dict(se=True, residual=True, mix=(0.5, 0.5)),
               "eq se + grid": dict(se=True, grid_inverse=True)}
    floats = {"plain": src}
    for k, kw in recipes.items():
        floats[k] = w / f"{k.replace(' ', '').replace('+', '_')}.onnx"
        equalise(src, floats[k], cal, **kw)
    floats["eq se + res + mix + cle4"] = w / "full-cle4.onnx"
    cross_layer_equalise(floats["eq se + res + mix"], floats["eq se + res + mix + cle4"], max_scale=4.0)
    built = {k: apply_transform("quantize_static_int8", {**ACT, "per_channel": False, "calib_samples": 16},
                                ModelArtifact(path=p), ctx).path for k, p in floats.items()}
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 4
    sess = {"fp32": ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]),
            **{k: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"]) for k, p in built.items()}}
    conf = {k: np.zeros((21, 21), np.int64) for k in sess}
    for i in ids[:n]:
        x, meta = task.preprocess(api, i)
        for k, s in sess.items():
            conf[k] += task.per_image(s.run(None, {"input": x}), meta, i)
    cls = np.flatnonzero(conf["fp32"].sum(1))
    base = miou(conf["fp32"], cls)
    print(f"fp32 mIoU {100 * base:.1f}", flush=True)
    for k in built:
        print(f"  {k:28s} {100 * (miou(conf[k], cls) - base):+7.1f} pts", flush=True)


if __name__ == "__main__":
    main()
