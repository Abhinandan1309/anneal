"""Which equalisation option breaks LRASPP, and under which error? Pixel agreement with FP32.

Each equalised model is exact in float; the question is what quantization does to it. Two errors
are applied separately: per-tensor int8 weights (float activations), and TIDL-like activations
(symmetric power-of-two int8, per-channel weights).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples" / "tasks"))
sys.path.insert(0, str(HERE))
from lraspp_split import ACT, fake_quant_weights  # noqa: E402


def main() -> None:
    import onnxruntime as ort
    from run_tasks import Calib, Segmentation, coco

    from anneal.core.artifact import ModelArtifact
    from anneal.core.cle import cross_layer_equalise
    from anneal.core.equalize import equalise
    from anneal.core.transforms import TransformContext, apply_transform

    n_img = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    api, task = coco(), Segmentation()
    ids = task.image_ids(api)
    cal = [task.preprocess(api, i)[0] for i in ids[-16:]]
    src = ROOT / "examples" / "models" / "lraspp_mobilenet_v3_large-fp32.onnx"
    w = ROOT / "scratch" / "lraspp_eq_screen"
    w.mkdir(parents=True, exist_ok=True)
    ctx = TransformContext(workdir=w, calibset=Calib(cal))
    variants = {"plain": None, "eq": {}, "eq grid": {"grid_inverse": True}, "eq mix": {"mix": (0.5, 0.5)},
                "eq se": {"se": True}, "eq residual": {"residual": True}, "cle4": "cle"}
    floats = {}
    for k, kw in variants.items():
        if kw is None:
            floats[k] = src
        elif kw == "cle":
            floats[k] = w / "cle4.onnx"
            cross_layer_equalise(src, floats[k], max_scale=4.0)
        else:
            floats[k] = w / f"{k.replace(' ', '-')}.onnx"
            r = equalise(src, floats[k], cal, **kw)
            print(f"  {k}: {r.summary()['by_kind']}", flush=True)
    models = {}
    for k, p in floats.items():
        models[f"{k} | weights"] = fake_quant_weights(p, w / f"{p.stem}-wq.onnx")
        models[f"{k} | activations"] = apply_transform(
            "quantize_static_int8", {**ACT, "per_channel": True, "calib_samples": 16}, ModelArtifact(path=p), ctx).path
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ref_s = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    sess = {k: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"]) for k, p in models.items()}
    agree = {k: [] for k in sess}
    for i in ids[:n_img]:
        x, _ = task.preprocess(api, i)
        ref = ref_s.run(None, {"input": x})[0].argmax(1)
        for k, s in sess.items():
            agree[k].append(float((s.run(None, {"input": x})[0].argmax(1) == ref).mean()))
    for k in sess:
        print(f"  {k:28s} pixel agreement with FP32 {100 * np.mean(agree[k]):6.2f}%", flush=True)


if __name__ == "__main__":
    main()
