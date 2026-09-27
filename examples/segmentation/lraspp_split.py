"""LRASPP under TIDL-like rules: is the loss in the per-tensor weights or in the activations?

* ``weights only``        per-tensor int8 weights (fake-quantized in the float model), float activations
* ``weights only + bc``   the same after analytic bias correction (anneal.core.bias_correction)
* ``weights only + eq``   per-tensor recipe (equalisation, SE sites, mix 0.5, CLE cap 4) first
* ``activations only``    TIDL-like activations (symmetric power-of-two int8) with per-channel weights
* ``both``                everything TIDL-like

    python lraspp_split.py --images 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples" / "tasks"))
ACT = {"activation_type": "int8", "activation_symmetric": True, "pow2_activation_scales": True,
       "calibrate_method": "minmax", "float_mixed_outputs": False}


def fake_quant_weights(src: Path, dst: Path) -> Path:
    import onnx
    from onnx import numpy_helper

    from anneal.core.bias_correction import quantize_weights

    m = onnx.load(str(src))
    inits = {i.name: i for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type in ("Conv", "Gemm") and len(n.input) > 1 and n.input[1] in inits:
            w = numpy_helper.to_array(inits[n.input[1]])
            inits[n.input[1]].CopyFrom(numpy_helper.from_array(quantize_weights(w, False).astype(np.float32), n.input[1]))
    onnx.save(m, str(dst))
    return dst


def main() -> None:
    import onnxruntime as ort
    from run_tasks import Calib, Segmentation, coco, miou

    from anneal.core.artifact import ModelArtifact
    from anneal.core.bias_correction import correct_biases
    from anneal.core.cle import cross_layer_equalise
    from anneal.core.equalize import equalise
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=150)
    ap.add_argument("--calib-images", type=int, default=16)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    api, task = coco(), Segmentation()
    ids_all = task.image_ids(api)
    ids, calib_ids = ids_all[:args.images], ids_all[-args.calib_images:]
    src = ROOT / "examples" / "models" / "lraspp_mobilenet_v3_large-fp32.onnx"
    work = ROOT / "scratch" / "lraspp_split"
    work.mkdir(parents=True, exist_ok=True)
    calib_x = [task.preprocess(api, i)[0] for i in calib_ids]
    ctx = TransformContext(workdir=work, calibset=Calib(calib_x))
    eq = work / "eq-pt.onnx"
    equalise(src, eq, calib_x, residual=True, se=True, mix=(0.5, 0.5))
    cross_layer_equalise(eq, eq, max_scale=4.0)
    correct_biases(src, work / "bc.onnx", calib_x, per_channel=False)
    correct_biases(eq, work / "eq-bc.onnx", calib_x, per_channel=False)
    models = {
        "weights only": fake_quant_weights(src, work / "wq.onnx"),
        "weights only + bc": fake_quant_weights(work / "bc.onnx", work / "wq-bc.onnx"),
        "weights only + eq": fake_quant_weights(eq, work / "wq-eq.onnx"),
        "weights only + eq + bc": fake_quant_weights(work / "eq-bc.onnx", work / "wq-eq-bc.onnx"),
        "activations only": apply_transform("quantize_static_int8", {**ACT, "per_channel": True, "calib_samples": args.calib_images},
                                            ModelArtifact(path=src), ctx).path,
        "activations only + eq": apply_transform("quantize_static_int8", {**ACT, "per_channel": True, "calib_samples": args.calib_images},
                                                 ModelArtifact(path=eq), ctx).path,
        "both": apply_transform("quantize_static_int8", {**ACT, "per_channel": False, "calib_samples": args.calib_images},
                                ModelArtifact(path=src), ctx).path,
    }
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 4
    sess = {"fp32": ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]),
            **{k: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"]) for k, p in models.items()}}
    inp = sess["fp32"].get_inputs()[0].name
    conf = {k: np.zeros((21, 21), np.int64) for k in sess}
    for i in ids:
        x, meta = task.preprocess(api, i)
        for k, s in sess.items():
            conf[k] += task.per_image(s.run(None, {inp: x}), meta, i)
    cls = np.flatnonzero(conf["fp32"].sum(1))
    base = miou(conf["fp32"], cls)
    print(f"fp32 mIoU {100 * base:.1f}", flush=True)
    for k in models:
        print(f"  {k:26s} {100 * (miou(conf[k], cls) - base):+7.1f} pts", flush=True)


if __name__ == "__main__":
    main()
