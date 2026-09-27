"""Where does LRASPP break under TIDL's 8-bit rules? Keep one part float at a time, locally.

Real TIDL (examples/tidl/results/lraspp_n300.json): 8-bit -45.2 mIoU points, 16-bit +0.4,
Anneal's equalisation -25.8, the per-tensor recipe -50.8. The same model's backbone as a
classifier (MobileNetV3-Large) reaches -4.1 on TIDL with the per-tensor recipe, so the backbone
alone does not explain it. This bisects in the TIDL-faithful emulation (per-tensor weights,
symmetric power-of-two int8 feature maps), after checking the emulation reproduces the collapse.

    python lraspp_bisect.py --images 150
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
sys.path.insert(0, str(ROOT / "examples" / "tasks"))

TIDL_LIKE = {"per_channel": False, "activation_type": "int8", "activation_symmetric": True,
             "pow2_activation_scales": True, "calibrate_method": "minmax", "float_mixed_outputs": False}


def groups(model_path: Path) -> dict[str, list[str]]:
    import onnx

    nodes = onnx.load(str(model_path)).graph.node
    name = lambda n: n.name  # noqa: E731
    bb = lambda n: n.name.startswith("/net/backbone/")  # noqa: E731
    stage = lambda n: int(n.name.split("/net/backbone/backbone.")[1].split("/")[0])  # noqa: E731
    se = ("avgpool", "fc1", "fc2", "scale_activation", "block.2/activation", "block.2/Mul")
    return {
        "head": [name(n) for n in nodes if not bb(n)],
        "backbone": [name(n) for n in nodes if bb(n)],
        "backbone 0-6": [name(n) for n in nodes if bb(n) and stage(n) <= 6],
        "backbone 7-16": [name(n) for n in nodes if bb(n) and stage(n) >= 7],
        "SE branches": [name(n) for n in nodes if bb(n) and any(k in n.name for k in se)],
        "head sigmoid branch": [name(n) for n in nodes if not bb(n) and any(
            k in n.name for k in ("cbr", "scale", "Sigmoid", "Mul", "avgpool", "GlobalAveragePool"))],
        "resize + output": [name(n) for n in nodes if n.op_type in ("Resize", "Shape", "Slice", "Concat", "Constant")
                            or (not bb(n) and n.op_type == "Add")],
    }


def main() -> None:
    import onnxruntime as ort
    from run_tasks import Calib, Segmentation, coco, miou

    from anneal.core.artifact import ModelArtifact
    from anneal.core.transforms import TransformContext, apply_transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=150)
    ap.add_argument("--calib-images", type=int, default=16)
    ap.add_argument("--variants", default="plain,head,backbone,backbone 0-6,backbone 7-16,SE branches,"
                                          "head sigmoid branch,resize + output")
    ap.add_argument("--src", default=str(ROOT / "examples" / "models" / "lraspp_mobilenet_v3_large-fp32.onnx"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    api = coco()
    task = Segmentation()
    ids_all = task.image_ids(api)
    ids, calib_ids = ids_all[:args.images], ids_all[-args.calib_images:]
    src = Path(args.src)
    work = ROOT / "scratch" / "lraspp_bisect" / (args.tag or src.stem)
    work.mkdir(parents=True, exist_ok=True)
    ctx = TransformContext(workdir=work, calibset=Calib([task.preprocess(api, i)[0] for i in calib_ids]))
    g = groups(src)
    for k, v in g.items():
        print(f"  group {k}: {len(v)} nodes", flush=True)
    built = {}
    for label in [v.strip() for v in args.variants.split(",") if v.strip()]:
        params = {**TIDL_LIKE, "calib_samples": args.calib_images}
        if label != "plain":
            params["float_nodes"] = g[label]
        built[label] = str(apply_transform("quantize_static_int8", params, ModelArtifact(path=src), ctx).path)
        print(f"  built {label}", flush=True)
        gc.collect()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 4
    sessions = {"fp32": ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]),
                **{k: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for k, p in built.items()}}
    inp = sessions["fp32"].get_inputs()[0].name
    conf = {k: np.zeros((21, 21), np.int64) for k in sessions}
    for n, i in enumerate(ids, 1):
        x, meta = task.preprocess(api, i)
        for k, s in sessions.items():
            conf[k] += task.per_image(s.run(None, {inp: x}), meta, i)
        if n % 50 == 0:
            print(f"  {n}/{len(ids)}", flush=True)
    classes = np.flatnonzero(conf["fp32"].sum(1))
    base = miou(conf["fp32"], classes)
    rows = {"fp32": base}
    print(f"fp32 mIoU {100 * base:.1f}", flush=True)
    for k in built:
        m = miou(conf[k], classes)
        rows[k] = m
        print(f"  float {k:22s} {100 * (m - base):+7.1f} pts  (mIoU {100 * m:.1f})", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"lraspp_bisect_{args.tag or 'plain'}_n{len(ids)}.json").write_text(
        json.dumps({"n": len(ids), "rules": TIDL_LIKE, "miou": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
