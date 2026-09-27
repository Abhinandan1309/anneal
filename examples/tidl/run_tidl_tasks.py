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
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "tasks"))
sys.path.insert(0, str(ROOT / "examples" / "tidl"))

from run_tidl import AL9, COMMON, session  # noqa: E402

VARIANTS = {
    "tidl 8-bit al9": ("plain", AL9),
    "tidl 8-bit al9 no clip": ("plain", {**AL9, "advanced_options:weight_clipping": 0, "advanced_options:activation_clipping": 0}),
    "tidl 8-bit al9 ch-wise": ("plain", {**AL9, "advanced_options:channel_wise_quantization": 1}),
    "tidl 8-bit al9 ch-wise no clip": ("plain", {**AL9, "advanced_options:channel_wise_quantization": 1,
                                                "advanced_options:weight_clipping": 0, "advanced_options:activation_clipping": 0}),
    "tidl 16-bit": ("plain", {**COMMON, "tensor_bits": 16}),
    "tidl 8-bit": ("plain", COMMON),
    "tidl 8-bit + equalised": ("equalised", COMMON),
    # U-Net: the skip concatenations share one scale on TIDL (anneal.core.equalize_concat), and
    # conv-ReLU pairs meet per-tensor weights (anneal.core.cle)
    "tidl 8-bit + concat eq": ("concat_eq", COMMON),
    "tidl 8-bit + concat eq + cle": ("concat_eq_cle", COMMON),
    # per-tensor-weight aware: gated, squeeze-excite and residual sites with the activation/weight
    # mix t=0.5, then ReLU/ReLU6 cross-layer equalisation
    "tidl 8-bit + equalised (per-tensor)": ("equalised_pt", COMMON),
    # the same plus gated dense sites into k x k convs (YOLOv8's Conv -> SiLU -> 3x3 Conv)
    "tidl 8-bit + equalised (per-tensor + dense kxk)": ("equalised_pt_dense", COMMON),
    # the per-tensor variant with CLE capped at 4x (MobileNetV2 on TIDL: uncapped -12.9pp, 4x -1.2pp) and without CLE
    "tidl 8-bit + equalised (per-tensor cle4)": ("equalised_pt_cle4", COMMON),
    "tidl 8-bit + equalised (per-tensor no cle)": ("equalised_pt_nocle", COMMON),
    # isolate LRASPP's regression: squeeze-excite sites without the mix, and the mix without them
    "tidl 8-bit + equalised (se only)": ("eq_se", COMMON),
    "tidl 8-bit + equalised (mix only)": ("eq_mix", COMMON),
    # no weight mix (it collapsed LRASPP in emulation); gate-side 1/s on the int8 grid, gate inputs
    # clipped to the span the gate sees
    "tidl 8-bit + equalised (grid clip)": ("eq_grid_clip", COMMON),
    "tidl 8-bit + equalised (se grid clip)": ("eq_se_grid_clip", COMMON),
    # which TIDL option breaks LRASPP's early layers (traces: block 1 falls to 3 dB)? its percentile
    # weight/activation clipping on a 16-channel depthwise layer is the suspect
    "tidl 8-bit no weight clip": ("plain", {**COMMON, "advanced_options:weight_clipping": 0}),
    "tidl 8-bit no act clip": ("plain", {**COMMON, "advanced_options:activation_clipping": 0}),
    "tidl 8-bit no clip": ("plain", {**COMMON, "advanced_options:weight_clipping": 0,
                                     "advanced_options:activation_clipping": 0}),
    "tidl 8-bit ch-wise": ("plain", {**COMMON, "advanced_options:channel_wise_quantization": 1}),
    # YOLOv8n: each output concatenates three detection scales, which TIDL gives one shared scale.
    # Exact concat equalisation, or the two head concats kept on the ARM (each scale then keeps its
    # own scale), alone and with TIDL's mixed precision
    "tidl auto mixed + concat eq": ("concat_eq", {**COMMON, "advanced_options:mixed_precision_factor": 1.2}),
    "tidl 8-bit + head concat on arm": ("plain", {**COMMON, "deny_list:layer_name": "/model.22/Concat,/model.22/Concat_1"}),
    "tidl auto mixed + head concat on arm": ("plain", {**COMMON, "advanced_options:mixed_precision_factor": 1.2,
                                                       "deny_list:layer_name": "/model.22/Concat,/model.22/Concat_1"}),
    # TIDL's own mixed-precision search, alone and on top of Anneal's equalisation
    "tidl auto mixed": ("plain", {**COMMON, "advanced_options:mixed_precision_factor": 1.2}),
    "tidl auto mixed + equalised": ("equalised", {**COMMON, "advanced_options:mixed_precision_factor": 1.2}),
    # Explicit 16-bit feature maps (TIDL's output_feature_16bit_names_list; their weights follow).
    # Real LRASPP traces (run 36301780043): the worst 8-bit tensors are the linear bottlenecks
    # (projection convs with no activation, and residual Adds: 1.6-3.8 dB), and the damage starts
    # at backbone.1's projection (11 -> 2.8 dB). See select_16bit for the rules.
    "tidl 8-bit + 16-bit linear": ("plain", {**COMMON, "_16bit": ["linear"]}),
    "tidl 8-bit + 16-bit backbone 0-3": ("plain", {**COMMON, "_16bit": [r"re:/backbone\.[0-3]/"]}),
    "tidl 8-bit + 16-bit backbone 0-6": ("plain", {**COMMON, "_16bit": [r"re:/backbone\.[0-6]/"]}),
    "tidl 8-bit + 16-bit linear + backbone 0-3": ("plain", {**COMMON, "_16bit": ["linear", r"re:/backbone\.[0-3]/"]}),
    "tidl 8-bit + equalised + 16-bit linear": ("equalised", {**COMMON, "_16bit": ["linear"]}),
    # Run 36316677635: backbone 0-3 in 16 bits (11 layers) -44.7 -> -6.0; 0-6 (31) no better;
    # any 'linear' set -50.8 (identical for three sets: degenerate). Narrow it, and combine.
    "tidl 8-bit + 16-bit backbone 0-1": ("plain", {**COMMON, "_16bit": [r"re:/backbone\.[01]/"]}),
    "tidl 8-bit + 16-bit backbone 2-3": ("plain", {**COMMON, "_16bit": [r"re:/backbone\.[23]/"]}),
    "tidl 8-bit + 16-bit backbone 1": ("plain", {**COMMON, "_16bit": [r"re:/backbone\.1/"]}),
    "tidl 8-bit + equalised + 16-bit backbone 0-3": ("equalised", {**COMMON, "_16bit": [r"re:/backbone\.[0-3]/"]}),
    "tidl 8-bit + equalised + 16-bit backbone 0-1": ("equalised", {**COMMON, "_16bit": [r"re:/backbone\.[01]/"]}),
}


def select_16bit(model_path: Path, rules: list[str]) -> list[str]:
    """Layer names (as TIDL names fused layers) for its 16-bit list, from rules.

    ``linear``: every Conv whose output is not activated (projection convs, the classifiers) and
    every Add; ``re:<regex>``: every Conv/Add/Mul output whose name matches.
    """
    import onnx
    from run_tidl import FUSED, fused_end

    g = onnx.load(str(model_path)).graph
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    picked: list[str] = []
    for n in g.node:
        if n.op_type not in ("Conv", "Add", "Mul"):
            continue
        out = n.output[0]
        for rule in rules:
            if rule == "linear":
                cs = consumers.get(out, [])
                hit = n.op_type == "Add" or (n.op_type == "Conv" and not any(c.op_type in FUSED for c in cs))
            elif rule.startswith("re:"):
                hit = re.search(rule[3:], out) is not None
            else:
                raise ValueError(f"unknown 16-bit rule {rule!r}")
            if hit:
                name = fused_end(model_path, out)
                if name not in picked:
                    picked.append(name)
                break
    return picked


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


def trace_layers(label: str, model_path, art, task, api, img_id: int, inp: str) -> None:
    """Run one image with TIDL's layer traces on and keep everything needed to diff it offline."""
    import glob
    import os

    out = Path("tidl-traces") / re.sub(r"[^A-Za-z0-9]+", "_", label.replace("+", "plus")).strip("_")
    out.mkdir(parents=True, exist_ok=True)
    roots = ["/tmp", os.getcwd(), str(art)]
    for r in roots:  # traces of an earlier variant have the same names: remove them first
        for f in glob.glob(os.path.join(r, "**", "tidl_trace_*"), recursive=True):
            if "tidl-traces" not in f:
                os.remove(f)
    before = {f for r in roots for f in glob.glob(os.path.join(r, "**", "*"), recursive=True)}
    x, _ = task.preprocess(api, img_id)
    np.save(out / "input.npy", x)
    shutil.copy(model_path, out / "model_float.onnx")
    s = session(model_path, ["TIDLExecutionProvider", "CPUExecutionProvider"],
                {"artifacts_folder": str(art), "debug_level": 3})
    y = s.run(None, {inp: x})
    np.save(out / "tidl_output.npy", y[0])
    del s
    new = [f for r in roots for f in glob.glob(os.path.join(r, "**", "*"), recursive=True)
           if f not in before and os.path.isfile(f) and "tidl-traces" not in f]
    keep = [f for f in new if "trace" in os.path.basename(f).lower() or f.endswith((".y", ".bin"))]
    for f in keep[:4000]:
        shutil.copy(f, out / os.path.basename(f))
    for f in glob.glob(os.path.join(str(art), "**", "*"), recursive=True):
        if os.path.isfile(f) and (f.endswith(".txt") or "layer_info" in f or f.endswith(".svg")):
            shutil.copy(f, out / ("art_" + os.path.basename(f)))
    print(f"  {label}: traced {len(keep)} files ({len(new)} new in total) -> {out}", flush=True)
    print("    examples:", [os.path.basename(f) for f in keep[:6]], flush=True)


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
    ap.add_argument("--trace", action="store_true",
                    help="also run one image with TIDL layer traces (debug_level 3) and keep them, the "
                         "layer info, the float model and the input under tidl-traces/ for a per-layer diff")
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
    eq_pt = work / f"{args.model}-equalised-per-tensor.onnx"
    r_pt = equalise(src, eq_pt, calib, residual=True, se=True, mix=(0.5, 0.5))
    c_pt = cross_layer_equalise(eq_pt, eq_pt)
    print(f"{args.model}: per-tensor equalisation {r_pt.summary()['by_kind']}, cle pairs {len(c_pt.pairs)}", flush=True)
    eq_ptd = work / f"{args.model}-equalised-per-tensor-dense.onnx"
    equalise(src, eq_ptd, calib, residual=True, se=True, mix=(0.5, 0.5))
    d_pt, _, _ = equalise_dense(eq_ptd, eq_ptd, calib, mix=(0.5, 0.5), any_kernel=True)
    cross_layer_equalise(eq_ptd, eq_ptd)
    print(f"{args.model}: per-tensor dense kxk sites {len(d_pt)}", flush=True)
    eq_se, eq_mix = work / f"{args.model}-eq-se.onnx", work / f"{args.model}-eq-mix.onnx"
    equalise(src, eq_se, calib, se=True)
    equalise(src, eq_mix, calib, mix=(0.5, 0.5))
    eq_nocle, eq_cle4 = work / f"{args.model}-pt-nocle.onnx", work / f"{args.model}-pt-cle4.onnx"
    equalise(src, eq_nocle, calib, residual=True, se=True, mix=(0.5, 0.5))
    cross_layer_equalise(eq_nocle, eq_cle4, max_scale=4.0)
    from anneal.core.surrogate import clip_gate_inputs

    eq_gc, eq_se_gc = work / f"{args.model}-eq-grid-clip.onnx", work / f"{args.model}-eq-se-grid-clip.onnx"
    equalise(src, eq_gc, calib, grid_inverse=True)
    clip_gate_inputs(eq_gc, eq_gc)
    equalise(src, eq_se_gc, calib, se=True, grid_inverse=True)
    clip_gate_inputs(eq_se_gc, eq_se_gc)
    models = {"plain": src, "equalised": eq, "eq_grid_clip": eq_gc, "eq_se_grid_clip": eq_se_gc, "concat_eq": cat_eq, "concat_eq_cle": cat_eq_cle, "equalised_pt": eq_pt,
              "equalised_pt_dense": eq_ptd, "equalised_pt_cle4": eq_cle4, "equalised_pt_nocle": eq_nocle,
              "eq_se": eq_se, "eq_mix": eq_mix}
    for p in models.values():
        onnx.shape_inference.infer_shapes_path(str(p), str(p))

    fp = session(src, ["CPUExecutionProvider"], None)
    inp = fp.get_inputs()[0].name
    sessions = {"fp32": fp}
    rows, timing = {}, {}
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in wanted if v not in VARIANTS]
    if unknown:  # fail before minutes of export and compilation, not after
        raise SystemExit(f"unknown variants {unknown}; labels must not contain commas. Known: {list(VARIANTS)}")
    for label in wanted:
        which, opts = VARIANTS[label]
        opts = {**opts, "advanced_options:calibration_frames": len(calib)}
        rules16 = opts.pop("_16bit", None)
        names16: list[str] = []
        if rules16:
            names16 = select_16bit(models[which], rules16)
            opts["advanced_options:output_feature_16bit_names_list"] = ",".join(names16)
            print(f"  {label}: {len(names16)} layers in 16 bits", flush=True)
        art = work / "artifacts" / re.sub(r"[^A-Za-z0-9]+", "_", label.replace("+", "plus")).strip("_")  # TI tools run shell commands on this path
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
            if names16:  # a name TIDL fused away is ignored silently: record which ones it kept
                known = " ".join(f.read_text(errors="replace") for f in art.rglob("*layer_info*.txt"))
                timing[label]["int16_layers"] = names16
                timing[label]["int16_in_layer_info"] = sum(n in known for n in names16)
                print(f"  {label}: {timing[label]['int16_in_layer_info']}/{len(names16)} 16-bit names in TIDL layer info",
                      flush=True)
            if args.trace:
                trace_layers(label, models[which], art, task, api, ids[0], inp)
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
