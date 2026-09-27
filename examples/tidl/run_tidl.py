"""Does Anneal's equalisation hold on a second NPU vendor? TI TDA4VM, bit-exact host emulation.

TI's TIDL tools compile an ONNX model for the TDA4VM (J721E) C7x/MMA accelerator and run it in
host emulation on x86, which reproduces the device's integer arithmetic. TDA4VM is the harshest
quantizer tested here: symmetric (no zero point) per-tensor activations with power-of-two scales.

Variants, all scored on Imagenette validation images (1000-way) and paired against FP32:

* ``fp32``                     onnxruntime CPU, the reference
* ``tidl 8-bit``               TIDL's own calibration, as exported
* ``tidl 8-bit + equalised``   the same on Anneal's equalised model (exact in float)
* ``tidl 16-bit``              the vendor's precision fallback
* ``tidl auto mixed``          TIDL's automated mixed precision (mixed_precision_factor 1.2)

Runs in CI on Ubuntu (TIDL tools need x86 Linux): see .github/workflows/tidl-lab.yml.

    python examples/tidl/run_tidl.py --models efficientnet_b0 --images 512 --out result.json
"""

from __future__ import annotations

import argparse
import json
import re
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".anneal_cache"

# TI's reference compile options (runtimes/examples/python/basic_example/config.yaml at the pinned
# commit), with more calibration frames. ti_internal_nc_flag and add_data_convert_ops are needed
# for host emulation: without them 8-bit calibration fails ("TIDL Compute Invoke Failed").
COMMON = {"debug_level": 0, "tensor_bits": 8, "accuracy_level": 1,
          "advanced_options:calibration_frames": 32, "advanced_options:calibration_iterations": 10,
          "advanced_options:mixed_precision_factor": -1, "advanced_options:quantization_scale_type": 0,
          "advanced_options:high_resolution_optimization": 0, "advanced_options:pre_batchnorm_fold": 1,
          "ti_internal_nc_flag": 1601, "advanced_options:activation_clipping": 1,
          "advanced_options:weight_clipping": 1, "advanced_options:bias_calibration": 1,
          "advanced_options:channel_wise_quantization": 0, "advanced_options:add_data_convert_ops": 3,
          "advanced_options:inference_mode": 0, "advanced_options:num_cores": 1}
# TIDL honours advanced_options (clipping, bias calibration, channel-wise weights) only at
# accuracy_level 9; at level 1 (COMMON) it applies its own defaults and ignores them (LRASPP: five
# option settings gave the identical -40.31 mIoU pts).
AL9 = {**COMMON, "accuracy_level": 9}
VARIANTS = {
    "tidl 8-bit al9": ("plain", AL9),
    "tidl 8-bit al9 no clip": ("plain", {**AL9, "advanced_options:weight_clipping": 0, "advanced_options:activation_clipping": 0}),
    "tidl 8-bit al9 ch-wise": ("plain", {**AL9, "advanced_options:channel_wise_quantization": 1}),
    "tidl 8-bit al9 ch-wise no clip": ("plain", {**AL9, "advanced_options:channel_wise_quantization": 1,
                                                "advanced_options:weight_clipping": 0, "advanced_options:activation_clipping": 0}),
    "tidl 8-bit al9 ch-wise + equalised (per-tensor grid)": ("equalised_pt_grid", {**AL9, "advanced_options:channel_wise_quantization": 1}),
    "tidl 8-bit": ("plain", COMMON),
    "tidl 8-bit + equalised": ("equalised", COMMON),
    "tidl 16-bit": ("plain", {**COMMON, "tensor_bits": 16}),
    "tidl auto mixed": ("plain", {**COMMON, "advanced_options:mixed_precision_factor": 1.2}),
    # TI's reference options quantize weights per tensor; channel-wise is TIDL's alternative
    "tidl 8-bit ch-wise": ("plain", {**COMMON, "advanced_options:channel_wise_quantization": 1}),
    "tidl 8-bit ch-wise + equalised": ("equalised", {**COMMON, "advanced_options:channel_wise_quantization": 1}),
    "tidl 8-bit + equalised (residual)": ("equalised_res", COMMON),
    # per-tensor-weight aware (TIDL quantizes weights per tensor): squeeze-excite and residual sites,
    # activation/weight mix t=0.5, then ReLU cross-layer equalisation
    "tidl 8-bit + equalised (per-tensor)": ("equalised_pt", COMMON),
    # + Anneal's analytic bias correction for per-tensor weights (anneal.core.bias_correction); TIDL
    # runs its own iterative bias calibration too, so this measures whether ours adds anything
    "tidl 8-bit + equalised (per-tensor) + bias corr": ("equalised_pt_bc", COMMON),
    # the gate-side 1/s constants exactly on the power-of-two int8 grid (lossless if TIDL quantizes them)
    "tidl 8-bit + equalised (per-tensor grid)": ("equalised_pt_grid", COMMON),
    # + a Clip before every gate so its input's 8 bits cover only the span the gate can see
    "tidl 8-bit + equalised (per-tensor grid clip)": ("equalised_pt_grid_clip", COMMON),
    "tidl auto mixed + equalised (per-tensor grid)": ("equalised_pt_grid", {**COMMON, "advanced_options:mixed_precision_factor": 1.2}),
    # ReLU/ReLU6 cross-layer equalisation (anneal.core.cle), full and with scales capped at 16x,
    # since uncapped CLE can widen per-tensor activation ranges ~650x on MobileNetV2
    "tidl 8-bit + cle": ("cle", COMMON),
    "tidl 8-bit + cle (max scale 16)": ("cle16", COMMON),
    "tidl 8-bit + cle (max scale 4)": ("cle4", COMMON),
    # activation-aware CLE: s_act^0.5 * s_cle^0.5 (best in the local TIDL emulation, +6.5pp on MobileNetV2)
    "tidl 8-bit + cle (activation-aware)": ("cle_t05", COMMON),
    # 16-bit feature maps for the tensors Anneal ranks most damaging at 8 bits (noise injection on
    # calibration images, anneal.core.activation_sensitivity), through TIDL's own 16-bit list
    "tidl 8-bit + anneal 16-bit top4": ("plain", {**COMMON, "_top16": 4}),
    "tidl 8-bit + anneal 16-bit top8": ("plain", {**COMMON, "_top16": 8}),
    "tidl 8-bit + equalised + anneal 16-bit top4": ("equalised", {**COMMON, "_top16": 4}),
    # Anneal's all-8-bit fixes first, then TIDL's own mixed-precision search on what remains
    "tidl auto mixed + equalised (per-tensor)": ("equalised_pt", {**COMMON, "advanced_options:mixed_precision_factor": 1.2}),
    "tidl auto mixed + cle (max scale 4)": ("cle4", {**COMMON, "advanced_options:mixed_precision_factor": 1.2}),
    # Pre-quantized import (docs/quantization.md, "Pre-quantized Models"): TIDL takes an ONNX QDQ
    # model's own scales instead of calibrating. Its QDQ layer table lists per-channel symmetric
    # weights for convolutions, which its own PTQ on TDA4VM does not give (ch-wise: 0% at al9).
    # TDA4VM needs symmetric activations. Scales from Anneal's quantizer (onnxruntime, min-max).
    "tidl prequant qdq (per-tensor)": ("qdq_pt", {**COMMON, "advanced_options:prequantized_model": 1}),
    "tidl prequant qdq (per-channel)": ("qdq_pc", {**COMMON, "advanced_options:prequantized_model": 1}),
    "tidl prequant qdq (per-channel pow2)": ("qdq_pc_pow2", {**COMMON, "advanced_options:prequantized_model": 1}),
    "tidl prequant qdq (per-channel) + equalised": ("qdq_eq_pc", {**COMMON, "advanced_options:prequantized_model": 1}),
}

#: which float model each pre-quantized variant quantizes, and how
QDQ_BUILDS = {"qdq_pt": ("plain", False, False), "qdq_pc": ("plain", True, False),
              "qdq_pc_pow2": ("plain", True, True), "qdq_eq_pc": ("equalised", True, False)}


def session(model: Path, providers: list[str], options: dict | None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 1  # as TI's wrapper
    so.add_session_config_entry("session.disable_input_validation", "1")
    so.add_session_config_entry("session.disable_output_validation", "1")
    provider_options = [options, {}] if options is not None else [{}]
    return ort.InferenceSession(str(model), providers=providers, provider_options=provider_options, sess_options=so)


#: ops TIDL fuses into the layer before them: the fused layer's output tensor is the last one's
FUSED = {"Relu", "Clip", "LeakyRelu", "PRelu", "BatchNormalization", "HardSwish"}


def fused_end(model_path: Path, tensor: str) -> str:
    """The tensor TIDL names a layer by: follow single consumers that it fuses (Conv -> Relu)."""
    import onnx

    g = onnx.load(str(model_path)).graph
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    while len(consumers.get(tensor, [])) == 1 and consumers[tensor][0].op_type in FUSED:
        tensor = consumers[tensor][0].output[0]
    return tensor


def main() -> None:
    import onnx

    from anneal.core.artifact import sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import equalise
    from anneal.models import export_torchvision

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="efficientnet_b0")
    ap.add_argument("--images", type=int, default=512)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    tools = os.environ.get("TIDL_TOOLS_PATH")
    if not tools:
        raise SystemExit("TIDL_TOOLS_PATH is not set: source edgeai-tidl-tools/scripts/setup/setup_env.sh J721E")

    report = {"soc": "J721E (TDA4VM)", "tidl_tools_path": tools, "n": args.images, "models": {}}
    work = Path("tidl-work")
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        mdir = work / name
        mdir.mkdir(parents=True, exist_ok=True)
        src = mdir / f"{name}-fp32.onnx"
        if not src.exists():
            export_torchvision(name, src)
        # TIDL wants static shapes: batch 1, and shape-inferred.
        m = onnx.load(str(src))
        for vi in list(m.graph.input) + list(m.graph.output):
            d = vi.type.tensor_type.shape.dim[0]
            d.ClearField("dim_param")
            d.dim_value = 1
        onnx.save(m, str(src))
        shape = sample_shape(src)
        calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=1, limit=64, sample_shape=shape)
        calib_imgs = list(calib.calibration_batches(64))
        eq = mdir / f"{name}-equalised.onnx"
        equalise(src, eq, calib_imgs)  # the exported graph now has batch 1
        eq_res = mdir / f"{name}-equalised-residual.onnx"
        equalise(src, eq_res, calib_imgs, residual=True)
        from anneal.core.cle import cross_layer_equalise

        eq_pt = mdir / f"{name}-equalised-per-tensor.onnx"
        r = equalise(src, eq_pt, calib_imgs, se=True, mix=(0.5, 0.5))
        cle = cross_layer_equalise(eq_pt, eq_pt)
        print(f"  {name}: per-tensor equalisation {r.summary()['by_kind']}, cle pairs {len(cle.pairs)}", flush=True)
        cle_path, cle16 = mdir / f"{name}-cle.onnx", mdir / f"{name}-cle16.onnx"
        cross_layer_equalise(src, cle_path)
        cross_layer_equalise(src, cle16, max_scale=16.0)
        cle4 = mdir / f"{name}-cle4.onnx"
        cross_layer_equalise(src, cle4, max_scale=4.0)
        cle_t05 = mdir / f"{name}-cle-t05.onnx"
        cross_layer_equalise(src, cle_t05, batches=calib_imgs, t=0.5)
        eq_pt_grid = mdir / f"{name}-equalised-per-tensor-grid.onnx"  # 1/s on the int8 grid
        equalise(src, eq_pt_grid, calib_imgs, se=True, mix=(0.5, 0.5), grid_inverse=True)
        cross_layer_equalise(eq_pt_grid, eq_pt_grid)
        from anneal.core.surrogate import clip_gate_inputs

        eq_pt_grid_clip = mdir / f"{name}-equalised-per-tensor-grid-clip.onnx"  # + a Clip before every gate
        clip_gate_inputs(eq_pt_grid, eq_pt_grid_clip)
        from anneal.core.bias_correction import correct_biases

        eq_pt_bc = mdir / f"{name}-equalised-per-tensor-bc.onnx"  # per-tensor int8 weights, as TIDL
        correct_biases(eq_pt, eq_pt_bc, calib_imgs, per_channel=False)
        models = {"plain": src, "equalised": eq, "equalised_res": eq_res, "equalised_pt": eq_pt,
                  "equalised_pt_bc": eq_pt_bc, "equalised_pt_grid": eq_pt_grid,
                  "equalised_pt_grid_clip": eq_pt_grid_clip,
                  "cle": cle_path, "cle16": cle16, "cle4": cle4, "cle_t05": cle_t05}
        for p in models.values():
            onnx.shape_inference.infer_shapes_path(str(p), str(p))
        needed = {VARIANTS[v.strip()][0] for v in args.variants.split(",") if v.strip() in VARIANTS}
        if needed & set(QDQ_BUILDS):
            from anneal.core.artifact import ModelArtifact
            from anneal.core.transforms import TransformContext, apply_transform

            qctx = TransformContext(workdir=mdir / "qdq", calibset=calib)
            for key in sorted(needed & set(QDQ_BUILDS)):
                base, per_channel, pow2 = QDQ_BUILDS[key]
                params = {"per_channel": per_channel, "activation_type": "int8", "activation_symmetric": True,
                          "pow2_activation_scales": pow2, "calibrate_method": "minmax", "calib_samples": 64,
                          "float_mixed_outputs": False}
                models[key] = apply_transform("quantize_static_int8", params, ModelArtifact(path=models[base]), qctx).path
                q = onnx.load(str(models[key]))
                print(f"  {name}: {key} from {base}: {sum(n.op_type == 'QuantizeLinear' for n in q.graph.node)} Q, "
                      f"{sum(n.op_type == 'DequantizeLinear' for n in q.graph.node)} DQ, opset "
                      f"{[o.version for o in q.opset_import if o.domain in ('', 'ai.onnx')]}", flush=True)
        ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=1, limit=args.images, sample_shape=shape)
        xs, ys = zip(*[(x, int(y[0])) for x, y in ev.batches()])
        ys = np.array(ys)

        def predict(s, label: str = "") -> np.ndarray:
            inp = s.get_inputs()[0].name
            out = []
            for i, x in enumerate(xs):
                y = s.run(None, {inp: x})[0]
                if i < 2:  # enough to tell a harness fault (shape, constant output) from quantization loss
                    f = np.asarray(y, dtype=np.float64).ravel()
                    print(f"    {label or 'fp32'} image {i}: shape {np.shape(y)} dtype {np.asarray(y).dtype} "
                          f"min {f.min():.3g} max {f.max():.3g} top5 {np.argsort(-f)[:5].tolist()}", flush=True)
                out.append(int(np.argmax(y)))
            return np.array(out)

        preds = {"fp32": predict(session(src, ["CPUExecutionProvider"], None))}
        rows, timing, rankings = {}, {}, {}
        for key in sorted(needed & set(QDQ_BUILDS)):  # the same QDQ model in onnxruntime: does TIDL follow it?
            preds[f"onnxruntime {key}"] = predict(session(models[key], ["CPUExecutionProvider"], None), key)
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in wanted if v not in VARIANTS]
        if unknown:  # fail before minutes of export and compilation, not after
            raise SystemExit(f"unknown variants {unknown}; labels must not contain commas. Known: {list(VARIANTS)}")
        for label in wanted:
            which, opts = VARIANTS[label]
            art = mdir / "artifacts" / re.sub(r"[^A-Za-z0-9]+", "_", label.replace("+", "plus")).strip("_")  # TI tools run shell commands on this path
            shutil.rmtree(art, ignore_errors=True)
            art.mkdir(parents=True)
            t = time.time()
            try:
                opts = dict(opts)
                k16 = opts.pop("_top16", None)
                if k16:
                    from anneal.core.activation_sensitivity import rank_activation_tensors

                    if which not in rankings:
                        rankings[which] = rank_activation_tensors(models[which], calib_imgs[:32], calib_imgs[32:])
                    top = [fused_end(models[which], r.tensor) for r in rankings[which][:k16]]
                    opts["advanced_options:output_feature_16bit_names_list"] = ",".join(top)
                    timing.setdefault(label, {})["int16_tensors"] = top
                    print(f"  {name} {label}: 16-bit {top}", flush=True)
                comp = session(models[which], ["TIDLCompilationProvider", "CPUExecutionProvider"],
                               {**opts, "artifacts_folder": str(art), "tidl_tools_path": tools})
                inp = comp.get_inputs()[0].name
                for x in calib_imgs[: opts["advanced_options:calibration_frames"]]:
                    comp.run(None, {inp: x})
                del comp
                if k16:  # did TIDL keep these names as layers? (a fused-away name is ignored silently)
                    known = " ".join(f.read_text(errors="replace") for f in art.rglob("*layer_info*.txt"))
                    found = [n for n in top if n in known]
                    timing[label]["int16_names_in_layer_info"] = found
                    print(f"  {name} {label}: {len(found)}/{len(top)} 16-bit names found in TIDL layer info "
                          f"({len(known)} chars; files {[f.name for f in art.rglob('*') if f.is_file()][:12]})", flush=True)
                timing.setdefault(label, {})["compile_s"] = time.time() - t
                t = time.time()
                preds[label] = predict(session(models[which], ["TIDLExecutionProvider", "CPUExecutionProvider"],
                                               {"artifacts_folder": str(art), "debug_level": 0}), label)
                timing[label]["infer_s"] = time.time() - t
            except Exception as exc:  # a variant the toolchain cannot compile is a result too
                rows[label] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                print(f"  {name} {label}: FAILED {rows[label]['error']}", flush=True)
                continue
            print(f"  {name} {label}: done ({timing[label]})", flush=True)

        ref = preds["fp32"] == ys
        out = {"fp32": {"accuracy": float(ref.mean())}, "fp32_correct": "".join("1" if r else "0" for r in ref)}
        for label, p in preds.items():
            if label == "fp32":
                continue
            right = p == ys
            b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right))
            d, lo, hi = paired_delta_ci(b, c, len(ys))
            out[label] = {"accuracy": float(right.mean()), "delta_pp": d, "ci95_pp": [lo, hi],
                          "mcnemar_p": mcnemar_exact(b, c), "agreement": float(np.mean(p == preds["fp32"])),
                          "correct": "".join("1" if r else "0" for r in right), **timing.get(label, {})}
        out.update(rows)
        report["models"][name] = out
        print(name, json.dumps(out, indent=1), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
