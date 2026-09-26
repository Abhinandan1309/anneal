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
VARIANTS = {
    "tidl 8-bit": ("plain", COMMON),
    "tidl 8-bit + equalised": ("equalised", COMMON),
    "tidl 16-bit": ("plain", {**COMMON, "tensor_bits": 16}),
    "tidl auto mixed": ("plain", {**COMMON, "advanced_options:mixed_precision_factor": 1.2}),
    # TI's reference options quantize weights per tensor; channel-wise is TIDL's alternative
    "tidl 8-bit ch-wise": ("plain", {**COMMON, "advanced_options:channel_wise_quantization": 1}),
    "tidl 8-bit ch-wise + equalised": ("equalised", {**COMMON, "advanced_options:channel_wise_quantization": 1}),
    "tidl 8-bit + equalised (residual)": ("equalised_res", COMMON),
}


def session(model: Path, providers: list[str], options: dict | None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 1  # as TI's wrapper
    so.add_session_config_entry("session.disable_input_validation", "1")
    so.add_session_config_entry("session.disable_output_validation", "1")
    provider_options = [options, {}] if options is not None else [{}]
    return ort.InferenceSession(str(model), providers=providers, provider_options=provider_options, sess_options=so)


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
        models = {"plain": src, "equalised": eq, "equalised_res": eq_res}
        for p in models.values():
            onnx.shape_inference.infer_shapes_path(str(p), str(p))
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
        rows, timing = {}, {}
        for label in [v.strip() for v in args.variants.split(",") if v.strip()]:
            which, opts = VARIANTS[label]
            art = mdir / "artifacts" / label.replace(" ", "_").replace("+", "plus")
            shutil.rmtree(art, ignore_errors=True)
            art.mkdir(parents=True)
            t = time.time()
            try:
                comp = session(models[which], ["TIDLCompilationProvider", "CPUExecutionProvider"],
                               {**opts, "artifacts_folder": str(art), "tidl_tools_path": tools})
                inp = comp.get_inputs()[0].name
                for x in calib_imgs[: opts["advanced_options:calibration_frames"]]:
                    comp.run(None, {inp: x})
                del comp
                timing[label] = {"compile_s": time.time() - t}
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
