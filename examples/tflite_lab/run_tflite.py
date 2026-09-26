"""Does the INT8 failure, and Anneal's fix, survive a different runtime? TFLite / XNNPACK.

Everything else in this repository runs on onnxruntime. TFLite is the most widely deployed
edge runtime (Android, Raspberry Pi, microcontrollers) and quantizes the same way in
principle -- per-channel weights, one scale per activation tensor -- through different code.
If EfficientNet collapses under TFLite's standard full-integer post-training quantization
too, the failure belongs to the method, not to onnxruntime; if Anneal's equalisation fixes it
there too, the fix is runtime-independent.

Only equalisation is carried over. Percentile calibration and a float stem are onnxruntime
knobs; TFLite's converter calibrates by min/max and quantizes every op.

    python run_tflite.py convert  --models efficientnet_b0 --out models/     (needs tensorflow, onnx2tf, torch)
    python run_tflite.py evaluate --models efficientnet_b0 --models-dir models/ --out result.json

Per model, three .tflite files: ``fp32`` (conversion fidelity), ``int8`` (the model as
exported) and ``int8-equalised`` (after anneal.core.equalize).
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".anneal_cache"


def onnx_path(name: str) -> Path:
    path = ROOT / "examples" / "models" / f"{name}-fp32.onnx"
    if not path.exists():
        from anneal.models import export_torchvision

        export_torchvision(name, path)
    return path


def nhwc(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(x, (0, 2, 3, 1)))


def static_batch(path: Path, work: Path) -> Path:
    """A batch-1, simplified copy. The dynamic batch axis otherwise leaves shape arithmetic
    (Shape -> ConstantOfShape) that becomes TFLite FILL ops, which INT8 cannot quantize."""
    import onnx
    from onnxsim import simplify

    model = onnx.load(str(path))
    dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    dim.ClearField("dim_param")
    dim.dim_value = 1
    for out in model.graph.output:
        d = out.type.tensor_type.shape.dim
        if len(d):
            d[0].ClearField("dim_param")
            d[0].dim_value = 1
    simplified, ok = simplify(model)
    if not ok:
        raise RuntimeError(f"onnxsim could not validate the simplified {path.name}")
    dst = work / f"{path.stem}-b1.onnx"
    onnx.save(simplified, str(dst))
    return dst


def quantize_saved_model(saved_model: Path, calib_nhwc: np.ndarray) -> bytes:
    """TFLite full-integer post-training quantization with float input and output."""
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([x[None]] for x in calib_nhwc)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return conv.convert()


def convert(models: list[str], out: Path) -> None:
    from anneal.core.dataset import load_calibset
    from anneal.core.equalize import equalise

    out.mkdir(parents=True, exist_ok=True)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64)
    calib_images = np.concatenate(list(calib.calibration_batches(64)))[:64]
    calib_nhwc = nhwc(calib_images).astype(np.float32)  # already normalised, as the model expects
    for name in models:
        src = onnx_path(name)
        work = out / "work" / name
        work.mkdir(parents=True, exist_ok=True)
        eq = work / f"{name}-equalised.onnx"
        result = equalise(src, eq, [calib_images[i:i + 8] for i in range(0, 64, 8)])
        print(f"{name}: {len(result.sites)} equalised sites", flush=True)
        for variant, onnx_file in (("", static_batch(src, work)), ("-equalised", static_batch(eq, work))):
            tf_dir = work / f"tf{variant}"
            if tf_dir.exists():
                shutil.rmtree(tf_dir)
            # onnx2tf only rewrites NCHW to NHWC (-b 1: static shapes) into a float SavedModel
            # (-osd). Its own quantizer (-oiqt) loads a downloaded, pickled sample file even when
            # calibration data is given, which this repository refuses to unpickle; TensorFlow's
            # converter quantizes instead, as most TFLite users do.
            subprocess.run([sys.executable, "-m", "onnx2tf", "-i", str(onnx_file), "-o", str(tf_dir),
                            "-b", "1", "-n", "-osd"], check=True)
            if variant == "":
                shutil.copy(next(tf_dir.glob("*_float32.tflite")), out / f"{name}-fp32.tflite")
            # INT8 weights (per channel) and activations (per tensor), float input/output.
            (out / f"{name}-int8{variant}.tflite").write_bytes(quantize_saved_model(tf_dir, calib_nhwc))
            print(f"  wrote {name}-int8{variant}.tflite", flush=True)


def interpreter(path: Path, threads: int = 1):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    return it


def run(it, x: np.ndarray) -> np.ndarray:
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(inp["index"], x.astype(inp["dtype"]))
    it.invoke()
    return it.get_tensor(out["index"])


def evaluate(models: list[str], models_dir: Path, limit: int, out: Path) -> None:
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_evalset
    from anneal.core.environment import cpu_features

    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=1, limit=limit)
    report = {"cpu": cpu_features(), "machine": platform.machine(), "n": len(ev), "models": {}}
    for name in models:
        variants = {v: models_dir / f"{name}-{v}.tflite" for v in ("fp32", "int8", "int8-equalised")}
        its = {v: interpreter(p) for v, p in variants.items() if p.exists()}
        preds = {v: [] for v in its}
        labels = []
        for x, y in ev.batches():
            for v, it in its.items():
                preds[v].append(int(np.argmax(run(it, nhwc(x)))))
            labels.append(int(y[0]))
        y = np.array(labels)
        ref = np.array(preds["fp32"]) == y
        rows = {"fp32": {"accuracy": float(ref.mean())}}
        for v in its:
            if v == "fp32":
                continue
            right = np.array(preds[v]) == y
            b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right))
            d, lo, hi = paired_delta_ci(b, c, len(y))
            rows[v] = {"accuracy": float(right.mean()), "delta_pp": d, "ci95_pp": [lo, hi],
                       "mcnemar_p": mcnemar_exact(b, c),
                       "agreement": float(np.mean(np.array(preds[v]) == np.array(preds["fp32"])))}
        # Single-thread latency, 30 runs after 5 warm-ups, on one fixed image.
        x0 = nhwc(next(iter(ev.batches()))[0])
        for v, it in its.items():
            for _ in range(5):
                run(it, x0)
            times = []
            for _ in range(30):
                t = time.perf_counter()
                run(it, x0)
                times.append((time.perf_counter() - t) * 1000)
            rows[v]["p50_ms"] = float(np.median(times))
        report["models"][name] = rows
        print(name, json.dumps(rows), flush=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["convert", "evaluate"])
    ap.add_argument("--models", default="efficientnet_b0,mobilenet_v3_large,mobilenet_v2,resnet50")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-dir", default="tflite-models")
    ap.add_argument("--limit", type=int, default=1024)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.command == "convert":
        convert(models, Path(args.out))
    else:
        evaluate(models, Path(args.models_dir), args.limit, Path(args.out))


if __name__ == "__main__":
    main()
