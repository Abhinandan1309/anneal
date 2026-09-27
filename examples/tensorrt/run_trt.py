"""Does Anneal's equalisation hold under NVIDIA TensorRT (Jetson's runtime)? Accuracy and latency.

TensorRT INT8 quantizes activations per tensor, symmetrically; weights per channel. Two INT8 paths
are scored, as users run them:

* ``trt int8``               implicit quantization with TensorRT's entropy calibrator (trtexec --int8
                             with a calibrator; deprecated in TensorRT 10 but still the common path)
* ``modelopt int8``          explicit QDQ from NVIDIA ModelOpt, NVIDIA's recommended path

each on the exported model and on Anneal's equalised one (``+ eq``; ``+ eq res`` also rewrites
residual sites), against ``trt fp16``, the vendor fallback. Latency is the median batch-1 GPU time.

Needs an NVIDIA GPU (TensorRT 10.x: 11 removed implicit INT8; torch with CUDA). Runs as a Kaggle kernel: see
examples/tensorrt/kaggle/. Scored on Imagenette validation images (1000-way), paired against FP32.

    python examples/tensorrt/run_trt.py --models efficientnet_b0 --images 1000 --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".anneal_cache"
VARIANTS = ["trt fp16", "trt int8", "trt int8 + eq", "trt int8 + eq res",
            "modelopt int8", "modelopt int8 + eq", "modelopt int8 + eq res"]
TIMM = {"efficientvit_b0": "efficientvit_b0.r224_in1k"}


def export(name: str, dst: Path) -> None:
    import torch

    if name in TIMM:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "zoo_gated"))
        import timm
        from export_models import Renormalise

        net = timm.create_model(TIMM[name], pretrained=True).eval()
        cfg = net.pretrained_cfg
        model, size = Renormalise(net, cfg["mean"], cfg["std"]).eval(), cfg["input_size"][-1]
        with torch.no_grad():
            torch.onnx.export(model, torch.randn(1, 3, size, size), str(dst), input_names=["input"],
                              output_names=["logits"], opset_version=17, dynamo=False)
    else:
        from anneal.models import export_torchvision

        export_torchvision(name, dst)


class Calibrator:
    """TensorRT entropy calibrator over a list of (1, C, H, W) float32 batches."""

    def __new__(cls, batches, cache: Path):
        import tensorrt as trt
        import torch

        class _C(trt.IInt8EntropyCalibrator2):
            def __init__(self) -> None:
                super().__init__()
                self.batches, self.i, self.cache = batches, 0, cache
                self.buf = torch.empty(batches[0].shape, dtype=torch.float32, device="cuda")

            def get_batch_size(self) -> int:
                return int(batches[0].shape[0])

            def get_batch(self, names):
                if self.i >= len(self.batches):
                    return None
                self.buf.copy_(torch.from_numpy(self.batches[self.i]))
                self.i += 1
                return [int(self.buf.data_ptr())]

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, cache) -> None:
                self.cache.write_bytes(bytes(cache))

        return _C()


def build(onnx_path: Path, mode: str, calib_batches, work: Path, fp16: bool = True, conv_only_int8: bool = False):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16:  # off: ops outside INT8 run in FP32 (FP16 can overflow on rescaled equalised tensors)
        config.set_flag(trt.BuilderFlag.FP16)
    if mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(calib_batches, work / f"{onnx_path.stem}.calib")
    if conv_only_int8:
        # INT8 only for convolutions and fully connected layers; every other layer (attention's
        # matmuls, divisions, reductions, gates) pinned to FP16, so TensorRT's tactic timing can no
        # longer pick INT8 for them (EfficientViT-B0: -0.1 / -8.9 / -25.4 / -70.3 across builds)
        keep = {trt.LayerType.CONVOLUTION}  # TensorRT 10 has no FULLY_CONNECTED layer type
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in keep or layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHUFFLE):
                continue
            outs = [layer.get_output(k) for k in range(layer.num_outputs)]
            if outs and all(o.dtype == trt.float32 for o in outs):  # shape/int layers keep their types
                layer.precision = trt.float16
                for k in range(layer.num_outputs):
                    layer.set_output_type(k, trt.float16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    elif mode == "qdq":
        config.set_flag(trt.BuilderFlag.INT8)  # explicit: scales come from the QDQ nodes
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT build failed")
    return trt.Runtime(logger).deserialize_cuda_engine(plan)


def run_engine(engine, xs: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    import tensorrt as trt
    import torch

    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    tdt = {trt.float32: torch.float32, trt.float16: torch.float16}
    x_buf = torch.empty(tuple(engine.get_tensor_shape(inp)), dtype=tdt[engine.get_tensor_dtype(inp)], device="cuda")
    y_buf = torch.empty(tuple(engine.get_tensor_shape(out)), dtype=tdt[engine.get_tensor_dtype(out)], device="cuda")
    ctx.set_tensor_address(inp, x_buf.data_ptr())
    ctx.set_tensor_address(out, y_buf.data_ptr())
    stream = torch.cuda.Stream()
    ys, times = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for x in xs:
        x_buf.copy_(torch.from_numpy(x))  # copy_ casts to the engine's input dtype
        with torch.cuda.stream(stream):
            start.record(stream)
            ctx.execute_async_v3(stream.cuda_stream)
            end.record(stream)
        stream.synchronize()
        times.append(start.elapsed_time(end))
        ys.append(y_buf.float().cpu().numpy().copy())
    return ys, float(np.median(times[10:] if len(times) > 20 else times))


def modelopt_qdq(src: Path, dst: Path, calib_batches, exclude: list[str] | None = None) -> None:
    """ModelOpt's explicit INT8 QDQ. Calibrated on the CPU (its GPU calibration failed on Kaggle
    with onnxruntime's "CopyTensorAsync is not implemented"), and with the unquantized ops left in
    FP32: newer ModelOpt converts them to FP16 by default, which also retypes the model's input."""
    import inspect

    import onnx
    from modelopt.onnx.quantization import quantize

    name = onnx.load(str(src)).graph.input[0].name
    kw = dict(onnx_path=str(src), quantize_mode="int8", calibration_data={name: np.concatenate(calib_batches)},
              calibration_method="entropy", output_path=str(dst), calibration_eps=["cpu"],
              high_precision_dtype="fp32", nodes_to_exclude=exclude or None)
    accepted = inspect.signature(quantize).parameters
    dropped = [k for k in kw if k not in accepted]
    if dropped:
        print(f"    modelopt: this version has no {dropped}", flush=True)
    quantize(**{k: v for k, v in kw.items() if k in accepted})
    m = onnx.load(str(dst))
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    elem = m.graph.input[0].type.tensor_type.elem_type
    print(f"    modelopt qdq: {ops.get('QuantizeLinear', 0)} Q, {ops.get('Cast', 0)} Cast, input dtype "
          f"{onnx.TensorProto.DataType.Name(elem)}", flush=True)


def main() -> None:
    import onnx

    from anneal.core.artifact import sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import equalise

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="efficientnet_b0")
    ap.add_argument("--images", type=int, default=1000)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    import tensorrt as trt

    report = {"runtime": f"TensorRT {trt.__version__}", "n": args.images, "models": {}}
    from importlib.metadata import PackageNotFoundError, version

    report["versions"] = {}
    for pkg in ("nvidia-modelopt", "onnxruntime-gpu", "onnxruntime", "onnx", "timm", "torch"):
        try:
            report["versions"][pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    print(f"versions {report['versions']}", flush=True)
    try:
        import torch

        report["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    work = Path("trt-work")
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        mdir = work / name
        mdir.mkdir(parents=True, exist_ok=True)
        src = mdir / f"{name}-fp32.onnx"
        if not src.exists():
            export(name, src)
        m = onnx.load(str(src))  # static batch 1: TensorRT then needs no optimisation profile
        for vi in list(m.graph.input) + list(m.graph.output):
            d = vi.type.tensor_type.shape.dim[0]
            d.ClearField("dim_param")
            d.dim_value = 1
        onnx.save(m, str(src))
        shape = sample_shape(src)
        calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=1, limit=128, sample_shape=shape)
        calib_imgs = list(calib.calibration_batches(128))
        models = {"plain": src, "eq": mdir / f"{name}-eq.onnx", "eq res": mdir / f"{name}-eq-res.onnx",
                  "eq grid": mdir / f"{name}-eq-grid.onnx", "eq pos": mdir / f"{name}-eq-pos.onnx"}
        n_sites = {"eq": len(equalise(src, models["eq"], calib_imgs[:64]).sites),
                   "eq res": len(equalise(src, models["eq res"], calib_imgs[:64], residual=True).sites),
                   "eq grid": len(equalise(src, models["eq grid"], calib_imgs[:64], grid_inverse=True).sites),
                   # no mirrored channels: every 1/s positive (TensorRT's gate Mul diverged at a site
                   # with negative scales in the QDQ prefix bisection)
                   "eq pos": len(equalise(src, models["eq pos"], calib_imgs[:64], allow_negative=False).sites)}
        print(f"{name}: equalised sites {n_sites}", flush=True)
        ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=1, limit=args.images, sample_shape=shape)
        pairs = list(ev.batches())
        xs = [x for x, _ in pairs]
        ys = np.concatenate([y for _, y in pairs])

        def decode(outs) -> np.ndarray:
            return np.concatenate([np.asarray(ev.decode(o)) for o in outs])

        import onnxruntime as ort

        s = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        preds = {"fp32": decode([s.run(None, {s.get_inputs()[0].name: x})[0] for x in xs])}
        rows, extra = {}, {}
        for label in [v.strip() for v in args.variants.split(",") if v.strip()]:
            # "... (no fp16)" builds without FP16 fallback; "... #k" is an independent rebuild (TensorRT's
            # INT8 build is not deterministic: EfficientViT-B0 plain scored -69.4 and -0.2 in two runs)
            base = label.split(" #")[0].replace(" (no fp16)", "").replace(" (conv-only int8)", "")
            fp16 = "(no fp16)" not in label
            conv_only = "(conv-only int8)" in label
            gates_out = base.endswith(" (gates excluded)")
            base = base.replace(" (gates excluded)", "")
            which = ("eq res" if base.endswith("eq res") else "eq grid" if base.endswith("eq grid")
                     else "eq pos" if base.endswith("eq pos")
                     else "eq" if base.endswith("+ eq") else "plain")
            t = time.time()
            try:
                if label.startswith("modelopt"):
                    qdq = mdir / f"{name}-{which.replace(' ', '-')}{'-nogates' if gates_out else ''}-qdq.onnx"
                    if not qdq.exists():
                        # the gate-side Mul(x', 1/s) nodes (and their Sigmoid) left unquantized
                        modelopt_qdq(models[which], qdq, calib_imgs,
                                     exclude=[r"anneal_eq_gate_mul_.*"] if gates_out else None)
                    if f"{base} (ort cpu)" not in preds:
                        # the same QDQ model on onnxruntime's CPU: separates ModelOpt's calibration
                        # from TensorRT's build of it (the two should agree closely)
                        qs = ort.InferenceSession(str(qdq), providers=["CPUExecutionProvider"])
                        qin = qs.get_inputs()[0]
                        cast = np.float16 if "float16" in qin.type else np.float32
                        preds[f"{base} (ort cpu)"] = decode([qs.run(None, {qin.name: x.astype(cast)})[0] for x in xs])
                        extra[f"{base} (ort cpu)"] = {"latency_ms": float("nan")}
                    engine = build(qdq, "qdq", None, mdir, fp16=fp16)
                else:
                    engine = build(models[which], "fp16" if base == "trt fp16" else "int8", calib_imgs, mdir, fp16=fp16,
                                   conv_only_int8=conv_only)
                build_s = time.time() - t
                outs, ms = run_engine(engine, xs)
                preds[label] = decode(outs)
                extra[label] = {"build_s": build_s, "latency_ms": ms}
                print(f"  {name} {label}: {ms:.3f} ms, first logits max {float(np.max(outs[0])):.3g}", flush=True)
            except Exception as exc:  # a variant the toolchain cannot build is a result too
                rows[label] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                print(f"  {name} {label}: FAILED {rows[label]['error']}", flush=True)

        ref = preds["fp32"] == ys
        out = {"fp32": {"accuracy": float(ref.mean())}, "equalised_sites": n_sites,
               "fp32_correct": "".join("1" if r else "0" for r in ref)}
        for label, p in preds.items():
            if label == "fp32":
                continue
            right = p == ys
            b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right))
            d, lo, hi = paired_delta_ci(b, c, len(ys))
            out[label] = {"accuracy": float(right.mean()), "delta_pp": d, "ci95_pp": [lo, hi],
                          "mcnemar_p": mcnemar_exact(b, c), "correct": "".join("1" if r else "0" for r in right),
                          **extra.get(label, {})}
            print(f"  {label:26s} {d:+7.2f}pp vs FP32 [{lo:+.2f},{hi:+.2f}]  {extra[label]['latency_ms']:.3f} ms",
                  flush=True)
        out.update(rows)
        report["models"][name] = out
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
