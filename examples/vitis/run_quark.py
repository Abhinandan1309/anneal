"""Does Anneal's equalisation hold under AMD's quantizer? Vitis AI / Ryzen AI via AMD Quark.

AMD's XINT8 preset (what its CNN NPUs and DPUs run) is the harshest scheme tested here: symmetric
power-of-two scales, *per-tensor* for activations and weights alike. Anneal's equalisation is exact
only under per-channel weights, so XINT8 is a stress test, not a formality. Quark also ships
cross-layer equalisation (CLE, Nagel et al. 2019), the natural baseline.

Variants, each scored on Imagenette validation images (1000-way) and paired against FP32:

* ``xint8``                      AMD's NPU preset on the exported model
* ``xint8 + cle``                the same with Quark's CLE
* ``xint8 + anneal eq``          the preset on Anneal's equalised model
* ``xint8 keep sigmoid``         XINT8 without its Sigmoid -> HardSigmoid swap (the NPU default,
                                 which changes EfficientNet's SiLU before any quantization)
* ``xint8 keep sigmoid + anneal eq``
* ``fp32 hardsigmoid``           no quantization, only that swap: the float ceiling of XINT8
* ``a8w8``                       float scales, per-tensor symmetric int8 (Quark's A8W8 preset)
* ``a8w8 + anneal eq``
* ``a8w8 per-ch w``              A8W8 with per-channel weights (is per-tensor weight the limit?)
* ``a8w8 per-ch w + anneal eq``

(XINT8 with per-channel weights is refused by Quark: its NPU mode is per-tensor only.)

Quark JIT-builds C++ custom ops, so this runs in CI on Linux: see .github/workflows/quark-lab.yml.

    python examples/vitis/run_quark.py --models efficientnet_b0 --images 1000 --out result.json
"""

from __future__ import annotations

import argparse
import copy
import shutil
import json
import sys
import time
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".anneal_cache"


def gate_nodes(path: Path) -> list[str]:
    """The gate ops Anneal inserted: surrogate HardSigmoid terms and the gate-side 1/s rescale."""
    import onnx

    from anneal.core.equalize import GATE_MUL_PREFIX
    from anneal.core.surrogate import PREFIX

    from anneal.core.equalize import GATE_CONV_PREFIX

    return [n.name for n in onnx.load(str(path)).graph.node
            if n.name.startswith((PREFIX, GATE_MUL_PREFIX, GATE_CONV_PREFIX, "anneal_gclip_"))]


def qconfig(label: str, gates: list[str] | None = None):
    from quark.onnx import CLEConfig, Int8Spec, QConfig, QLayerConfig, QuantGranularity
    from quark.onnx.quantization.config.custom_config import A8W8_QCONFIG, XINT8_QCONFIG

    if label.startswith("a8w8"):
        cfg = copy.deepcopy(A8W8_QCONFIG)
        if "per-ch w" in label:
            cfg = QConfig(global_config=QLayerConfig(activation=Int8Spec(),
                                                     weight=Int8Spec(quant_granularity=QuantGranularity.Channel)),
                          extra_options=dict(cfg.extra_options))
        return cfg
    cfg = copy.deepcopy(XINT8_QCONFIG)
    for tag, method in (("minmax cal", "MinMax"), ("percentile cal", "Percentile")):
        if tag in label:  # activations calibrated differently (weights keep XINT8's own)
            from quark.onnx import CalibMethod, XInt8Spec

            cfg = QConfig(global_config=QLayerConfig(activation=XInt8Spec(calibration_method=getattr(CalibMethod, method)),
                                                     weight=XInt8Spec()), extra_options=dict(cfg.extra_options))
    if "no shift adjust" in label:  # diagnostic only: the DPU's shift constraints off
        for k in ("AdjustShiftCut", "AdjustShiftBias", "AdjustShiftRead", "AdjustShiftWrite"):
            cfg.extra_options[k] = False
    if "keep sigmoid" in label:
        cfg.extra_options["ConvertSigmoidToHardSigmoid"] = False
    if "exact pool" in label:  # isolate the DPU's approximations of average pooling / ReduceMean
        cfg.extra_options["ConvertAvgPoolToDPUVersion"] = False
        cfg.extra_options["ConvertReduceMeanToDPUVersion"] = False
    if "cle" in label:
        cfg.algo_config = [CLEConfig()]
    if "float gates" in label:  # upper bound: the gate ops stay float
        cfg.exclude = list(gates or [])
    if "a16 gates" in label:  # deployable: 16-bit activations on the gate ops only (NPU A16W8 path)
        from quark.onnx import Int16Spec, XInt8Spec

        cfg.specific_layer_config = {QLayerConfig(input_tensors=Int16Spec(), output_tensors=Int16Spec(),
                                                  weight=XInt8Spec()): list(gates or [])}
    return cfg


VARIANTS = ["fp32 hardsigmoid", "xint8", "xint8 + cle", "xint8 + anneal eq", "xint8 keep sigmoid",
            "xint8 keep sigmoid + anneal eq", "a8w8", "a8w8 + anneal eq", "a8w8 per-ch w", "a8w8 per-ch w + anneal eq",
            # the HardSigmoid-sum surrogate (anneal.core.surrogate): only HardSigmoids reach the NPU
            "fp32 surrogate", "xint8 + surrogate", "xint8 + anneal eq + surrogate",
            # per-tensor-aware equalisation (XINT8 quantizes weights per tensor): squeeze-excite and
            # residual sites, activation/weight mix t=0.5, ReLU CLE capped at 4x
            "xint8 + anneal pt-eq", "xint8 + anneal pt-eq + surrogate", "a8w8 + anneal pt-eq",
            # the deployable NPU path: percentile calibration (MinMSE cost EfficientNet-B0 ~25pp)
            "xint8 percentile cal + anneal pt-eq + surrogate", "xint8 percentile cal + surrogate",
            # each gate fed x' through a depthwise 1x1 conv (weight 1/s) instead of Mul(x', 1/s), the
            # surrogate folded into it: the NPU fuses Conv + HardSigmoid, never quantizing imbalanced x
            "xint8 percentile cal + anneal pt-eq gate-conv + surrogate",
            # EfficientNet-B1 loses ~37pp in the gate branches (emulation): keep only those ops wider
            "xint8 percentile cal + anneal pt-eq + surrogate + float gates",
            "xint8 percentile cal + anneal pt-eq + surrogate + a16 gates",
            # analytic bias correction for XINT8's per-tensor pow2 weights (anneal.core.bias_correction):
            # weights-only emulation B1 -17.3 -> -0.8
            "xint8 percentile cal + anneal pt-eq + bias corr + surrogate",
            "xint8 percentile cal + anneal pt-eq + bias corr + surrogate + a16 gates",
            "xint8 percentile cal + anneal pt-eq + bias corr + surrogate + float gates",
            # the formula's Q(W) was wrong for Quark (bias corr hurt): read Quark's own Q(W) instead
            "xint8 percentile cal + anneal pt-eq + surrogate + bias corr (measured)",
            "xint8 percentile cal + anneal pt-eq + surrogate + a16 gates + bias corr (measured)",
            # gate inputs clipped to the span HardSigmoid can see (exact), alone and with the gate-conv form
            "xint8 percentile cal + anneal pt-eq + surrogate + gate clip + bias corr (measured)",
            "xint8 percentile cal + anneal pt-eq + surrogate + gate clip + a16 gates + bias corr (measured)",
            "xint8 percentile cal + anneal pt-eq gate-conv + surrogate + gate clip + bias corr (measured)",
            "xint8 percentile cal + anneal pt-eq gate-conv + surrogate + gate clip + a16 gates + bias corr (measured)",
            # every gate-side 1/s on the power-of-two int8 grid (equalise(grid_inverse=True)): lossless
            "xint8 percentile cal + anneal pt-eq grid + surrogate",
            "xint8 percentile cal + anneal pt-eq grid + surrogate + bias corr (measured)",
            "xint8 percentile cal + anneal pt-eq grid + surrogate + a16 gates + bias corr (measured)",
            "xint8 percentile cal + anneal pt-eq grid + surrogate + gate clip + a16 gates + bias corr (measured)",
            # noise-optimal per-channel scales (equalise(derived=True), anneal.core.equalize_opt) with grid 1/s
            "xint8 percentile cal + anneal pt-eq grid derived + surrogate + bias corr (measured)",
            "xint8 percentile cal + anneal pt-eq grid derived + surrogate + a16 gates + bias corr (measured)",
            # ablation of the all-8-bit B1 result (-3.5pp): gate-conv + clip without bias correction
            "xint8 percentile cal + anneal pt-eq gate-conv + surrogate + gate clip",
            # ReLU CNNs (MnasNet lost 6pp more with the gated recipe): the advisor's CNN recipe
            "xint8 percentile cal + anneal CLE4", "xint8 percentile cal + anneal CLE4 + bias corr (measured)"]


def hardsigmoid_copy(src: Path, dst: Path) -> None:
    """The float model with every Sigmoid swapped for HardSigmoid, as Quark's NPU mode does
    (``make_node("HardSigmoid", ...)`` with ONNX's default alpha 0.2, beta 0.5)."""
    import onnx
    from onnx import helper

    m = onnx.load(str(src))
    for i, n in enumerate(m.graph.node):
        if n.op_type == "Sigmoid":
            m.graph.node[i].CopyFrom(helper.make_node("HardSigmoid", list(n.input), list(n.output), name=n.name))
    onnx.save(m, str(dst))


class Reader:
    """onnxruntime CalibrationDataReader over a list of batches."""

    def __init__(self, name: str, batches: list[np.ndarray]) -> None:
        self.name, self.batches, self.i = name, batches, 0

    def get_next(self):
        if self.i >= len(self.batches):
            return None
        self.i += 1
        return {self.name: self.batches[self.i - 1]}

    def rewind(self) -> None:
        self.i = 0


def session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    try:  # Quark may emit its own custom ops (power-of-two QDQ variants)
        from quark.onnx import get_library_path

        so.register_custom_ops_library(get_library_path())
    except Exception as exc:  # noqa: BLE001 - the standard-op models still run
        print(f"  (no Quark custom-op library: {exc})", flush=True)
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main() -> None:
    import onnx

    from anneal.core.artifact import sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import equalise
    from anneal.models import export_torchvision

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="efficientnet_b0")
    ap.add_argument("--images", type=int, default=1000)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    from quark.onnx import ModelQuantizer

    report = {"quantizer": "AMD Quark", "n": args.images, "models": {}}
    work = Path("quark-work")
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        mdir = work / name
        mdir.mkdir(parents=True, exist_ok=True)
        src = mdir / f"{name}-fp32.onnx"
        if not src.exists():  # native input size; timm models through the zoo's exporter
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "zoo_gated"))
            from export_models import export_any

            export_any(name, src)
            m = onnx.load(str(src))  # batch 1 (Quark and the NPU compile static shapes)
            for vi in list(m.graph.input) + list(m.graph.output):
                d = vi.type.tensor_type.shape.dim[0]
                d.ClearField("dim_param")
                d.dim_value = 1
            onnx.save(m, str(src))
        shape = sample_shape(src)
        inp = onnx.load(str(src)).graph.input[0].name
        calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=1, limit=64, sample_shape=shape)
        calib_imgs = list(calib.calibration_batches(64))
        eq = mdir / f"{name}-equalised.onnx"
        eq_result = equalise(src, eq, calib_imgs)
        print(f"{name}: equalised {len(eq_result.sites)} sites", flush=True)
        from anneal.core.surrogate import replace_sigmoids

        sur, eq_sur = mdir / f"{name}-surrogate.onnx", mdir / f"{name}-equalised-surrogate.onnx"
        replace_sigmoids(src, sur, calib_imgs, k_terms=3)
        replace_sigmoids(eq, eq_sur, calib_imgs, k_terms=3)
        from anneal.core.cle import cross_layer_equalise

        pt = mdir / f"{name}-pt-equalised.onnx"
        equalise(src, pt, calib_imgs, residual=True, se=True, mix=(0.5, 0.5))
        cross_layer_equalise(pt, pt, max_scale=4.0)
        pt_sur = mdir / f"{name}-pt-equalised-surrogate.onnx"
        replace_sigmoids(pt, pt_sur, calib_imgs, k_terms=3)
        from anneal.core.bias_correction import correct_biases

        pt_bc_sur = mdir / f"{name}-pt-equalised-bc-surrogate.onnx"  # XINT8: per-tensor pow2 weights
        correct_biases(pt, mdir / f"{name}-pt-equalised-bc.onnx", calib_imgs, per_channel=False, pow2=True)
        replace_sigmoids(mdir / f"{name}-pt-equalised-bc.onnx", pt_bc_sur, calib_imgs, k_terms=3)
        cle4 = mdir / f"{name}-cle4.onnx"  # the advisor's recipe for ReLU CNNs on per-tensor targets
        cross_layer_equalise(src, cle4, max_scale=4.0)
        pt_grid, pt_grid_sur = mdir / f"{name}-pt-equalised-grid.onnx", mdir / f"{name}-pt-equalised-grid-surrogate.onnx"
        equalise(src, pt_grid, calib_imgs, residual=True, se=True, mix=(0.5, 0.5), grid_inverse=True)
        cross_layer_equalise(pt_grid, pt_grid, max_scale=4.0)
        replace_sigmoids(pt_grid, pt_grid_sur, calib_imgs, k_terms=3)
        pt_grid_der_sur = mdir / f"{name}-pt-equalised-grid-derived-surrogate.onnx"
        if "grid derived" in args.variants or not args.variants:
            pt_grid_der = mdir / f"{name}-pt-equalised-grid-derived.onnx"
            equalise(src, pt_grid_der, calib_imgs, residual=True, se=True, mix=(0.5, 0.5), grid_inverse=True,
                     derived=True)
            cross_layer_equalise(pt_grid_der, pt_grid_der, max_scale=4.0)
            replace_sigmoids(pt_grid_der, pt_grid_der_sur, calib_imgs, k_terms=3)
        pt_gc = mdir / f"{name}-pt-equalised-gate-conv.onnx"
        equalise(src, pt_gc, calib_imgs, residual=True, se=True, mix=(0.5, 0.5), gate_conv=True)
        cross_layer_equalise(pt_gc, pt_gc, max_scale=4.0)
        pt_gc_sur = mdir / f"{name}-pt-equalised-gate-conv-surrogate.onnx"
        replace_sigmoids(pt_gc, pt_gc_sur, calib_imgs, k_terms=3)
        models = {"plain": src, "eq": eq, "sur": sur, "eq sur": eq_sur, "pt": pt, "pt sur": pt_sur,
                  "pt gc sur": pt_gc_sur, "pt bc sur": pt_bc_sur,
                  "pt grid sur": pt_grid_sur, "pt grid der sur": pt_grid_der_sur, "cle4": cle4}

        ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=1, limit=args.images, sample_shape=shape)  # batch 1: Quark may fix it
        batches = list(ev.batches())
        ys = np.concatenate([y for _, y in batches])

        def predict(path: Path, label: str) -> np.ndarray:
            s = session(path)
            out = [np.asarray(ev.decode(s.run(None, {inp: x})[0])) for x, _ in batches]
            first = s.run(None, {inp: batches[0][0][:1]})[0].ravel()
            p = np.concatenate(out)
            print(f"    {label}: logits min {first.min():.3g} max {first.max():.3g}, {len(p)} predictions, "
                  f"{len(np.unique(p))} distinct", flush=True)
            if len(p) != len(ys):  # numpy would compare unequal lengths as all-False, i.e. 0% accuracy
                raise RuntimeError(f"{len(p)} predictions for {len(ys)} labels: output batch dimension changed")
            return p

        preds = {"fp32": predict(src, "fp32")}
        rows, timing = {}, {}
        for label in [v.strip() for v in args.variants.split(",") if v.strip()]:
            which = ("cle4" if "anneal CLE4" in label else "pt grid der sur" if "pt-eq grid derived" in label
                     else "pt grid sur" if "pt-eq grid" in label
                     else "pt bc sur" if "bias corr" in label and "measured" not in label
                     else "pt gc sur" if "gate-conv" in label
                     else "pt sur" if "pt-eq" in label and "surrogate" in label else "pt" if "pt-eq" in label
                     else "eq sur" if "anneal eq" in label and "surrogate" in label else "eq" if "anneal eq" in label
                     else "sur" if "surrogate" in label else "plain")
            dst = mdir / (label.replace(" ", "_").replace("+", "plus") + ".onnx")
            t = time.time()
            try:
                base_model = models[which]
                if "gate clip" in label:  # Clip before every gate: exact for HardSigmoid
                    from anneal.core.surrogate import clip_gate_inputs

                    clipped = mdir / f"{Path(base_model).stem}-gclip.onnx"
                    if not clipped.exists():
                        print(f"    clipped {clip_gate_inputs(base_model, clipped)} gate inputs", flush=True)
                    base_model = clipped
                if "bias corr (measured)" in label:
                    # two passes: quantize, read Quark's own Q(W), correct biases, quantize again
                    from anneal.core.bias_correction import correct_biases, weights_from_qdq

                    base_label = label.replace(" + bias corr (measured)", "")
                    first = mdir / "bc-first-pass.onnx"
                    ModelQuantizer(qconfig(base_label, gate_nodes(base_model))).quantize_model(
                        str(base_model), str(first), Reader(inp, calib_imgs))
                    qw = weights_from_qdq(first)
                    corrected = mdir / f"{Path(base_model).stem}-bc-measured.onnx"
                    r = correct_biases(base_model, corrected, calib_imgs, quantized=qw)
                    print(f"    measured Q(W) for {len(qw)} layers, corrected {len(r.layers)} biases", flush=True)
                    ModelQuantizer(qconfig(base_label, gate_nodes(corrected))).quantize_model(
                        str(corrected), str(dst), Reader(inp, calib_imgs))
                elif label == "fp32 hardsigmoid":
                    hardsigmoid_copy(src, dst)
                elif label == "fp32 surrogate":
                    shutil.copy(models["sur"], dst)
                else:
                    gates = gate_nodes(base_model)
                    if "gates" in label:
                        print(f"    {len(gates)} gate ops kept wider", flush=True)
                    ModelQuantizer(qconfig(label, gates)).quantize_model(str(base_model), str(dst), Reader(inp, calib_imgs))
                timing[label] = {"quantize_s": time.time() - t}
                if "gate-conv" in label:  # does Quark leave the gate conv's output unquantized (fused)?
                    import onnx as _onnx

                    qm = _onnx.load(str(dst))
                    prod = {o: n for n in qm.graph.node for o in n.output}
                    between, total = 0, 0
                    for n in qm.graph.node:
                        if n.op_type == "HardSigmoid":
                            total += 1
                            src_node = prod.get(n.input[0])
                            if src_node is not None and src_node.op_type == "DequantizeLinear":
                                between += 1
                    print(f"    fusion check: {between}/{total} HardSigmoids read a dequantized (8-bit) input", flush=True)
                preds[label] = predict(dst, label)
            except Exception as exc:  # a variant the toolchain cannot quantize is a result too
                rows[label] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                print(f"  {name} {label}: FAILED {rows[label]['error']}", flush=True)
                continue
            print(f"  {name} {label}: done ({timing[label]})", flush=True)

        ref = preds["fp32"] == ys
        out = {"fp32": {"accuracy": float(ref.mean())}, "equalised_sites": len(eq_result.sites)}
        for label, p in preds.items():
            if label == "fp32":
                continue
            right = p == ys
            b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right))
            d, lo, hi = paired_delta_ci(b, c, len(ys))
            out[label] = {"accuracy": float(right.mean()), "delta_pp": d, "ci95_pp": [lo, hi],
                          "mcnemar_p": mcnemar_exact(b, c), "agreement": float(np.mean(p == preds["fp32"])),
                          "correct": "".join("1" if r else "0" for r in right), **timing.get(label, {})}
            print(f"  {label:28s} {d:+7.2f}pp vs FP32 [{lo:+.2f},{hi:+.2f}]", flush=True)
        out["fp32_correct"] = "".join("1" if r else "0" for r in ref)
        out.update(rows)
        report["models"][name] = out
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
