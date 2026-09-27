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

    return [n.name for n in onnx.load(str(path)).graph.node if n.name.startswith((PREFIX, GATE_MUL_PREFIX))]


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

        cfg.specific_layer_config = {QLayerConfig(activation=Int16Spec(), weight=XInt8Spec()): list(gates or [])}
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
            "xint8 percentile cal + anneal pt-eq + surrogate + a16 gates"]


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
        if not src.exists():
            export_torchvision(name, src)
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
        pt_gc = mdir / f"{name}-pt-equalised-gate-conv.onnx"
        equalise(src, pt_gc, calib_imgs, residual=True, se=True, mix=(0.5, 0.5), gate_conv=True)
        cross_layer_equalise(pt_gc, pt_gc, max_scale=4.0)
        pt_gc_sur = mdir / f"{name}-pt-equalised-gate-conv-surrogate.onnx"
        replace_sigmoids(pt_gc, pt_gc_sur, calib_imgs, k_terms=3)
        models = {"plain": src, "eq": eq, "sur": sur, "eq sur": eq_sur, "pt": pt, "pt sur": pt_sur,
                  "pt gc sur": pt_gc_sur}

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
            which = ("pt gc sur" if "gate-conv" in label
                     else "pt sur" if "pt-eq" in label and "surrogate" in label else "pt" if "pt-eq" in label
                     else "eq sur" if "anneal eq" in label and "surrogate" in label else "eq" if "anneal eq" in label
                     else "sur" if "surrogate" in label else "plain")
            dst = mdir / (label.replace(" ", "_").replace("+", "plus") + ".onnx")
            t = time.time()
            try:
                if label == "fp32 hardsigmoid":
                    hardsigmoid_copy(src, dst)
                elif label == "fp32 surrogate":
                    shutil.copy(models["sur"], dst)
                else:
                    gates = gate_nodes(models[which])
                    if "gates" in label:
                        print(f"    {len(gates)} gate ops kept wider", flush=True)
                    ModelQuantizer(qconfig(label, gates)).quantize_model(str(models[which]), str(dst), Reader(inp, calib_imgs))
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
