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
import json
import sys
import time
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".anneal_cache"


def qconfig(label: str):
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
    if "keep sigmoid" in label:
        cfg.extra_options["ConvertSigmoidToHardSigmoid"] = False
    if "cle" in label:
        cfg.algo_config = [CLEConfig()]
    return cfg


VARIANTS = ["fp32 hardsigmoid", "xint8", "xint8 + cle", "xint8 + anneal eq", "xint8 keep sigmoid",
            "xint8 keep sigmoid + anneal eq", "a8w8", "a8w8 + anneal eq", "a8w8 per-ch w", "a8w8 per-ch w + anneal eq"]


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
        models = {"plain": src, "eq": eq}

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
            which = "eq" if "anneal eq" in label else "plain"
            dst = mdir / (label.replace(" ", "_").replace("+", "plus") + ".onnx")
            t = time.time()
            try:
                if label == "fp32 hardsigmoid":
                    hardsigmoid_copy(src, dst)
                else:
                    ModelQuantizer(qconfig(label)).quantize_model(str(models[which]), str(dst), Reader(inp, calib_imgs))
                timing[label] = {"quantize_s": time.time() - t}
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
