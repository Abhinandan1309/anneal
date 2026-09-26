"""How much of AMD's Sigmoid -> HardSigmoid swap does the HardSigmoid-sum surrogate recover? Float only.

AMD's NPU/DPU toolchain replaces every Sigmoid with HardSigmoid(u/6 + 1/2) before quantizing.
Measured in float, that swap alone cost EfficientNet-B0 48.8pp and EfficientNet-B1 75.7pp top-1.
:mod:`anneal.core.surrogate` replaces each Sigmoid by a per-gate fitted sum of K such
HardSigmoids with exact asymptotes; K=3 gave B0 +0.00pp and B1 -2.0pp against FP32.

Variants, each scored on Imagenette validation images (1000-way) and paired against FP32:

* ``AMD swap h(x)``   every Sigmoid -> HardSigmoid(alpha=1/6, beta=1/2)
* ``surrogate K=1..3`` :func:`anneal.core.surrogate.replace_sigmoids`, fitted on calibration images

    python examples/vitis/surrogate_float.py examples/models/efficientnet_b0-fp32.onnx --images 1000
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".anneal_cache"


def amd_swap(src: Path, dst: Path) -> None:
    """What Quark's ConvertSigmoidToHardSigmoid does to the float graph."""
    import onnx
    from onnx import helper

    from anneal.core.surrogate import ALPHA, BETA

    model = onnx.load(str(src))
    for n in model.graph.node:
        if n.op_type == "Sigmoid":
            n.op_type = "HardSigmoid"
            n.attribute.extend([helper.make_attribute("alpha", ALPHA), helper.make_attribute("beta", BETA)])
    onnx.save(model, str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path, help="FP32 ONNX classifier (ImageNet-1k outputs)")
    ap.add_argument("--images", type=int, default=1000, help="Imagenette validation images to score")
    ap.add_argument("--calib", type=int, default=16, help="calibration images the fits are measured on")
    ap.add_argument("--terms", default="1,2,3", help="comma-separated K values to try")
    ap.add_argument("--out", type=Path, help="write results as JSON here")
    args = ap.parse_args()

    import onnxruntime as ort

    from anneal.core.artifact import sample_shape
    from anneal.core.audit import paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.surrogate import replace_sigmoids

    src = args.model
    shape = sample_shape(src)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=args.calib, sample_shape=shape)
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=32, limit=args.images, sample_shape=shape)
    batches = list(ev.batches())
    y = np.concatenate([b[1] for b in batches])

    def correct(path: Path) -> np.ndarray:
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        i = s.get_inputs()[0].name
        return np.concatenate([np.asarray(ev.decode(s.run(None, {i: x})[0])) for x, _ in batches]) == y

    ref = correct(src)
    print(f"{src.name}: fp32 top-1 {ref.mean():.3f} on {len(y)} images", flush=True)
    results = {"model": src.name, "images": len(y), "fp32_top1": float(ref.mean()), "variants": {}}
    with tempfile.TemporaryDirectory() as tmp:
        variants: list[tuple[str, Path, dict]] = []
        plain = Path(tmp) / "amd.onnx"
        amd_swap(src, plain)
        variants.append(("AMD swap h(x)", plain, {}))
        for K in (int(k) for k in args.terms.split(",")):
            dst = Path(tmp) / f"k{K}.onnx"
            report = replace_sigmoids(src, dst, calib.calibration_batches(args.calib), k_terms=K)
            summary = {k: v for k, v in report.items() if k != "per_gate"}
            variants.append((f"surrogate K={K}", dst, summary))
        for label, path, summary in variants:
            r = correct(path)
            b, c = int(np.sum(ref & ~r)), int(np.sum(~ref & r))
            d, lo, hi = paired_delta_ci(b, c, len(y))
            extra = f"  median fit loss {summary['median_loss']:.2e}" if summary else ""
            print(f"  {label:16s} top-1 {r.mean():.3f}  {d:+7.2f}pp [{lo:+.2f},{hi:+.2f}]{extra}", flush=True)
            results["variants"][label] = {"top1": float(r.mean()), "delta_pp": d, "ci": [lo, hi], **summary}
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
