"""A/B: does ranking layers by measurement beat the weight-error proxy?

Selective quantization spares the top-k "most sensitive" layers from INT8. This quantizes
ResNet-18 twice at each k — once sparing the proxy's top-k, once sparing the measured
sweep's top-k — and scores both against the FP32 baseline on the same images.

Latency is irrelevant here (both variants use the same slow dynamic-INT8 kernels on this
target); the question is purely which ranking protects the model's behaviour better.

    python examples/ranking_ab.py --eval-limit 1024 --k 1

Requires the ResNet-18 example run and its sensitivity sweep in examples/resnet18-cpu1t/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anneal.core.artifact import ModelArtifact
from anneal.core.dataset import load_evalset
from anneal.core.measure import Benchmarker
from anneal.core.targets import default_target
from anneal.core.transforms import TransformContext, apply_transform, load_measured_ranking
from anneal.report import wilson_halfwidth_pp

HERE = Path(__file__).parent / "resnet18-cpu1t"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-limit", type=int, default=1024)
    parser.add_argument("--k", type=int, nargs="+", default=[1])
    parser.add_argument("--out", default=str(HERE / "ranking_ab.json"))
    args = parser.parse_args()

    baseline = ModelArtifact(path=HERE / "models" / "resnet18-fp32.onnx")
    measured = load_measured_ranking(HERE / "sensitivity" / "sensitivity.json")
    evalset = load_evalset(
        "imagenette", cache_dir=Path.home() / ".anneal_cache", batch_size=32, limit=args.eval_limit
    )
    bench = Benchmarker(default_target(), warmup=1, runs=2)
    base = bench.measure(baseline, evalset=evalset, record_baseline=True)
    print(f"baseline top-1 {base.accuracy * 100:.2f}% on {len(evalset)} images")

    rows = []
    for k in args.k:
        for ranking in ("proxy", "measured"):
            ctx = TransformContext(workdir=HERE / "ab_candidates")
            if ranking == "measured":
                ctx.extra["measured_ranking"] = measured
            candidate = apply_transform(
                "quantize_dynamic_sensitive",
                {"skip_top_k": k, "per_channel": True, "ranking": ranking},
                baseline,
                ctx,
            )
            m = bench.measure(candidate, evalset=evalset)
            changed = 1.0 - (m.top1_agreement or 0.0)
            row = {
                "k": k,
                "ranking": ranking,
                "spared": candidate.meta["excluded_nodes"],
                "accuracy": m.accuracy,
                "changed_fraction": changed,
                "changed_images": round(changed * len(evalset)),
            }
            rows.append(row)
            print(
                f"k={k} {ranking:8s} spared={row['spared']}  "
                f"top-1 {m.accuracy * 100:.2f}%  changed {changed * 100:.2f}% "
                f"({row['changed_images']} images)"
            )

    Path(args.out).write_text(
        json.dumps(
            {
                "n_eval": len(evalset),
                "baseline_accuracy": base.accuracy,
                "accuracy_resolution_pp": wilson_halfwidth_pp(base.accuracy, len(evalset)),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
