# Auditing Microsoft Olive's output with Anneal

A head-to-head on ResNet-18: Olive quantizes it (first with defaults, then with its search),
`anneal audit` checks the results on the same machine, and a controlled experiment explains
the difference between the two tools' static INT8 recipes.

Everything here ran on one laptop: AMD Ryzen 7 4800H (Zen 2 — AVX2, no VNNI), Windows 11,
onnxruntime 1.30.0 for both tools, olive-ai 0.13.0.

> **Correction (22 Sep 2026).** An earlier version of this page reported static INT8 with
> `reduce_range` at **1.52x** FP32 speed. That measurement ran while the laptop was on
> battery with battery saver engaged — undetected at the time. Throttling slowed FP32 by
> ~25% and INT8 hardly at all, inflating every INT8 speedup. Re-measured on AC power the
> same recipe is **1.05x** on one thread. The accuracy results were unaffected (the models
> are deterministic) and are unchanged below. This episode is why Anneal now checks power
> state before measuring. The battery-session data is kept as
> [`static_recipe_ab.json`](static_recipe_ab.json) for the record.

## 1. What Olive reports, with default settings

[`olive_config.json`](olive_config.json) runs Olive's `OnnxStaticQuantization` pass with its
defaults, calibrated on 64 Imagenette **train** images and evaluated with Olive's own
evaluator on 256 validation images — the same 256 Anneal's search uses, preprocessed by the
same code ([`olive_data.py`](olive_data.py)), so both tools score the FP32 model at an
identical 66.797%.

| | FP32 | Olive INT8 |
|---|---:|---:|
| accuracy (256 images) | 66.80% | 66.41% |
| avg latency (Olive's session defaults, all cores) | 11.82 ms | 20.12 ms |

Olive's own numbers say the quantized model is **1.70x slower**. It is written out as the
result anyway, because this workflow sets no search objective — Olive applies the pass and
reports; deciding is left to the user. Raw figures:
[`olive_reported_metrics.json`](olive_reported_metrics.json).

## 2. What the audit adds

```console
$ anneal audit resnet18-fp32.onnx olive_model.onnx --target cpu-4t --eval-limit 4000
```

Re-audited on AC power: [`audit-default-cpu4t-ac/audit.md`](audit-default-cpu4t-ac/audit.md).

- **Slower here too:** 0.72x the original's speed at p50 on 4 threads (0.74x at p99).
- **Accuracy is genuinely fine — and now provably so.** On all 3,925 images the delta is
  +0.33pp, exact McNemar p = 0.32, and the paired 95% interval rules out any loss larger
  than 0.27pp. Olive's 256-image "−0.39pp" was noise in the other direction.
- **8.1% of predictions change class** (67 images broke, 80 were fixed). Aggregate accuracy
  hides that almost entirely.
- **Why it is slow** (from the earlier profiled audit, [`audit-default-cpu4t/`](audit-default-cpu4t/)):
  only part of the graph became `QLinearConv`; the rest runs as float `Conv` wrapped in
  `QuantizeLinear`/`DequantizeLinear`, plus new layout `Transpose`s.

## 3. Why Olive's recipe kept accuracy and Anneal's didn't

Anneal's own static INT8 lost 4.28pp on the same images. The recipes differed in several
ways at once, so [`../static_recipe_ab.py`](../static_recipe_ab.py) isolates them — same 64
train images for calibration, same 1,024 eval images, measured on AC power:

| static INT8 variant | top-1 Δ vs FP32 | predictions changed | speed, 1 thread | speed, 4 threads |
|---|---:|---:|---:|---:|
| U8S8, per-channel (Anneal's old default) | **−4.20pp** (p = 6e-5) | 17.3% | 1.06x | 1.10x |
| U8S8, per-channel, **reduce_range** | **+0.39pp** (p = 0.48) | 4.0% | 1.05x | 1.14x |
| S8S8, per-channel | −4.30pp (p = 5e-5) | 17.8% | 0.84x | 0.75x |
| U8S8, per-tensor | +1.27pp (p = 0.07) | 8.2% | 1.05x | 1.19x |
| S8S8, per-tensor (Olive's default) | +1.46pp (p = 0.03) | 8.1% | 0.84x | 0.72x |

Raw results: [`static_recipe_ab_ac_cpu1t.json`](static_recipe_ab_ac_cpu1t.json),
[`static_recipe_ab_ac_cpu4t.json`](static_recipe_ab_ac_cpu4t.json).

- **My first hypothesis was wrong.** I expected unsigned activations (U8S8) to be the
  culprit. S8S8 per-channel breaks exactly as badly.
- **Full-range per-channel weights are the problem, and 7-bit weights fix it.** Per-channel
  scaling pushes every output channel's weights out to ±127; `reduce_range` restores accuracy
  completely. That is consistent with the intermediate saturation onnxruntime documents for
  x86 CPUs without VNNI; confirming the kernel-level cause needs the same experiment on a
  VNNI CPU, which the [hardware lab](../hardware_lab/) is built to run.
- **S8S8 kernels are slower than FP32 on this CPU** in every session — which is why Olive's
  default model is accurate but slow.
- **The speed prize is small.** On one thread, where repeated measurements agree to within
  a few percent, the best INT8 recipes are about **1.05x** FP32. Four-thread figures on this
  laptop vary by ±20% between runs even on AC power, so the 4-thread column shows what one
  run measured, not a reliable ranking.
- The per-tensor "gains" of +1.3/+1.5pp are marginal at this sample size and should not be
  read as improvements.

Anneal's search had `reduce_range` in its action space and never tried it on the plain
graph. The heuristic policy now retries a broken static INT8 model with `reduce_range` and
with per-tensor scales before anything else.

## 4. Olive's search against Anneal's search

To compare the tools fairly, [`olive_search_config.json`](olive_search_config.json) runs
Olive's own search: 30 trials over its static-quantization space (precision, per-channel,
reduce_range, calibration method, quantization format, pre-processing), the same data, the
same 1pp accuracy goal. Anneal's search was re-run with the fixed policy
([`../resnet18-cpu1t-v2/`](../resnet18-cpu1t-v2/)). Both tools' final picks were then audited
identically — all 3,925 images, AC power, A-B-A timing (reports in
[`headtohead/`](headtohead/)):

| | Anneal's pick | Olive's pick |
|---|---|---|
| recipe | U8S8, per-tensor, QDQ | uint8 weights, per-tensor, reduce_range, QOperator, entropy calibration |
| why it was picked | fastest within budget (1 thread) | highest accuracy on 256 images |
| speed, 1 thread | 1.06x | 1.01x |
| speed, 4 threads | 0.92x | 1.12x (p99: 0.73x) |
| accuracy change, 3,925 images | +0.43pp (p = 0.19) | −0.48pp (p = 0.23) |
| worst loss the data cannot rule out | **0.17pp** | **1.23pp — beyond the 1pp budget** |
| predictions changed | 7.8% | 12.5% |

- **Olive's search found the reduce_range fix on its own.** With an objective it is a
  capable searcher; the gap in section 1 was the default workflow, not the tool.
- **Neither pick is faster everywhere.** Each wins on the thread count it was chosen for,
  within the noise noted above. There is no single best recipe — which is the premise of
  this project.
- **Olive's final choice was made by noise.** Among near-identical candidates it picked the
  one with the highest accuracy on 256 images — 68.36%, or +1.56pp, which is four images.
  On the full set that model is at −0.48pp, and its loss cannot be confirmed to be within
  budget. Choosing the best-looking of many candidates on a small sample systematically
  selects a lucky one (the winner's curse). Anneal's pick, judged the same way, is
  confirmed within budget.

## Reproducing

```bash
# Olive, in its own environment
pip install olive-ai onnxruntime==1.30.0 requests pillow
CI=1 OLIVE_DISABLE_TELEMETRY=1 python -c "from olive.workflows import run; run('examples/olive_resnet18/olive_config.json')"
CI=1 OLIVE_DISABLE_TELEMETRY=1 python -c "from olive.workflows import run; run('examples/olive_resnet18/olive_search_config.json')"

# the audits and the recipe experiment, in Anneal's environment
anneal audit examples/resnet18-cpu1t/models/resnet18-fp32.onnx scratch/olive_out/default/model.onnx \
       --target cpu-4t --eval-limit 4000
python examples/static_recipe_ab.py --eval-limit 1024 --target cpu-1t
```

**On telemetry:** olive-ai 0.13.0 sends usage telemetry by default. Reading its source, the
startup heartbeat is logged *before* `OLIVE_DISABLE_TELEMETRY=1` is checked, while CI
detection (e.g. `CI=1`) disables telemetry before the heartbeat. These runs set both. I did
not verify what, if anything, is transmitted in the default configuration.

**On the latency settings:** Olive was configured to time with 4 intra-op threads, but its
FP32 figure (8 ms) is about half what Anneal measures at 4 threads, so the setting may not
have applied as intended. The head-to-head therefore uses only Anneal's own timings for
both models; Olive's latencies are quoted solely as what Olive reported.
