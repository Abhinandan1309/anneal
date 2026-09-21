# Auditing Microsoft Olive's output with Anneal

A head-to-head on ResNet-18: Olive quantizes it, `anneal audit` checks the result on the
same machine, and a controlled experiment explains the difference between the two tools'
static INT8 recipes.

Everything here ran on one laptop: AMD Ryzen 7 4800H (Zen 2 — AVX2, no VNNI), Windows 11,
onnxruntime 1.30.0 for both tools, olive-ai 0.13.0.

## 1. What Olive reports

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
reports; deciding is left to the user. That is a fair design choice, and not something
Olive hid. Raw figures: [`olive_reported_metrics.json`](olive_reported_metrics.json).

## 2. What the audit adds

```console
$ anneal audit resnet18-fp32.onnx olive_model.onnx --target cpu-4t --eval-limit 4000 --profile
```

Full report: [`audit-default-cpu4t/audit.md`](audit-default-cpu4t/audit.md).

- **Slower on this target too:** 1.35x the original's p50 latency on 4 threads.
- **Accuracy is genuinely fine — and now provably so.** On all 3,925 images the delta is
  +0.33pp, exact McNemar p = 0.32, and the paired 95% interval rules out any loss larger
  than 0.27pp. Olive's 256-image "−0.39pp" was noise in the other direction.
- **8.1% of predictions change class** (67 images broke, 80 were fixed, 319 changed in
  total). Aggregate accuracy hides that almost entirely.
- **Why it is slower:** only part of the graph became `QLinearConv`. The rest runs as float
  `Conv` wrapped in `QuantizeLinear`/`DequantizeLinear`, which cost ~5 ms per inference by
  themselves, plus new layout `Transpose`s.

## 3. Why Olive's recipe kept accuracy and Anneal's didn't

Anneal's own static INT8 lost 4.28pp on the same images. The recipes differed in several
ways at once, so [`../static_recipe_ab.py`](../static_recipe_ab.py) isolates them — same
64 train images for calibration, same 1,024 eval images, 4 threads:

| static INT8 variant | top-1 Δ vs FP32 | predictions changed | speed |
|---|---:|---:|---:|
| U8S8, per-channel (Anneal's old default) | **−4.20pp** (p = 6e-5) | 17.3% | 1.56x |
| U8S8, per-channel, **reduce_range** | **+0.39pp** (p = 0.48) | 4.0% | **1.52x** |
| S8S8, per-channel | −4.30pp (p = 5e-5) | 17.8% | 0.84x |
| U8S8, per-tensor | +1.27pp (p = 0.07) | 8.2% | 1.48x |
| S8S8, per-tensor (Olive's default) | +1.46pp (p = 0.03) | 8.1% | 0.83x |

Raw results: [`static_recipe_ab.json`](static_recipe_ab.json).

- **My first hypothesis was wrong.** I expected unsigned activations (U8S8) to be the
  culprit. S8S8 per-channel breaks exactly as badly.
- **Full-range per-channel weights are the problem, and 7-bit weights fix it.** Per-channel
  scaling pushes every output channel's weights out to ±127, and `reduce_range` restores
  accuracy completely at the same speed. That is consistent with the intermediate
  saturation onnxruntime documents for x86 CPUs without VNNI; confirming the kernel-level
  cause would take the same experiment on a VNNI machine, which I have not run.
- **S8S8 kernels are slow here** — both S8S8 variants ran slower than FP32, which is why
  Olive's model is accurate but slower.
- **The best recipe beats both defaults:** per-channel + reduce_range, 1.52x faster with no
  detectable accuracy change. The per-tensor "gains" of +1.3/+1.5pp are marginal at this
  sample size and should not be read as improvements.

Anneal's search had `reduce_range` in its action space and never tried it on the plain
graph. The heuristic policy now retries a broken static INT8 model with `reduce_range` and
with per-tensor scales before anything else.

## Reproducing

```bash
# Olive, in its own environment
pip install olive-ai onnxruntime==1.30.0 requests pillow
CI=1 OLIVE_DISABLE_TELEMETRY=1 python -c "from olive.workflows import run; run('examples/olive_resnet18/olive_config.json')"

# the audit and the recipe experiment, in Anneal's environment
anneal audit examples/resnet18-cpu1t/models/resnet18-fp32.onnx scratch/olive_out/default/model.onnx \
       --target cpu-4t --eval-limit 4000 --profile
python examples/static_recipe_ab.py --eval-limit 1024
```

**On telemetry:** olive-ai 0.13.0 sends usage telemetry by default. Reading its source, the
startup heartbeat is logged *before* `OLIVE_DISABLE_TELEMETRY=1` is checked, while CI
detection (e.g. `CI=1`) disables telemetry before the heartbeat. These runs set both. I did
not verify what, if anything, is transmitted in the default configuration.

**On absolute latencies:** this is a laptop. FP32 ResNet-18 on 4 threads measured anywhere
from 18 ms to 30 ms across sessions (power plan, thermals, background load). Every
comparison above is between models measured in the same session; compare ratios, not
milliseconds across tables.
