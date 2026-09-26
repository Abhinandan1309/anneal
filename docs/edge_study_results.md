# Edge study: results

The predictions graded here were committed before any device result, in
[edge_study_design.md](edge_study_design.md), and narrowed there (also before any result) to
EfficientNet-B0 and ResNet-50 on the Galaxy S24, SA8775P ADP and Pixel 8. Predictions 1–4 are
graded below; prediction 5 was withdrawn with MobileNetV3. Every number comes from
[examples/qaihub/results/](../examples/qaihub/results/), produced by
[run_qaihub.py](../examples/qaihub/run_qaihub.py).

**Method, in brief.** Qualcomm AI Hub compiles each variant for the device. `hub int8` is AI Hub's
quantizer (W8A8) calibrated on 64 Imagenette train images; `hub int8 + equalised` is the same
quantizer on Anneal's equalised model. Imagenette validation images are scored 1000-way. Every
INT8 variant is compared, image by image, with the same device's FP32 predictions: paired 95% CI
on the accuracy difference and an exact McNemar test. Latency is AI Hub's estimated inference time
from **one profile job per variant**; compute units are the op counts the profile reports per unit.

## Grades at a glance

| # | Prediction | Grade |
|---|---|---|
| 1 | `hub int8` EfficientNet-B0 > 20pp below FP32 everywhere | **FAILED** on magnitude: 10.9–13.3pp, large and significant everywhere |
| 2 | Equalisation recovers ≥ 2/3 of the loss; `anneal recipe` within 3pp | **PARTLY**: 85–94% recovered everywhere; `anneal recipe` never ran, so untested |
| 3 | ResNet-50 `hub int8` within 1.5pp of FP32 everywhere | **HELD** where testable (S24, Pixel 8); SA8775P FP32 failed |
| 4 | INT8 ≥ 2x faster on every NPU; equalisation costs < 15% | **FAILED**: 2.04–2.26x on the S24 holds, but equalisation costs 29–32% latency |
| 5 | MobileNetV3 in between | withdrawn before the runs |

## All results

Top-1 on Imagenette validation (1000-way). Δ is INT8 minus the same device's FP32, in pp.

| Model | Device | Runtime | n | Variant | Top-1 | Δ (pp) [95% CI] | McNemar p | Latency (ms) | Compute units |
|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | Galaxy S24 | TFLite | 1,024 | fp32 | 74.6% | | | 0.847 | NPU 244 |
| | | | | hub int8 | 63.1% | −11.5 [−13.9, −9.1] | 1e-20 | 0.415 | NPU 246 |
| | | | | hub int8 + equalised | 73.1% | −1.5 [−3.0, +0.1] | 0.08 | 0.546 | NPU 262 |
| EfficientNet-B0 | Galaxy S24 | TFLite | 512 (first run) | fp32 | 73.4% | | | 0.843 | NPU 244 |
| | | | | hub int8 | 62.5% | −10.9 [−14.4, −7.5] | 7e-10 | 0.413 | NPU 246 |
| | | | | hub int8 + equalised | 71.9% | −1.6 [−3.9, +0.8] | 0.26 | 0.544 | NPU 262 |
| | | | | anneal recipe | compiled, did not run | | | | |
| EfficientNet-B0 | Galaxy S24 | QNN (DLC) | 1,024 | fp32 | 74.6% | | | 0.851 | NPU 242 |
| | | | | hub int8 | 62.6% | −12.0 [−14.4, −9.6] | 7e-22 | 0.415 | NPU 244 |
| | | | | hub int8 + equalised | 73.9% | −0.7 [−2.2, +0.9] | 0.46 | 0.534 | NPU 260 |
| EfficientNet-B0 | SA8775P ADP | TFLite | 1,024 | fp32 | 74.6% | | | 1.650 (retry) | NPU 244 |
| | | | | hub int8 | 63.1% | −11.5 [−13.9, −9.1] | 1e-20 | profile failed twice | |
| | | | | hub int8 + equalised | inference failed on 3 of 4 chunks | | | 1.071 (retry) | NPU 262 |
| EfficientNet-B0 | SA8775P ADP | TFLite | 256 (images 768–1023) | fp32 | 76.2% | | | | |
| | | | | hub int8 | 62.9% | −13.3 [−18.3, −8.2] | 6e-7 | | |
| | | | | hub int8 + equalised | 74.2% | −2.0 [−5.1, +1.2] | 0.33 | | |
| EfficientNet-B0 | Pixel 8 | TFLite | 512 | fp32 | 74.0% | | | 8.924 | GPU 244 |
| | | | | hub int8 | 61.3% | −12.7 [−16.0, −9.4] | 3e-14 | 9.199 | GPU 246 |
| | | | | hub int8 + equalised | 72.9% | −1.2 [−3.7, +1.4] | 0.45 | 9.708 | GPU 262 |
| ResNet-50 | Galaxy S24 | TFLite | 512 | fp32 | 78.5% | | | 1.384 | NPU 78 |
| | | | | hub int8 | 78.9% | +0.4 [−1.0, +1.8] | 0.79 | 0.612 | NPU 81 |
| ResNet-50 | SA8775P ADP | TFLite | 512 | fp32 | inference and profile failed | | | | |
| | | | | hub int8 | 78.9% | no paired reference | | 1.012 | NPU 81 |
| ResNet-50 | Pixel 8 | TFLite | 512 | fp32 | 78.5% | | | 16.620 | GPU 78 |
| | | | | hub int8 | 79.3% | +0.8 [−0.5, +2.1] | 0.39 | 17.262 | GPU 81 |

Notes on the table:

- **Two Galaxy S24 TFLite runs.** The first used 512 images against the pre-registered 1,024. It
  was rerun at 1,024; both are kept and agree. Grades use the 1,024-image run.
- **QNN.** The pre-registered QNN target was `qnn_context_binary`; AI Hub renamed it, and the run
  used `qnn_dlc`.
- **SA8775P.** Jobs failed intermittently (see toolchain findings). The 1,024-image FP32 and
  `hub int8` inferences completed; the equalised model's completed only on images 768–1023, so its
  comparison is on those 256 images, against the same device's FP32 on the same images. Latencies
  marked "retry" come from later profile jobs on the same compiled models.
- **Every variant ran on a single compute unit** (no CPU fallback). The equalised model adds the
  16 gate `Mul` ops (244 → 260 on QNN, 246 → 262 on TFLite).

## Predictions

### 1. The collapse is method-level: FAILED on magnitude

> `hub int8` EfficientNet-B0 is more than 20pp below `fp32` on every device and runtime.

The loss is 11.5pp (S24 TFLite), 12.0pp (S24 QNN), 11.5pp (SA8775P, 1,024 images; 13.3pp on the
256-image subset) and 12.7pp (Pixel 8). No 95% CI reaches 20pp; the largest loss inside any CI is 18.3pp,
on the 256-image subset. The threshold is missed on every device.

What held is the direction and the consistency: the loss is large, significant everywhere
(p ≤ 7e-10), and within about 2pp across two runtimes, three chips and a GPU. That fits a cause in
the quantized model rather than in any device's arithmetic, but a wrong magnitude is a failed
prediction. The 20pp was carried over from the CPU results, where per-tensor INT8 cost about 50pp;
AI Hub's quantizer loses much less than that, and this study does not isolate why.

### 2. The fix transfers: PARTLY

> `hub int8 + equalised` recovers at least two thirds of that loss on every device, and
> `anneal recipe` lands within 3pp of `fp32` wherever it compiles.

The first half **held** everywhere it could be measured:

| Device, runtime | `hub int8` Δ | `+ equalised` Δ | share of loss recovered |
|---|---|---|---|
| Galaxy S24, TFLite (1,024) | −11.5 | −1.5 | 87% |
| Galaxy S24, QNN (1,024) | −12.0 | −0.7 | 94% |
| SA8775P, TFLite (256) | −13.3 | −2.0 | 85% |
| Pixel 8, TFLite (512) | −12.7 | −1.2 | 91% |

No equalised result differs significantly from its device's FP32 (p = 0.08 to 0.46). The S24
TFLite CI, [−3.0, +0.1], only just includes zero.

The second half is **untested**. `anneal recipe` compiled on the Galaxy S24 but did not run on the
device, so there is no accuracy to grade. It was scheduled on the S24 only.

### 3. The control holds: HELD where testable

> ResNet-50 `hub int8` is within 1.5pp of `fp32` on every device.

Galaxy S24: +0.4pp [−1.0, +1.8], p = 0.79. Pixel 8: +0.8pp [−0.5, +2.1], p = 0.39. Both point
estimates are within 1.5pp; the upper CI bounds pass 1.5pp only in the direction of INT8 being
*better*. On the SA8775P the FP32 inference and profile failed, so there is no paired comparison;
its `hub int8` scores 78.9%, the same as the S24's `hub int8` on the same 512 images and 0.4pp
above the S24 and Pixel 8 FP32 (78.5%). That is consistent with the prediction but is not the
pre-registered test.

The stated reason (32-bit accumulation on NPUs) is not tested by this result, and the Pixel 8 ran
on its GPU, not an NPU.

### 4. INT8 pays on NPUs: FAILED

> `hub int8` is at least 2x faster than `fp32` on every NPU, and equalisation costs under 15% of
> that speed.

| Device, runtime, model | fp32 (ms) | hub int8 (ms) | speedup | + equalised (ms) | equalised vs hub int8 |
|---|---|---|---|---|---|
| Galaxy S24, TFLite, EfficientNet-B0 (1,024) | 0.847 | 0.415 | 2.04x | 0.546 | +32% latency |
| Galaxy S24, TFLite, EfficientNet-B0 (512) | 0.843 | 0.413 | 2.04x | 0.544 | +32% latency |
| Galaxy S24, QNN, EfficientNet-B0 | 0.851 | 0.415 | 2.05x | 0.534 | +29% latency |
| Galaxy S24, TFLite, ResNet-50 | 1.384 | 0.612 | 2.26x | | |
| SA8775P, TFLite, EfficientNet-B0 | 1.650 | failed twice | not measured | 1.071 | not measured |
| SA8775P, TFLite, ResNet-50 | failed | 1.012 | not measured | | |

- **The 2x half held on the S24** for both models and both runtimes, narrowly for EfficientNet-B0
  (2.04x, 2.05x). Each latency is one profile job, but the two independent S24 TFLite runs agree
  to 0.004 ms.
- **The 15% half failed.** The 16 gate `Mul` ops (6% more ops) add 29–32% latency, i.e. the
  equalised model keeps 76–78% of `hub int8`'s speed, not 85%. It is still 1.55–1.59x faster than
  FP32 on the S24, and 1.54x on the SA8775P (1.650 → 1.071 ms). On CPUs the same ops cost about
  10%; on this NPU they cost about three times that.
- **The SA8775P speedup is not measured**: no model has both an FP32 and a `hub int8` profile
  there.
- **The Pixel 8 is outside the prediction.** AI Hub placed every op on the GPU, not the Tensor G3's
  TPU, so it is not an NPU result. There INT8 was not faster: 0.97x (EfficientNet-B0) and 0.96x
  (ResNet-50), and the equalised model 0.92x of FP32.

### 5. MobileNetV3 sits in between: withdrawn

Withdrawn with its model in the narrowing amendment, before any result. Not run.

## What would change the conclusions (checks from the design)

- **Does the control collapse?** No. ResNet-50 `hub int8` is within +0.4 and +0.8pp of FP32 on
  the S24 and Pixel 8, so the pipeline (static shapes, calibration, compilation) is not suspect.
  The SA8775P control is unpaired (FP32 failed).
- **Does FP32 differ across devices by more than noise?** No. EfficientNet-B0 FP32 is 74.6% on the
  S24 TFLite, S24 QNN and SA8775P (1,024 images), and 73.4% (S24) vs 74.0% (Pixel 8 GPU) on 512
  images, three images apart. ResNet-50 FP32 is 78.5% on both the S24 and Pixel 8. The SA8775P and
  S24 TFLite even score the same number of images correct for FP32 and `hub int8`; their INT8 top-1
  predictions differ from their FP32 ones on 254 vs 253 images, so the runs are not copies.
  Comparisons use per-device FP32 in any case.
- **Does a variant fail to compile or run?** Yes, on two counts, reported below.

## Toolchain findings

- **`anneal recipe` compiles but does not run on the Galaxy S24.** AI Hub compiled Anneal's own QDQ
  model for TFLite (compile job `jp2rodjxg`), but inference and profiling on the device failed with
  no error message. Anneal's recipe therefore has no NPU result; the fix that transferred is
  equalisation fed to the vendor's quantizer.
- **SA8775P ADP jobs fail intermittently**, with "Failed to fully run the model, failed after
  compiling", for every variant including FP32, on models that ran elsewhere and, for each variant,
  also succeeded there at least once. Recorded failures on EfficientNet-B0: equalised inference
  `jgnz1q2kg`, and chunks `jgnz1wljg`, `jpyo8yr05`, `jp0mox20g` (chunk `jp8ejkmqp` succeeded); FP32
  profile `jprlxdk0p` (later succeeded as `jpxl0zn8p`); `hub int8` profile `jp2rod8rg` and
  `j5m09lq7g`; the equalised profile succeeded as `jpyo82e85`. The first run's failed job IDs
  (all three profiles, equalised inference) and ResNet-50's failed FP32 jobs are not recorded in
  the results files. Because the same compiled models succeeded on retry, the failures are not attributed to the models.
- **The Pixel 8 runs TFLite models on its GPU.** Every op of every variant was placed on the GPU,
  so the Pixel 8 tests another vendor's silicon (Q3) but not another vendor's NPU, and its latencies
  say nothing about INT8 on an NPU.

## Limitations

- **Imagenette, not ImageNet.** Images are Imagenette validation images scored 1000-way; absolute
  accuracies are not comparable with published ImageNet numbers, only the paired deltas are
  meaningful.
- **Sample sizes.** 512 images give a CI of about ±3pp, enough for the 12pp collapse but not to
  resolve a 1–2pp residual. The SA8775P fix comparison has only 256 images (CI −5.1 to +1.2pp)
  against the pre-registered 1,024.
- **One profile job per latency.** Run-to-run variance is not measured, except that two S24 TFLite
  runs agree within 0.004 ms. The 2.04x margin over 2x is small.
- **Narrowed scope.** One phone NPU generation, one automotive chip, one GPU; one QNN check. Q1's
  generation sweep, the Exynos device and the IoT board were dropped before the runs, so "every
  device" in the grades means these three.
- **Calibration** is 64 images, as pre-registered; AI Hub's quantizer settings beyond W8A8 were not
  varied.
