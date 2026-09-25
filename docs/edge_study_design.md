# Edge study: design and predictions (written before the results)

This file is committed **before** the device runs it describes, so its predictions cannot be
adjusted to fit the data. Results go in a separate file that links back here; any prediction
that fails is reported as a finding.

## Why

Every INT8 result in this repository so far ran on server and laptop CPUs through
onnxruntime. Edge deployment happens on NPUs, with vendor toolchains. The question is which of
Anneal's findings belong to the *quantization method* and which to the *hardware*.

## Questions, and the one factor each varies

| # | Question | Varies | Held fixed | Devices (Qualcomm AI Hub) |
|---|---|---|---|---|
| Q1 | Is the EfficientNet INT8 collapse a property of the quantization method rather than the hardware? | NPU generation | vendor, runtime (TFLite), model | Galaxy S21 (Snapdragon 888), S22 (8 Gen 1), S23 (8 Gen 2), S24 (8 Gen 3), S25 (8 Elite) |
| Q2 | Does Anneal's fix carry over to another vendor's quantizer and runtimes? | runtime | device (Galaxy S24) | TFLite vs QNN |
| Q3 | Is any of it specific to Qualcomm silicon? | chip vendor | runtime (TFLite) | Pixel 8 (Google Tensor G3), Galaxy A53 (Samsung Exynos 1280) |
| Q4 | Does it hold outside phones? | deployment domain | runtime | SA8775P ADP (automotive), QCS6490 / RB3 Gen 2 (IoT) |
| Q5 | What does INT8, and the fix, cost or gain in speed on an NPU? | variant | device | all of the above (AI Hub profile jobs) |

## Models

- **EfficientNet-B0**: SiLU with depthwise convs; collapses under per-tensor INT8 on every CPU tested; the fix under test.
- **MobileNetV3-Large**: Hardswish; a partial failure.
- **ResNet-50**: ReLU; the **control**. It should not collapse on any device. If it does, the pipeline is suspect before any finding is.

## Variants, per model and device

1. `fp32`: the float model compiled for the device. It is the reference for everything else on that device.
2. `hub int8`: Qualcomm's quantizer (W8A8, 64 calibration images) on the model as exported. This is what the standard flow ships.
3. `hub int8 + equalised`: the same quantizer on Anneal's equalised model, which is exact in float.
4. `anneal recipe`: Anneal's own QDQ model (equalise + asymmetric percentile + float stem), compiled as is.

Calibration uses 64 Imagenette train images. Scoring uses Imagenette validation images, 1000-way, paired against the device's own FP32 predictions (McNemar test, paired 95% CI). The sample is 512 images (about ±3pp) for Q1, Q3 and Q4, where the effects are expected to be tens of points, and 1,024 images for the fix comparisons on the Galaxy S24 and SA8775P (Q2, Q4), where the differences are 1–3pp. Licensed ImageNet images are not uploaded.

## Predictions

1. **The collapse is method-level.** `hub int8` EfficientNet-B0 is more than 20pp below `fp32` on every device and runtime, because the cause is one activation scale shared by channels of very different range, not any device's arithmetic.
2. **The fix transfers.** `hub int8 + equalised` recovers at least two thirds of that loss on every device, and `anneal recipe` lands within 3pp of `fp32` wherever it compiles.
3. **The control holds.** ResNet-50 `hub int8` is within 1.5pp of `fp32` on every device, because NPUs accumulate INT8 products in 32 bits, so the x86 16-bit saturation found on CPUs does not arise.
4. **INT8 pays on NPUs.** `hub int8` is at least 2x faster than `fp32` on every NPU, and equalisation costs under 15% of that speed.
5. **MobileNetV3 sits in between.** Its `hub int8` loss is smaller than EfficientNet's (under 15pp), and equalisation roughly halves it.

## Order and budget

Q1 and Q2 on EfficientNet-B0 run first, and ResNet-50 on the same devices as the control. Q3, Q4 and MobileNetV3 run only if the free AI Hub tier allows (about 300 cloud jobs in all).

## What would change the conclusions

- **The control collapses:** the pipeline (static shapes, calibration, compilation) is broken, and nothing else is interpreted until that is fixed.
- **FP32 accuracy differs across devices by more than noise:** the compiled float models are not equivalent, and cross-device comparisons use per-device FP32 only, which they already do.
- **A variant fails to compile or run on some device:** that is a result about the toolchain, and is reported rather than dropped.
