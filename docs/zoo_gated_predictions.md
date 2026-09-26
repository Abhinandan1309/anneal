# More gated-depthwise models: predictions (written before the runs)

Committed before any accuracy on these models exists, as for the edge study. The claim under test:
the summed predicted equalisation gain (`rank_sites`, 8 Imagenette images; values in
[predicted_gain_totals.json](../examples/tasks/results/predicted_gain_totals.json)) says in advance whether
per-tensor INT8 collapses and whether equalisation is needed.

| Model | Source | Predicted gain | Advisor family |
|---|---|---:|---|
| FBNetV3-B | timm | 165.5 | gated-depthwise |
| EfficientNet-B3 | torchvision | 153.5 | gated-depthwise |
| LCNet-100 | timm | 69.8 | gated-depthwise |
| EfficientNet-B2 | torchvision | 64.1 | gated-depthwise |
| EfficientViT-B1 | timm | 58.5 | convnext |
| MobileViT-S | timm | 23.4 | convnext |
| EfficientViT-B0 | timm | 18.7 | convnext |
| MobileNetV3-Small | torchvision | 5.2 | gated-depthwise |

Reference points already measured on ImageNet: gain 2.9 (SSDLite, COCO) no collapse; 58-366 for the
classifiers that collapse (MobileNetV3-Large 58: default -5.4pp; EfficientNet-B0 73: -45.5pp; B1 91: -79pp).

**Predictions** (ImageNet validation, 5,000 scored images, emulated 32-bit accumulation, 64 held-out
calibration images; recipes: onnxruntime default min/max, asymmetric 99.99 percentile + float stem
without equalisation, the same with equalisation, and that plus `int16_top_k=3`):
1. Gain >= 50 (FBNetV3-B, B3, LCNet, B2, EfficientViT-B1): the default loses more than 5pp, and
   equalisation recovers at least two thirds of the loss left by percentile + stem alone.
2. Gain < 10 (MobileNetV3-Small): the default loses less than 5pp and equalisation changes accuracy by
   less than 1pp.
3. Gain 10-50 (MobileViT-S, EfficientViT-B0): no prediction on collapse (the threshold is not
   established there); reported as the data that places it.
4. Where the advisor's family is `convnext` (EfficientViT, MobileViT) but equalisation helps by more
   than 1pp, the advisor's family rule is wrong for these models and is corrected.

## Results (graded 2026-09-27; predictions above unchanged)

ImageNet validation, 5,000 images, emulated 32-bit accumulation. "Left" is the loss of percentile + stem
without equalisation; "recovered" the share of it equalisation removes. `joint_damage` (added after
prediction 2 failed, before the other runs finished): share of 64 unlabelled calibration images whose
top-1 flips when every equalisation site's tensors are fake-quantized at once, on the plain model.

| Model | Predicted gain | joint_damage | Default | Left (no eq) | With eq | Recovered | Prediction |
|---|---:|---:|---:|---:|---:|---:|---|
| FBNetV3-B | 165.5 | 0.14 | -42.66 | -3.26 | -1.60 | 51% | 1: collapse yes, recovery < 2/3 -> **fail** |
| EfficientNet-B3 | 153.5 | 0.05 | -4.28 | +0.06 | +0.26 | n/a | 1: no collapse -> **fail** |
| LCNet-100 | 69.8 | 0.70 | -72.10 | -53.24 | -2.68 | 95% | 1: **pass** |
| EfficientNet-B2 | 64.1 | 0.13 | -5.00 | -1.56 | -0.78 | 50% | 1: not > 5pp, recovery < 2/3 -> **fail** |
| EfficientViT-B1 | 58.5 | 0.78 | -80.20 | -80.18 | -80.14 | 0% | 1: collapse yes, equalisation alone no -> **fail** |
| MobileViT-S | 23.4 | 0.53 | -56.54 | -10.22 | -0.82 | 92% | 3: collapses |
| EfficientViT-B0 | 18.7 | 0.91 | -72.3 | -72.1 | -72.1 | 0% | 3: collapses |
| MobileNetV3-Small | 5.2 | 0.97 | -66.10 | -60.40 | -0.90 | 99% | 2: **fail** (collapses; equalisation +59.5pp) |

**Verdict.** The summed predicted gain does not predict collapse: 5.2 collapsed, 153.5 did not; prediction
1 passed on one model of five and prediction 2 failed. `joint_damage` orders the equalisation benefit
almost monotonically (B1 1.00 -> +61pp, MobileNetV3-Small 0.97 -> +59.5, LCNet 0.70 -> +50.6,
MobileViT-S 0.53 -> +9.4, B0 0.55 -> +3.3, FBNetV3-B 0.14 -> +1.7, B2 0.13 -> +0.8, B3 0.05 -> none);
a threshold near 0.3 separates the two groups. EfficientViT fails differently (its attention; its own
recipe reaches -7.0pp on Imagenette, [efficientvit_fix.py](../examples/advise/efficientvit_fix.py)).

**Prediction 4 triggers:** equalisation helps MobileViT-S by 9.4pp, but the advisor files it under
`convnext` (no equalisation); the family rule is corrected.
