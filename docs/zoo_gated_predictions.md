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
