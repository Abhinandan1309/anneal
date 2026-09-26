# The advisor: verification, timing and ablations

`anneal advise model.onnx --verify` recommends a recipe from the model's family and the target's
INT8 path, then scores it, its alternatives and onnxruntime's default against FP32 (paired,
McNemar). `<model>-<int8 path>.json` are those runs (Imagenette, 512–1,024 images);
`retime.py` re-times them on this laptop (AC power, 1 thread, three interleaved rounds) into
`*-timing.json`. Change vs FP32 in pp:

| Model, target (mode, n) | Recommended | Measured | onnxruntime default | Speed vs FP32 (advised / default) |
|---|---|---:|---:|---|
| EfficientNet-B0, x86 no VNNI (fused, 1,024) | equalise + percentile + stem + reduce_range | −0.49 (p 0.66), best of 4 | −51.17 | 1.04x / 1.12x |
| ResNet-50, x86 no VNNI (fused, 1,024) | percentile + stem + reduce_range ¹ | −0.10 (p 1) | −13.09 | 1.05x / 1.13x |
| MobileNetV3-L, ARM (emulated, 1,024) | equalise + percentile + stem | −2.05; percentile + stem −1.86 (noise) | −12.40 | – |
| ViT-B/16, x86 VNNI (emulated, 512) | quantize compute ops only | +0.20 (p 1) | −8.40 | – |

¹ Old advice, since corrected by the ablation below.

**ResNet-50 ablation** (`resnet50_ablation.py` → `resnet50_ablation.json`; ImageNet, the same
10,000 images as [../imagenet](../imagenet/README.md)). On ImageNet the old advice lost 0.44pp
to plain symmetric 99.999 percentile (p = 0.003). Separated:

| Recipe | Emulated (32-bit) | Fused (x86, no VNNI) |
|---|---:|---:|
| minmax (onnxruntime default) | −0.19 | −8.28 |
| old advice: asym 99.99 + stem (fused: + reduce_range) | −0.54 | −0.75 |
| asym 99.99 | −0.37 | |
| asym 99.999 + stem | −0.35 | −0.19 |
| sym 99.999 (no stem) | −0.10 | |
| **new advice: sym 99.999 + stem** | **−0.14** | **−0.04** |
| sym 99.999 + reduce_range | | −0.63 |

The 99.99 percentile over-clips and `reduce_range` costs ~0.5pp (p = 0.005) once the float stem
has removed the saturation, so plain CNNs are now advised symmetric 99.999 + float stem, no
`reduce_range` (commit 906393b).

**EfficientNet-B0 recipe ablation** (`recipe_ablation.py` → `efficientnet_b0_recipe_ablation.json`;
ImageNet, first 10,000 scored images, emulated; FP32 78.29%):

| advised: eq + asym 99.99 + stem | eq + sym 99.999 + stem | eq + asym 99.999 + stem | no stem | minmax, not percentile | no equalisation | default |
|---:|---:|---:|---:|---:|---:|---:|
| **−0.38** | −0.83 | −1.01 | −1.31 | −2.88 | −3.64 | −46.48 |

Every alternative is significantly worse than the advice (p ≤ 0.023). Equalisation is worth
~3.3pp, percentile ~2.5pp and the float stem ~0.9pp; the best percentile is family-dependent
(99.999 for ReLU CNNs, 99.99 for gated-depthwise). **MobileNetV3-Large: pending** (running).
