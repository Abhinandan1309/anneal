# ImageNet: Anneal against standard post-training calibration

**Data.** ImageNet-1k validation set, the Hugging Face parquet export (licensed; accept the
terms at huggingface.co/datasets/ILSVRC/imagenet-1k and download `data/validation-*.parquet`).
Every 50th image (1,000) is held out for calibration and 64 of those are used; the other 49,000
are scored, 1000-way. EfficientNet-B0 and MobileNetV3-Large are scored on all 49,000; ResNet-50,
ConvNeXt-Tiny and ViT-B/16 on the first 10,000 (a prefix of the shuffled files; ~±0.6pp).

**Recipes.** All built by the same pipeline (onnxruntime static INT8, QDQ, U8S8, per-channel
weights): `minmax` (onnxruntime's default), `entropy`, symmetric `percentile` 99.999, and the
recipe `anneal advise` recommends. Histogram calibrators (entropy, percentile) need too much
memory on the transformer and ConvNeXt, which get minmax only.

**Modes.** *Emulated*: the QDQ graph run in float with graph optimisations off, as a
32-bit-accumulating CPU or NPU computes it; comparable to published simulated-quantization
results. *Fused*: onnxruntime's real kernels on this laptop (x86 without VNNI, 16-bit pair
sums), with the recipe advised for that CPU.

**Results** (change vs FP32 in pp, emulated unless marked; [results/](results/)):

| Model (n) | FP32 | minmax | entropy | percentile | Anneal (32-bit) | Anneal, fused | minmax, fused |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 (49,000) | 77.62 | −45.46 | −6.33 | −11.20 | **−0.52** | −1.26 | −52.63 |
| MobileNetV3-Large (49,000) | 75.29 | −5.37 | −7.20 | −2.69 | **−1.01** | −1.50 | −12.81 |
| ResNet-50 (10,000) | 81.00 | −0.19 | −23.39 | −0.10 | −0.54 ¹ | −0.75 ¹ | −8.28 |
| ConvNeXt-Tiny (10,000) | 82.43 | −0.87 | – | – | **−0.53** | −0.55 | −2.67 |
| ViT-B/16 (10,000) | 81.68 | −6.73 | – | – | **−0.73** | −1.57 | −6.85 |

- Anneal's recipes: equalise + asymmetric percentile + float stem (EfficientNet, MobileNetV3;
  + `reduce_range` fused on EfficientNet), asymmetric percentile + float stem (ResNet-50,
  + `reduce_range` fused; ConvNeXt, + `reduce_range` in both modes), minmax on the compute ops
  only (ViT). EfficientNet-B0's −0.52pp is significant
  (McNemar p = 3.9e-10); it is not lossless.
- `entropy` was re-scored after the entropy-calibration fix (EfficientNet-B0 is marked
  `"redone"`); it is onnxruntime's entropy, not TensorRT's, and over-clips ResNet-50.
- Dense-consumer equalisation on top of the advice: no gain (EfficientNet-B0 −0.52,
  MobileNetV3 −1.05, ConvNeXt −0.55pp).
- ¹ The old ReLU-CNN advice. The corrected advice scores −0.14pp emulated, −0.04pp fused
  ([ablation](../advise/README.md)).

**Run.** `python examples/imagenet/run_imagenet.py --models efficientnet_b0,resnet50`. One data
pass per model feeds FP32 and every recipe the same batches; per-image predictions go to
`results/<model>-predictions.npz`, so every statistic can be recomputed. Finished models are
skipped. `--redo entropy` re-scores the named recipes on a finished model and merges them in,
after checking the stored FP32 predictions are reproduced exactly.
