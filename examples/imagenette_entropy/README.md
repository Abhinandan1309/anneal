# Entropy calibration: a baseline that was not what it said

**What was wrong.** onnxruntime's `quantize_static` cannot pass histogram sizes to its entropy
calibrator, which then runs with 128 histogram bins folded to 128 quantized bins. With nothing to
fold, the KL search returns the min/max range: every "entropy" result produced this way was min/max
under another name. On full ImageNet (EfficientNet-B0, 49,000 images) the two gave identical
predictions on every image; in the agent ledgers they gave identical accuracy.

**The fix.** `anneal.core.transforms` now passes TensorRT's KL setting, 2048 bins folded to 128
(`ENTROPY_NUM_BINS`, `ENTROPY_NUM_QUANTIZED_BINS`), with a regression test.

**Re-scored** (`rerun_entropy.py`, original parameters and sample sizes, [results.json](results.json)):

| Case | n | FP32 | minmax (old) | entropy, old = minmax | entropy, fixed |
|---|---:|---:|---:|---:|---:|
| ResNet-18 agent run ([findings](../../docs/findings.md) table rows 3, 9) | 256 | 66.8% | 64.5% (64.1%) | 64.1% | **44.9%** |
| MobileNetV3-Large agent run | 256 | 71.5% | 44.1% (42.2%) | 42.2% | **46.5%** |
| EfficientNet-B0 recipe sweep | 512 | 74.2% | 22.5% (22.5%) | 39.6%* | **58.8%** |

\* The sweep calibrated in chunks, so its broken "entropy" differed from its min/max.

Min/max reproduces the old numbers exactly for EfficientNet-B0 and to within 1 and 5 of 256 images
for the other two (onnxruntime was upgraded in between).

**A second caveat.** Even with the fix, onnxruntime's entropy calibration is not TensorRT's. On
ResNet-18 it clips most ReLU outputs to exactly its smallest candidate threshold (12.6% of the
min/max range), which costs 20pp where NVIDIA reports entropy costing about 0.1pp on ResNets.
Our entropy rows are therefore labelled *onnxruntime entropy*; comparisons with entropy
calibration as practised (TensorRT) use NVIDIA's published numbers (Wu et al. 2020), not ours.
