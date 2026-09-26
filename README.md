# Anneal

**Measured, hardware-aware INT8 for edge deployment.** Anneal tells you how to quantize a model
for the chip it will run on, and proves the answer with paired statistics on real data.

Its central finding: standard INT8 quantization breaks the gated-depthwise networks used on
edge devices (EfficientNet, MobileNetV3), on every CPU and on real phone and automotive NPUs.
The cause is one activation scale shared by channels whose ranges differ by orders of
magnitude. Anneal's fix, an exact channel equalisation through SiLU/Hardswish gates, needs no
retraining and brings the loss down to about half a point.

## Results

**ImageNet** (validation set; 64 held-out images calibrate, the rest are scored; 32-bit
accumulation as on ARM and NPUs; [details](examples/imagenet/)). Accuracy change vs FP32:

| Model | onnxruntime default | best standard calibration | **Anneal** |
|---|---:|---:|---:|
| EfficientNet-B0 (49,000 images) | −45.5pp | −6.3pp | **−0.52pp** |
| MobileNetV3-Large (49,000) | −5.4pp | −2.7pp | **−1.01pp** |
| ViT-B/16 (10,000) | −6.7pp | −6.7pp | **−0.73pp** |
| ConvNeXt-Tiny (10,000) | −0.9pp | −0.9pp | **−0.53pp** |
| ResNet-50 (10,000) | −0.2pp | −0.1pp | −0.14pp ¹ |

Published post-training results for EfficientNet-B0: −4.8pp (NVIDIA, entropy), −3.0pp (HPTQ).
Anneal's −0.52pp is small but statistically significant, not lossless. ¹ Corrected advice:
the first rule for ReLU CNNs lost 0.44pp to plain percentile; an [ablation](examples/advise/)
traced it to over-clipping and `reduce_range`, and the rule was changed (−0.04pp on real
non-VNNI x86 kernels). All three are within noise.

**Real edge devices** (Qualcomm AI Hub, Qualcomm's own quantizer, EfficientNet-B0,
Imagenette; [details](examples/qaihub/)):

| Device | Runtime | Qualcomm INT8 | + Anneal equalisation |
|---|---|---:|---:|
| Galaxy S24 (Snapdragon 8 Gen 3 NPU) | QNN | −12.0pp | **−0.7pp** |
| Galaxy S24 | TFLite | −11.5pp | **−1.5pp** |
| SA8775P (automotive NPU) ² | TFLite | −13.3pp | **−2.0pp** |
| Pixel 8 (Tensor G3, GPU) | TFLite | −12.7pp | **−1.2pp** |

No remaining loss is statistically significant. ResNet-50, the control, loses nothing where
scored (S24, Pixel 8). ² 256-image subset; the device's jobs failed intermittently. The
predictions were committed before the runs and are graded, including the ones that failed:
[design](docs/edge_study_design.md), [results](docs/edge_study_results.md).

**Beyond classification** (COCO val2017, 500 images, box mAP change in points, 32-bit;
[details](examples/tasks/)): neither detector collapses, so equalisation is not needed there.

| Detector (FP32 mAP) | onnxruntime default | percentile | Anneal |
|---|---:|---:|---:|
| SSDLite-MobileNetV3 (23.8) | −2.59 | −0.60 | −0.81 |
| YOLOv8n (40.5) | −0.89 | −0.94 | −0.52 |

`equalize_min_gain` skips equalisation when its predicted gain is small, as here.
Segmentation (LRASPP) is pending.

**A second finding, on x86.** CPUs without VNNI sum INT8 products in 16 bits, and the overflow
costs 8–18pp across nine CNNs; Anneal predicts it per layer without the affected CPU.

## Use it

```bash
git clone https://github.com/Abhinandan1309/anneal && cd anneal
pip install -e ".[torch]"

anneal advise model.onnx --verify      # recommended INT8 recipe for this model and CPU, measured
anneal run --model torchvision:resnet18 --target cpu-1t --eval imagenette --budget 10
```

`advise` reads the architecture and the CPU's INT8 arithmetic, recommends a recipe with its
evidence, and `--verify` scores it against the alternatives (McNemar test). `run` is the full
search: measured transforms, a Pareto frontier and a ledger of every trial. Also: `audit`,
`saturation`, `imbalance`, `profile`, `validate`, `export` (`anneal --help`).

## Limitations and open problems

- **Speed.** Equalisation adds gate multiplies: on the S24 NPU equalised INT8 takes 27–32% longer
  than plain INT8 (e.g. 0.425 → 0.541 ms; still ~1.6x faster than FP32). Equalising only the sites
  with the highest predicted gain failed (8 of 16 sites: −6.4pp), so the per-site prediction
  does not rank a site's value on the device. Per-site measurement is running.
- Anneal's full recipe (equalise + percentile + float stem) compiled for the S24 but did not
  run on it; only equalisation has been shown on devices. The cause is being bisected.
- Edge results use Imagenette (256–1,024 images, a public ImageNet subset), not ImageNet;
  the SA8775P was flaky and the Pixel 8 ran on its GPU, not its NPU.
- onnxruntime's entropy calibration is not TensorRT's; entropy comparisons cite NVIDIA's
  published numbers ([details](examples/imagenette_entropy/README.md)).

## More

- [The full record](docs/findings.md): every experiment, in the order it was found, corrections included
- [Literature review](docs/literature_review.md): what is known, and what is new here
- Data and scripts: [ImageNet](examples/imagenet/), [edge devices](examples/qaihub/),
  [detection](examples/tasks/), [advisor](examples/advise/). MIT licence.
