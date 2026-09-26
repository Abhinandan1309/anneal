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
accumulation as on ARM and NPUs). Accuracy change vs FP32:

| Model | onnxruntime default | best standard calibration | **Anneal** |
|---|---:|---:|---:|
| EfficientNet-B0 (49,000 images) | −45.5pp | −6.3pp | **−0.52pp** |
| MobileNetV3-Large (49,000) | −5.4pp | −2.7pp | **−1.01pp** |
| ViT-B/16 (10,000) | −6.7pp | −6.7pp | **−0.73pp** |
| ConvNeXt-Tiny (10,000) | −0.9pp | −0.9pp | **−0.53pp** |
| ResNet-50 (10,000) | −0.2pp | **−0.1pp** | −0.54pp ¹ |

Published post-training results for EfficientNet-B0: −4.8pp (NVIDIA, entropy), −3.0pp (HPTQ).
¹ The advisor's recipe lost to plain percentile calibration here; the rule for plain ReLU
networks is being corrected.

**Real edge devices** (Qualcomm AI Hub, Qualcomm's own quantizer, EfficientNet-B0):

| Device | Runtime | Qualcomm INT8 | + Anneal equalisation |
|---|---|---:|---:|
| Galaxy S24 (Snapdragon 8 Gen 3 NPU) | QNN | −12.0pp | **−0.7pp** |
| Galaxy S24 | TFLite | −11.5pp | **−1.5pp** |
| SA8775P (automotive NPU) | TFLite | −13.3pp | **−2.0pp** |
| Pixel 8 (Tensor G3, GPU) | TFLite | −12.7pp | **−1.2pp** |

No remaining loss is statistically significant. ResNet-50, the control, loses nothing on any
device. The predictions were committed before the runs and are graded, including the ones that
failed: [design](docs/edge_study_design.md), [results](docs/edge_study_results.md).

**A second finding, on x86.** CPUs without VNNI sum INT8 products in 16 bits, and the overflow
costs 8–18pp across nine CNNs; Anneal predicts it per layer without the affected CPU.

## Use it

```bash
git clone https://github.com/Abhinandan1309/anneal && cd anneal
pip install -e ".[torch]"

anneal advise model.onnx --verify      # recommended INT8 recipe for this model and CPU, measured
anneal run --model torchvision:resnet18 --target cpu-1t --eval imagenette --budget 10
```

`advise` reads the architecture from the graph and the CPU's INT8 arithmetic, recommends a
recipe with its evidence, and `--verify` scores it against the alternatives with a McNemar
test. `run` is the full search: it proposes transforms, measures each on the real runtime and
returns a Pareto frontier over latency, accuracy and size, with a ledger of every trial.
Other commands: `audit`, `saturation`, `imbalance`, `profile`, `validate`, `export`
(`anneal --help`).

## Limitations

- Edge results use Imagenette (512–1,024 images, a public ImageNet subset), not ImageNet.
- Equalisation adds gate multiplies: about 30% of plain INT8's speed on the S24 NPU (INT8 is
  still 1.6x faster than FP32).
- Anneal's full recipe (equalise + percentile + float stem) compiled for the S24 but did not
  run on it; only equalisation has been shown on devices.
- onnxruntime's entropy calibration is not TensorRT's; entropy comparisons cite NVIDIA's
  published numbers ([details](examples/imagenette_entropy/README.md)).

## More

- [The full record](docs/findings.md): every experiment, in the order it was found, corrections included
- [Literature review](docs/literature_review.md): what is known, and what is new here
- Data and scripts: [ImageNet](examples/imagenet/), [edge devices](examples/qaihub/), [advisor](examples/advise/)

MIT licence.
