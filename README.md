# Anneal

**Measured, hardware-aware INT8 for edge deployment.** Anneal tells you how to quantize a model
for the chip it will run on, and proves the answer with paired statistics on real data.

Its central finding: standard INT8 quantization breaks the gated-depthwise networks used on
edge devices (EfficientNet, MobileNetV3, LCNet, MobileViT) on every CPU and on the default INT8
path of all four vendor toolchains tested (AMD, Qualcomm, TI, NVIDIA): losses of 35–77 points
are common. The cause is one activation scale shared by channels whose ranges differ by orders of
magnitude. Anneal's fix, an exact channel equalisation through SiLU/Hardswish gates, needs no
retraining; with a per-target recipe it brings most model/toolchain pairs within about two
points of FP32.

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

Changes here are against each device's own float run (on the S24 vs true FP32: −12.8 → −1.5).
No remaining loss is statistically significant. ResNet-50, the control, loses nothing where
scored (S24, Pixel 8). ² 256-image subset; the device's jobs failed intermittently. The
predictions were committed before the runs and are graded, including the ones that failed:
[design](docs/edge_study_design.md), [results](docs/edge_study_results.md).

**Four vendor toolchains** (8 models; Imagenette, 1,000–1,500 images; vendor default INT8 →
Anneal's recipe for that target; [full grid, recipes and footnotes](docs/benchmark_grid.md)).
S24 and T4 are real devices; AMD and TI are the vendors' own quantizers and emulators on a PC.

| Model | AMD XINT8 (emulated) | Galaxy S24 NPU | TI TDA4VM (emulated) | NVIDIA T4 TensorRT |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | −75.1 → **−0.1** | −12.8 → **−0.7** | −73.0 → **−1.6** | −52.5 → **−0.1** |
| EfficientNet-B1 | −75.8 → −4.1 | −75.2 → **−1.7** | −76.7 → −8.9 | −75.7 → −3.5 |
| MobileNetV3-Small | −64.7 → **−1.7** | −58.0 → **−1.4** | −65.8 → **−2.1** | −64.1 → **−1.8** |
| MobileNetV3-Large | −44.8 → **−1.0** | −2.4 → **−1.2** | −14.8 → **0.0** | −9.9 → −2.2 |
| MobileNetV2 | −7.9 → **−0.8** | **−0.4** | −10.7 → **−1.2** | **−0.8** |
| LCNet-100 | −67.0 → −2.9 | −34.4 → **−1.8** | −69.3 → −3.7 | −64.0 → **−1.2** |
| MobileViT-S | −76.7 → **−0.9** | −17.3 → **−0.3** | −71.7 → −4.2 | −57.4 → **+0.6** |
| ResNet-50 (control) | −4.6 → **−0.2** | **+1.0** | **−1.9** → **−1.8** | **−1.1** → **−0.5** |

The recipes differ by target: TIDL adds 16-bit on the first few layers (TIDL's own option), AMD
keeps the gates in 16 bits, and on the S24 Anneal's own INT8 model (compiled by Qualcomm's QNN)
beats Qualcomm's quantizer on B1 and MobileNetV3-Small.

**Beyond classification** (COCO val2017, 500 images, box mAP change in points, 32-bit;
[details](examples/tasks/)): neither detector collapses, so equalisation is not needed there.

| Detector (FP32 mAP) | onnxruntime default | percentile | Anneal |
|---|---:|---:|---:|
| SSDLite-MobileNetV3 (23.8) | −2.59 | −0.60 | −0.81 |
| YOLOv8n (40.5) | −0.89 | −0.94 | −0.52 |

`equalize_min_gain` skips equalisation when its predicted gain is small, as here.
Segmentation (LRASPP-MobileNetV3, 300 images, TI TDA4VM emulation): TIDL 8-bit −44.7 mIoU →
**−1.2** with equalisation and 16 bits on four backbone layers.

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

- **Speed.** Equalisation adds gate multiplies: on the S24 NPU, B0 0.422 → 0.538 ms and B1
  0.559 → 0.893 ms (Anneal's own INT8), still faster than FP16 (0.838 / 1.196 ms). On a T4 at
  batch 1, every INT8 engine tested (vendor's or Anneal's) is slower than TensorRT FP16 for these
  small models: Anneal recovers INT8 accuracy there, but FP16 remains the better T4 choice.
- **Not solved everywhere.** TI TDA4VM: B1 −8.9, MobileViT −4.2 (TIDL's own 16-bit
  mode: −2.6); EfficientViT-B0 on TensorRT −70 → −12.8.
- **Emulated targets.** AMD and TI numbers come from the vendors' quantizers and bit-level
  emulators on a PC, not from boards; only the S24 and T4 numbers are measured on hardware.
- **Recipe selection.** Each target's recipe was chosen by comparing variants on the same
  Imagenette images it is reported on (1,000–1,500 per model), so the best cells carry some
  selection optimism; the results JSONs record every variant tried, not only the best.
- QNN runs Anneal's INT8 models faithfully only with uint8 activations; int8 activations or
  mixed-in uint16 tensors compile but score 0%. TIDL's pre-quantized QDQ import is faithful only
  for plain convnets.
- Edge results use Imagenette (256–1,500 images, a public ImageNet subset), not ImageNet;
  the SA8775P was flaky and the Pixel 8 ran on its GPU, not its NPU.
- onnxruntime's entropy calibration is not TensorRT's; entropy comparisons cite NVIDIA's
  published numbers ([details](examples/imagenette_entropy/README.md)).

## More

- [Benchmark grid](docs/benchmark_grid.md): 8 models x 4 toolchains, recipes, speed, footnotes
- [The full record](docs/findings.md): every experiment, in the order it was found, corrections included
- [Literature review](docs/literature_review.md): what is known, and what is new here
- Data and scripts: [ImageNet](examples/imagenet/), [edge devices](examples/qaihub/),
  [detection](examples/tasks/), [advisor](examples/advise/). MIT licence.
