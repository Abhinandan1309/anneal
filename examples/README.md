# Examples: scripts and committed results

Each folder holds the script that produced a result and the result itself (JSON, paired against
FP32 with 95% intervals and McNemar p-values). `*.onnx` models are not committed; the scripts
export them. Headline numbers: [README](../README.md), [benchmark grid](../docs/benchmark_grid.md),
[full record](../docs/findings.md).

| Folder | What it is | Where results were measured |
|---|---|---|
| [imagenet/](imagenet/) | Five models on ImageNet validation (10,000–49,000 images) | onnxruntime, CPU |
| [qaihub/](qaihub/) | Qualcomm AI Hub: Galaxy S24, SA8775P, Pixel 8 | devices |
| [tidl/](tidl/) | TI TDA4VM: classifiers, LRASPP, YOLOv8n, pre-quantized import | TI's emulator |
| [tensorrt/](tensorrt/) | NVIDIA T4: TensorRT implicit INT8, ModelOpt, Anneal's QDQ | device (Kaggle) |
| [vitis/](vitis/) | AMD Quark XINT8 | AMD's quantizer, onnxruntime on CPU |
| [openvino/](openvino/) | Intel OpenVINO / NNCF on a non-VNNI Ryzen CPU, ImageNet 5,000: equalisation does not help (B0 −1.8 → −1.9pp); MobileNetV3-L's −67pp is overflow, fixed by NNCF's overflow fix (−2.4) | CPU (real kernels) |
| [tasks/](tasks/) | COCO detection (SSDLite, YOLOv8n) and LRASPP segmentation | onnxruntime, CPU |
| [advise/](advise/) | The advisor's evidence: recipe ablations, site and tensor sensitivity | onnxruntime, CPU |
| [hardware_lab/](hardware_lab/) | The static INT8 recipe study on GitHub-hosted x86, ARM64, Windows and macOS machines: the 16-bit saturation finding | cloud CPUs |
| [imagenette_entropy/](imagenette_entropy/) | onnxruntime's entropy calibration, fixed and rerun | onnxruntime, CPU |
| [olive_resnet18/](olive_resnet18/) | Auditing Microsoft Olive's quantized ResNet-18 with `anneal audit` | CPU |
| [equalize/](equalize/) | Early equalisation studies: full-validation runs, latency, imbalance, causal mechanism tests, the refuted bias hypothesis | onnxruntime, CPU |
| [segmentation/](segmentation/) | LRASPP bisection and recipe screens; a UNet (Carvana) study | emulation |
| [saturation/](saturation/) | The overflow causal test and guard generalisation | CPU |
| [zoo/](zoo/), [zoo_gated/](zoo_gated/) | The 11-model zoo run; exporter for the gated models (timm and torchvision) | onnxruntime, CPU |
| [sequential_study/](sequential_study/) | Simulation and replay study of sequential acceptance testing | offline |
| [resnet18-cpu1t/](resnet18-cpu1t/), [resnet18-cpu1t-v2/](resnet18-cpu1t-v2/), [mobilenetv3-cpu1t/](mobilenetv3-cpu1t/) | `anneal run` outputs: ledger, report, Pareto frontier | CPU, 1 thread |
| [tflite_lab/](tflite_lab/) | TensorFlow Lite converter path (GitHub Actions) | CPU |
