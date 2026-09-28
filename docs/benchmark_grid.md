# Benchmark grid: 8 models x 4 INT8 toolchains

Top-1 change vs FP32 in percentage points, Imagenette validation images, paired against the same
images in FP32 (onnxruntime, CPU). Each cell: the vendor's default INT8 → Anneal's recipe for that
target. Bold = within about 2pp of FP32.

| Model | AMD XINT8 (Quark) | Galaxy S24 NPU (QNN) | TI TDA4VM (TIDL) | NVIDIA T4 (TensorRT) |
|---|---|---|---|---|
| EfficientNet-B0 | −75.1 → **−0.1** | −12.8 → **−0.7** | −73.0 → **−1.6** ᵇ | −52.5 → **−0.1** ᵉ |
| EfficientNet-B1 | −75.8 → −4.1 ᵃ | −75.2 → **−1.7** ᶜ | −76.7 → −8.9 | −75.7 → −3.5 ᵉ |
| MobileNetV3-Small | −64.7 → **−1.7** | −58.0 → **−1.4** ᶜ | −65.8 → **−2.1** | −64.1 → **−1.8** |
| MobileNetV3-Large | −44.8 → **−1.0** | −2.4 → **−1.2** | −14.8 → **0.0** ᶠ | −9.9 → −2.2 |
| MobileNetV2 (ReLU6) | −7.9 → **−0.8** | **−0.4** (no fix needed) | −10.7 → **−1.2** | **−0.8** (no fix needed) |
| LCNet-100 | −67.0 → −2.9 | −34.4 → **−1.8** | −69.3 → −3.7 | −64.0 → **−1.2** |
| MobileViT-S | −76.7 → **−0.9** | −17.3 → **−0.3** | −71.7 → −4.2 ᵈ | −57.4 → **+0.6** |
| ResNet-50 (control) | −4.6 → **−0.2** | **+1.0** | **−1.9** → **−1.8** | **−1.1** → **−0.5** |

Images: AMD 1,000; S24 1,024; TIDL 1,500 (MobileViT 300 default / 500 Anneal); T4 1,000.
95% intervals are about ±1.5–3pp; every cell's JSON has them, with McNemar p-values.

## What each column is

| Target | Where it ran | Vendor default | Anneal's recipe |
|---|---|---|---|
| AMD XINT8 | **Emulated**: AMD Quark's XINT8 scheme (power-of-two scales, the Ryzen AI NPU's arithmetic) run by onnxruntime on a CPU | Quark XINT8 config | per-tensor gated equalisation with grid 1/s + FP32 surrogate for the gate + gates in 16 bits + measured bias correction; ReLU nets: CLE (max scale 4) + bias correction |
| Galaxy S24 | **Device**: Snapdragon 8 Gen 3 NPU via Qualcomm AI Hub, QNN runtime | Qualcomm's quantizer, W8A8 | equalisation (grid) before Qualcomm's quantizer; ᶜ Anneal's own quantization (onnxruntime QDQ, per-channel, uint8), compiled by QNN |
| TI TDA4VM | **Emulated**: TI's `tidl_tools` (J721E) host emulation, pinned commit; not a physical board | TIDL 8-bit, accuracy level 1 | per-tensor equalisation (grid) + 16-bit on the earliest layers (4–46 layers) via TIDL's own `output_feature_16bit_names_list`; MNv2/ResNet-50: CLE |
| NVIDIA T4 | **Device**: T4 GPU (Kaggle), TensorRT 10.16, batch 1 | NVIDIA ModelOpt INT8 (explicit QDQ) | equalisation, ModelOpt quantizing convolutions only; ᵉ Anneal's own QDQ with 4 tensors in float, built without FP16 |

## Footnotes

- ᵃ AMD B1: the same recipe scored −3.1, −4.1 and −4.4 in three runs (code revisions, run noise).
  The gate-conv variant scored −2.3.
- ᵇ TIDL B0: noise-optimal scales (`equalize_derived`) + 16-bit features 0-2; with the default
  grid scales −2.7, pure 8-bit −4.0.
- ᵈ TIDL MobileViT: TIDL's own 16-bit mode is better here (−2.6).
- ᶠ TIDL MNv3-L: equalisation (grid) + TIDL's automatic mixed precision; pure 8-bit with equalisation −3.3.
- The S24 numbers for B0/B1 in early commits were measured against the phone's own FP16 run. On
  B1 that run is degraded: 69.2% vs 76.1% true FP32 (−6.8pp). This table uses true FP32.

## Speed

| Target | Anneal vs vendor INT8 latency | vs FP16 |
|---|---|---|
| S24 (per inference) | B0 0.422 → 0.538 ms; B1 0.559 → 0.893; MNv3-S 0.201 → 0.26 | faster: B0 0.838, B1 1.196, MNv3-S 0.291 ms |
| T4 (batch 1) | B1 2.56–2.62 ms (explicit QDQ) | **slower** than TensorRT FP16 (B1 1.32 ms, −0.1pp). At batch 1 on a T4, INT8 buys no speed for these models: Anneal recovers accuracy, FP16 remains the better T4 choice |
| AMD, TIDL | emulated: no latency measured | — |

## Things found on the way

- Intel OpenVINO / NNCF (ImageNet 5,000, non-VNNI CPU, real kernels) is the one toolchain where the
  gated models do not collapse: B0 −1.8pp, B1 −4.9 by default, and equalisation does not help
  (−1.9 / −5.4). Its MobileNetV3-Large loss (−67pp) is 16-bit overflow, fixed by NNCF's own
  overflow fix ([examples/openvino](../examples/openvino/)). Not part of the grid: no gated-model
  fix is needed there.

- TensorRT's implicit INT8 (FP16 fallback, deprecated) is fine on MobileNetV3/LCNet/ResNet-50 but
  loses 9.6 (B0), 14.8 (B1), 6.0 (MobileViT) and 69 (EfficientViT-B0) pp, and its builds are not
  deterministic.
- TensorRT: tensors kept in float next to QDQ INT8 overflow in FP16 (B0 −75.1pp); built without
  FP16 the same engine scores −0.1pp (+10% latency).
- QNN: Anneal's QDQ with int8 activations, or with uint16 tensors mixed in, compiles and runs but
  scores 0%; uint8 activations are faithful.
- TIDL: pre-quantized QDQ import is faithful only for plain convnets (ResNet-18); depthwise
  degrades, depthwise + ReLU6 is garbage. Accuracy level 9 is worse everywhere tested.
- TIDL segmentation (LRASPP-MobileNetV3, 300 images): −44.7 → −1.2 mIoU (equalisation + 16-bit
  on 4 backbone layers).
- Noise-optimal per-channel scales (`equalize_derived`): +26.7pp on TIDL B1 and +7.2pp on B0 in pure
  8-bit, +1.1pp on TIDL B0 with 16-bit early layers, but −3.0pp on AMD B1 and a tie on AMD/TIDL B1
  with 16-bit layers; opt-in.

## Sources

Results JSON: `examples/vitis/results/`, `examples/qaihub/results/`, `examples/tidl/results/`,
`examples/tensorrt/results/`. Scripts alongside. The run ledger with every intermediate result and
dead end is in [findings.md](findings.md).
