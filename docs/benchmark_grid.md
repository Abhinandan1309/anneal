# Benchmark grid: 8 models x 4 INT8 toolchains

Top-1 change vs FP32 in percentage points, Imagenette validation images, paired against the same
images in FP32 (onnxruntime, CPU). Each cell: the vendor's default INT8 → Anneal's recipe for that
target. Bold = within about 2pp of FP32.

| Model | AMD XINT8 (Quark) | Galaxy S24 NPU (QNN) | TI TDA4VM (TIDL) | NVIDIA T4 (TensorRT) |
|---|---|---|---|---|
| EfficientNet-B0 | −75.1 → **−0.1** | −12.8 → −2.6 ʰ | −73.0 → **−1.6** ᵇ | −52.5 → **−0.1** ᵉ |
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
- TIDL B1: TIDL's full 16-bit mode loses only 0.9pp, so the −8.9 is the cost of keeping most of
  the network at 8 bits, not a TIDL limit. TIDL's automatic mixed precision did not help: with
  equalisation it did not finish in 3 h; without, it crashed in TIDL's runtime (`b1_automixed_n1500.json`).
- ᵈ TIDL MobileViT: TIDL's own 16-bit mode is better here (−2.6).
- ᶠ TIDL MNv3-L: equalisation (grid) + TIDL's automatic mixed precision; pure 8-bit with equalisation −3.3.
- ʰ S24 B0: −0.7 when measured on 2026-09-27; re-running the identical model on 2026-10-03 gives
  −2.6 on every QAIRT version (see Checks below), likely a change in AI Hub's quantizer.
- The S24 numbers for B0/B1 in early commits were measured against the phone's own FP16 run. On
  B1 that run is degraded: 69.2% vs 76.1% true FP32 (−6.8pp). This table uses true FP32.

## Speed

| Target | Anneal vs vendor INT8 latency | vs FP16 |
|---|---|---|
| S24 (per inference) | B0 0.422 → 0.538 ms; B1 0.559 → 0.893; MNv3-S 0.201 → 0.26 | faster: B0 0.838, B1 1.196, MNv3-S 0.291 ms |
| T4 (batch 1) | B1 2.56–2.62 ms (explicit QDQ) | **slower** than TensorRT FP16 (B1 1.32 ms, −0.1pp). At batch 1 on a T4, INT8 buys no speed for these models: Anneal recovers accuracy, FP16 remains the better T4 choice |
| TIDL (TDA4VM, TI perf simulator) | LRASPP: 8-bit 12.0 ms, Anneal (eq + 16-bit backbone stages 0-1) 13.8 ms (+14%) | full 16-bit 21.5 ms (+78%). Simulator estimate, not a board measurement |
| AMD | emulated: no latency measured | — |

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

## Checks: toolchain version and holdout images (2026-10-03)

**QAIRT version (S24).** Each model was quantized once, then compiled and run with every QAIRT
version AI Hub offers (2.45, 2.49, 2.50; the grid used 2.50), on the grid's 1,024 images. All 8
models, vendor default and Anneal: **identical accuracy on all three versions**. The S24 results are
not an artifact of one toolchain release. Re-running today reproduces the grid within ±0.8pp on every
cell but one: B0 with Anneal is −2.6 today vs −0.7 on 2026-09-27. Our equalised model is
byte-identical to the one built then, the version does not matter, and AI Hub's quantizer is
deterministic today (same model twice → identical output), so the likely cause is a change in AI
Hub's quantizer between the two dates (the old jobs are no longer listed, so this is unproven).
Cells that use Anneal's own quantization (B1, MNv3-S) reproduce exactly.
Files: `examples/qaihub/results/*-s24-check.json`, script `examples/qaihub/run_s24_check.py`.

**Holdout images.** The recipes were chosen on Imagenette. The same frozen recipes (same
calibration, same parameters) were scored on Imagewoof: ten ImageNet dog breeds that no study had
scored (1,000 images; TIDL MobileViT 500). Change vs FP32 on the same images, vendor default → Anneal:

| Model | AMD XINT8 | Galaxy S24 | TI TDA4VM | NVIDIA T4 |
|---|---|---|---|---|
| EfficientNet-B0 | −76.2 → −1.4 | −15.5 → −3.7 | −76.2 → −5.0 | −62.2 → −0.8 |
| EfficientNet-B1 | −71.8 → −5.2 | −73.8 → −5.7 | −73.8 → −8.7 | −73.8 → −6.3 |
| MobileNetV3-Small | −61.8 → −4.8 | −61.4 → −5.7 | −61.8 → −5.7 | −61.8 → −8.1 |
| MobileNetV3-Large | −63.7 → −4.0 | −3.3 → −4.7 | −31.6 → −2.0 | −13.9 → −7.4 |
| MobileNetV2 | −15.4 → −1.0 | +0.4 | −23.4 → −2.6 | +0.5 |
| LCNet-100 | −66.0 → −6.0 | −43.3 → −7.1 | −66.0 → −4.3 | −66.0 → −7.3 |
| MobileViT-S | −74.3 → +1.9 | −23.9 → +1.7 | −73.6 → −2.8 | −63.4 → +0.2 |
| ResNet-50 | −2.9 → +0.3 | +0.4 | −1.4 → 0.0 | −0.7 → −0.9 |

What holds: the vendor defaults collapse on the new images too (most to near 0% accuracy), and
every collapsed cell recovers (from −15…−76 to −0.8…−8.7). What does not: the remaining loss is
larger than on Imagenette (13 of 32 cells within 2.2pp, vs 25 on the grid). Two causes are mixed
here (separated below): recipe selection on the Imagenette images, and harder
images (fine-grained breeds, where a small logit error flips one dog breed into another). One sign
of selection noise: on the T4, the recipe chosen per model is not always the better of the two on
the holdout (MNv3-L: chosen −7.4, the alternative −4.5; LCNet −7.3 vs −5.8). On
the S24, MNv3-L needs no fix and Anneal is 1.4pp worse than the vendor default there.
Files: `examples/*/results/holdout_imagewoof_*`, `examples/tensorrt/results/t4_v15_holdout_imagewoof_n1000.json`.

**Unseen Imagenette images (same difficulty, new images).** To separate the two causes, the grid's
recipes were scored on 1,000 Imagenette validation images that no run had used (the shuffled set
after the first 1,500; TIDL MobileViT 500). Vendor default → the grid's recipe:

| Model | AMD XINT8 | Galaxy S24 | TI TDA4VM | NVIDIA T4 |
|---|---|---|---|---|
| EfficientNet-B0 | −77.2 → −2.4 | −13.8 → **−2.0** | −73.0 → −4.3 | −53.7 → **−1.6** |
| EfficientNet-B1 | −77.0 → −5.0 | −77.0 → **−0.9** | −78.5 → −10.3 | −78.5 → −3.7 |
| MobileNetV3-Small | −65.0 → **−1.4** | −56.7 → **−1.1** | −65.0 → −2.3 | −62.9 → −2.5 |
| MobileNetV3-Large | −46.2 → −3.7 | −2.9 → −3.0 | −17.9 → **−1.2** | −11.5 → −2.8 |
| MobileNetV2 | −6.1 → **+0.7** | **+0.5** | −8.0 → **+2.0** | **−1.2** |
| LCNet-100 | −70.1 → −5.6 | −35.8 → −4.6 | −70.8 → −4.3 | −66.3 → −4.4 |
| MobileViT-S | −77.2 → **+0.1** | −22.9 → **+0.7** | −70.2 → **−0.8** | −58.0 → **+2.1** |
| ResNet-50 | −4.4 → **−2.1** | **+0.3** | −2.2 → **−2.1** | **−0.7** → **−0.3** |

**18 of 32 cells within 2.2pp** (bold), against 25 on the grid's images and 13 on Imagewoof. So
both effects are real and of similar size: choosing the recipe on the reported images made the grid
look about 1–2pp better per cell than new images of the same kind (25 → 18), and harder images cost
about as much again (18 → 13). The collapse and the recovery hold on every set: no collapsed cell
stays collapsed, the worst recovered cell is TI B1 (−10.3).

**One recipe per chip, fixed in advance.** To remove per-model choices, one recipe per chip was
fixed before these runs and scored on both new sets (AMD already used one rule: gated nets one
recipe, ReLU nets CLE):

- **S24**: Anneal's own QDQ (equalised, grid 1/s, uint8) for every model. Unseen: B0 −3.4, B1 −0.9,
  MNv3-S −1.4, MNv3-L −2.2, MNv2 −0.3, LCNet −3.5, ResNet-50 −0.4; MobileViT compiles but fails on
  the device. Imagewoof: B0 −4.3, B1 −5.7, MNv3-S −6.3, MNv3-L −9.5, LCNet −6.8. About as good as the
  per-model choice on Imagenette, worse on MNv3-L where no fix is needed.
- **TI**: equalised (grid 1/s) + 16-bit earliest layers for every gated model, CLE for ReLU nets.
  Close to the per-model choice except LCNet, where grid scales are much worse (Imagewoof −17.9 vs
  −4.3 with plain per-tensor scales).
- **T4**: Anneal's own QDQ (4 tensors float, no FP16) for every model. Unseen: B0 −1.6, B1 −3.7,
  MNv3-S −4.6, MNv3-L −2.6, MNv2 −0.8, LCNet −6.0, MobileViT −0.5, ResNet-50 −0.5. Neither T4 recipe
  wins on every model.

No single recipe per chip beats the per-model choice everywhere; the per-model choices transfer to
new images about as well as a fixed recipe does.

**The unsolved TI cells.** B1: plain per-tensor and derived scales were no better than grid scales
(unseen −14.7 and −11.3 vs −10.3); B1 on TI stays around −9 to −11. LCNet: 16-bit on more early
blocks helps on unseen Imagenette (stem-2 −4.3, stem-3 −3.1, stem-4 −2.6, chosen there), but on
Imagewoof stem-4 scores −5.6 (stem-3 −3.6): a small, unstable gain, so the grid keeps stem-2.
Files: `examples/*/results/unseen_*`, `examples/tensorrt/results/t4_v16_unseen_n1000.json`,
`examples/qaihub/results/*-s24-check-round2.json`.

## Sources

Results JSON: `examples/vitis/results/`, `examples/qaihub/results/`, `examples/tidl/results/`,
`examples/tensorrt/results/`. Scripts alongside. The run ledger with every intermediate result and
dead end is in [findings.md](findings.md).
