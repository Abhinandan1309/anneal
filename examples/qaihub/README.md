# Real edge devices: Qualcomm AI Hub

Does the INT8 collapse, and the equalisation fix, hold on real NPUs with the vendor's own
toolchain? Each script compiles models for a real device on Qualcomm AI Hub, runs them there and
scores them paired against the same device's FP32 output. The pre-registered design and the
graded predictions are in [docs/edge_study_design.md](../../docs/edge_study_design.md) and
[docs/edge_study_results.md](../../docs/edge_study_results.md).

**Images: Imagenette, not ImageNet** (ImageNet's licensed images are not uploaded). 64
calibration images, 256–1,024 scored, 1000-way. Differences of 1–2pp are within noise here.

| Script | What it does |
|---|---|
| `run_qaihub.py` | Per model, device and runtime (`tflite`, `qnn_dlc`): `fp32`, `hub int8` (Qualcomm's quantizer, W8A8), `hub int8 + equalised` (the same quantizer on Anneal's equalised model), `anneal recipe` (Anneal's own QDQ model) |
| `qaihub_retry_sa8775p.py`, `qaihub_retry2_sa8775p.py` | Re-run the SA8775P jobs that failed with "failed after compiling"; the second scores the equalised model in 4 chunks of 256 |
| `run_topk.py` | Selective equalisation: only the top k of 16 sites by `rank_sites`' predicted gain |
| `run_site_value.py` | Leave-one-site-out: each site's measured value and cost on the device |
| `run_recipe_debug.py` | Anneal's own QDQ models compiled by QNN, one ingredient at a time; also the MobileNetV3-Small and B1 recipes |
| `run_speed.py` | Where equalisation's NPU latency goes, and exact rewrites that move it (gate-conv, grid 1/s, gate clip); also the 8-model sweep |
| `run_perchannel_gate.py` | Fold 1/s into a per-channel dequantization of the gate input |
| `run_w8a16.py` | Qualcomm's quantizer at 16-bit activations (W8A16), with and without equalisation |

**EfficientNet-B0** ([results/](results/); change vs device FP32, McNemar p):

| Device (compute unit) | Runtime | n | Qualcomm INT8 | + equalisation | latency FP32 / INT8 / eq, ms |
|---|---|---:|---:|---:|---|
| Galaxy S24 (NPU) | QNN | 1,024 | −12.0 (7e-22) | **−0.7** (0.46) | 0.851 / 0.415 / 0.534 |
| Galaxy S24 (NPU) | TFLite | 1,024 | −11.5 (1e-20) | **−1.5** (0.08) | 0.847 / 0.415 / 0.546 |
| SA8775P (NPU) | TFLite | 256 ¹ | −13.3 (6e-7) | **−2.0** (0.33) | 1.65 / – / 1.071 |
| Pixel 8 (GPU) | TFLite | 512 | −12.7 (3e-14) | **−1.2** (0.45) | 8.924 / 9.199 / 9.708 |

¹ The device failed intermittently on every variant; the equalised model ran on images
768–1,023 only (`...-sa8775p-adp-tflite-retries.json`). Over all 1,024 images Qualcomm INT8
lost 11.5pp there. `anneal recipe` compiled for the S24 but did not run (`...-tflite-n512.json`); the cause was
found later (below: int8 activations).
ResNet-50, the control: +0.4pp (S24) and +0.8pp (Pixel 8), not significant; its SA8775P FP32
job did not run.

**Selective equalisation** (S24, QNN, 1,024 images; `...-topk-...-qnn_dlc-n1024.json`):

| sites equalised (k of 16) | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy change, pp | −12.0 | −11.2 | −11.0 | −10.0 | −6.4 | −0.7 |
| latency, ms (FP32 0.839) | 0.425 | 0.416 | 0.424 | 0.428 | 0.453 | 0.541 |

Accuracy returns only with the last 8 sites, which also carry most of the latency cost, so the
predicted per-site gain does not rank a site's value on the device.

**Each site's value** (`run_site_value.py`, leave one site out, S24 QNN, 1,024 images): dropping
the stem site costs 6.8pp (p < 0.001) and `features.3.0` 2.7pp (p = 0.002); each of the other 14
is within noise (≤ 1.6pp). The two that matter rank 9th and 7th by predicted gain. No single site
saves more than 0.03 ms.

**Speed** (`run_speed.py`, B0, QNN): plain INT8 0.414 ms (−12.0pp); equalised 0.536 ms (−0.7);
gate-conv rewrite 0.497 ms (−1.0), a third of the cost removed; grid 1/s 0.538 ms (+0.1, no
extra cost); gate clip +19% (QNN does not fuse Clip, so not used on QNN). Moving the scale after
SiLU is faster (0.475 ms) but loses 14.5pp, because QNN quantizes the conv output. Folding 1/s
into a per-channel dequantization of the gate input (`run_perchannel_gate.py`) is rejected by QNN
("Cannot apply per channel quantization on activation").

**16-bit activations instead** (`run_w8a16.py`, 1,024 images): B0 W8A16 −0.3pp at 0.829 ms, about
FP32's latency (0.845), against equalised W8A8 −0.7pp at 0.549 ms. MobileNetV3-Small W8A16 −5.5pp
(0.27 ms): 16-bit activations do not fix it.

**Anneal's own QDQ on QNN** (`run_recipe_debug.py`): QNN runs it faithfully only with uint8
activations; with int8 activations, or with uint16 tensors mixed in (onnxruntime's
`TensorQuantOverrides`), it compiles and scores 0%. With uint8: B0 −0.8pp vs device FP32 (256
images); MobileNetV3-Small −1.4pp vs true FP32 at 0.26 ms, where Qualcomm's quantizer with
equalisation gives −7.4; EfficientNet-B1 −1.7pp at 0.893 ms, against −3.2 with Qualcomm's
quantizer and equalisation (`...-anneal-qdq-s24-qnn_dlc-n1024.json`).

**A correction on references.** The EfficientNet-B0/B1 files before 2026-09-28 score against the
device's own FP16 run. On B1 that run is 6.8pp below true FP32 (69.2% vs 76.1%, same images), so
B1's "+3.6pp" there is −3.2pp against true FP32. Later files record `true_fp32_accuracy`. The
8-model sweep and the other toolchains are in [docs/benchmark_grid.md](../../docs/benchmark_grid.md).

    python run_qaihub.py --model efficientnet_b0 --device "Samsung Galaxy S24 (Family)" --runtime qnn_dlc --images 1024
