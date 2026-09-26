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
| `run_site_value.py` | Leave-one-site-out: each site's measured value and cost on the device. *Running; no results yet* |
| `run_recipe_debug.py` | Bisects why `anneal recipe` compiles for the S24 but does not run. *Queued; no results yet* |

**EfficientNet-B0** ([results/](results/); change vs device FP32, McNemar p):

| Device (compute unit) | Runtime | n | Qualcomm INT8 | + equalisation | latency FP32 / INT8 / eq, ms |
|---|---|---:|---:|---:|---|
| Galaxy S24 (NPU) | QNN | 1,024 | −12.0 (7e-22) | **−0.7** (0.46) | 0.851 / 0.415 / 0.534 |
| Galaxy S24 (NPU) | TFLite | 1,024 | −11.5 (1e-20) | **−1.5** (0.08) | 0.847 / 0.415 / 0.546 |
| SA8775P (NPU) | TFLite | 256 ¹ | −13.3 (6e-7) | **−2.0** (0.33) | 1.65 / – / 1.071 |
| Pixel 8 (GPU) | TFLite | 512 | −12.7 (3e-14) | **−1.2** (0.45) | 8.924 / 9.199 / 9.708 |

¹ The device failed intermittently on every variant; the equalised model ran on images
768–1,023 only (`...-sa8775p-adp-tflite-retries.json`). Over all 1,024 images Qualcomm INT8
lost 11.5pp there. `anneal recipe` compiled for the S24 but did not run (`...-tflite-n512.json`).
ResNet-50, the control: +0.4pp (S24) and +0.8pp (Pixel 8), not significant; its SA8775P FP32
job did not run.

**Selective equalisation** (S24, QNN, 1,024 images; `...-topk-...-qnn_dlc-n1024.json`):

| sites equalised (k of 16) | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy change, pp | −12.0 | −11.2 | −11.0 | −10.0 | −6.4 | −0.7 |
| latency, ms (FP32 0.839) | 0.425 | 0.416 | 0.424 | 0.428 | 0.453 | 0.541 |

Accuracy returns only with the last 8 sites, which also carry most of the latency cost, so the
predicted per-site gain does not rank a site's value on the device.

    python run_qaihub.py --model efficientnet_b0 --device "Samsung Galaxy S24 (Family)" --runtime qnn_dlc --images 1024
