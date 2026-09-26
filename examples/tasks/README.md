# Beyond classification: COCO detection and segmentation

Does the INT8 failure, and Anneal's fix, carry over to the segmentation and detection networks
deployed on edge devices?

| Script | What it does |
|---|---|
| `export_models.py` | Exports `lraspp_mobilenet_v3_large` (torchvision, 512×512), `ssdlite320_mobilenet_v3_large` (torchvision, 320×320) and `yolov8n` (Ultralytics, AGPL-3.0, benchmarking only; 640×640) to ONNX with raw head outputs. Anchor decoding and NMS stay outside the quantized graph, as in deployment. |
| `run_tasks.py` | Builds each recipe, scores it and FP32 on the same images, writes `results/<model>.json`. `--images` (default 500), `--calib-images` (default 32). |

**Protocol.** COCO val2017. Calibration uses the highest-id val images of the task (32), scoring
the first 500 by id, never the same images. Every recipe is emulated: onnxruntime with graph
optimisations off, i.e. 32-bit accumulation as on ARM and the NPUs in
[../qaihub](../qaihub/README.md). Detection metric: box mAP@[.5:.95]; its 95% CI is fold-level
(mAP recomputed on each tenth of the images, t-interval on the ten paired deltas, centred on
their mean, which need not equal the pooled delta).
Segmentation: mIoU over the 21 VOC classes, with a paired bootstrap CI.

**Results** (change vs FP32 in mAP points, 500 images; fold CI in brackets):

| Recipe | SSDLite-MobileNetV3 (FP32 23.8) | YOLOv8n (FP32 40.5) |
|---|---:|---:|
| onnxruntime default (minmax) | −2.59 [−4.46, −1.84] | −0.89 [−1.43, −0.31] |
| symmetric percentile 99.999 | −0.60 [−2.44, +0.13] | −0.94 [−1.27, +0.18] |
| Anneal advised | −0.81 [−2.03, −0.24] (equalise + asym percentile + stem) | −0.52 [−0.70, +0.76] (sym 99.999 + stem) |
| Anneal without equalisation | −0.70 [−1.58, −0.42] | – |
| Anneal advised + dense equalisation | – | −0.89 [−1.58, +0.32] |

- **Neither detector collapses**, unlike EfficientNet-B0 (−45pp on ImageNet). SSDLite's
  default loses 2.6 points; percentile calibration alone recovers most of it, and equalisation
  adds nothing measurable. YOLOv8n (SiLU, no depthwise convs) has no gated-depthwise sites,
  so the advisor gives it the plain-CNN recipe.
- `rank_sites`' total predicted gain separates the models: 2.9 for SSDLite against 58–366 for
  EfficientNet-B0/B1/V2-S and MobileNetV3 (not stored in `results/`; see commit 5fafa9b and
  the `equalise` docstring). The option `equalize_min_gain` (commit 5fafa9b) uses
  this to skip equalisation when it would not pay for its NPU cost.
- **LRASPP-MobileNetV3 segmentation: pending.** It ran out of memory calibrating on 32 images;
  a rerun with `--calib-images 16` is queued.

    python export_models.py
    python run_tasks.py --model ssdlite320_mobilenet_v3_large --images 500
