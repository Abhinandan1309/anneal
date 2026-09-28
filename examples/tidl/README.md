# TI TDA4VM (TIDL)

TI's own INT8 path for the TDA4VM (J721E): `edgeai-tidl-tools` at a pinned commit, run as TI's
host emulation on a GitHub runner. These are **emulated** numbers from TI's toolchain, not
measurements on a board. TIDL quantizes feature maps per tensor with power-of-two scales, and it
cannot import Anneal's QDQ models faithfully except for plain convnets (below). Anneal therefore changes the float model (exact equalisation) and uses TIDL's
own options: 16-bit feature maps for chosen layers (`output_feature_16bit_names_list`), and its
automatic mixed precision.

Headline numbers are in the [benchmark grid](../../docs/benchmark_grid.md). Imagenette
validation images, paired against FP32 (onnxruntime, CPU).

## Running

The `tidl-lab` workflow (Actions → tidl-lab → Run workflow) takes `models`, `images`,
`variants` (labels from `VARIANTS` in `run_tidl.py`) and `variant_timeout` (seconds, default
3600). Each variant compiles and runs in a forked child process with a 12 GB address-space
cap, so an importer segfault, a hang or an out-of-memory run costs that variant, not the job.
`tidl-tasks` runs `run_tidl_tasks.py` (LRASPP segmentation, YOLOv8n detection).

| Script | What it does |
|---|---|
| `run_tidl.py` | Classifiers: TIDL 8-bit and 16-bit, equalisation variants, 16-bit early layers by regex, pre-quantized QDQ import |
| `run_tidl_tasks.py` | LRASPP-MobileNetV3 (COCO, mIoU) and YOLOv8n (COCO, mAP); 16-bit layer selection rules |
| `build_qdq.py` | Builds the pre-quantized QDQ models in a clean environment (TIDL's `onnx` cannot import onnxruntime's quantizer) |
| `emulate_study.py` | A TIDL-like emulation in onnxruntime, for fast screening before a TIDL run |
| `trace_diff.py`, `cle_float_check.py` | Layer-by-layer comparison against float; CLE exactness check |

## Results

| File | What it holds |
|---|---|
| `mnv3_grid_n1500.json`, `mnv3s_eq_16bit_early_n1500.json` | MobileNetV3-S/L: equalisation (grid), 16-bit early layers, auto mixed precision |
| `b1_eq_16bit_early_n1500.json`, `b1_eq_16bit_f03_n1500.json`, `b0_eq_16bit_early_n1500.json` | EfficientNet-B1/B0 with 16-bit features 0-1 / 0-2 / 0-3 |
| `derived_vs_grid_*_n1500.json` | Noise-optimal scales (`equalize_derived`) against the default grid scales |
| `lcnet_eq_16bit_early_n1500.json`, `grid_lcnet_mobilevit_n1500.json` | LCNet |
| `mobilevit_*_n300.json`, `mobilevit_*_n500.json` | MobileViT-S (plain, equalised, 16-bit early stages, TIDL 16-bit) |
| `grid_resnet50_n1500.json`, `run_36271625503.json` | Controls: ResNet-50, MobileNetV2 (CLE with max scale 4) |
| `effnet_grid_clip_n1500.json`, `classifiers_al9_n1500.json` | Grid/clip variants; accuracy level 9 (worse everywhere tested) |
| `prequant_*_n1000.json`, `prequant_mnv3_n1500.json` | Pre-quantized QDQ import: faithful on ResNet-18, degraded with depthwise convs, garbage with depthwise + ReLU6 |
| `lraspp_*_n300.json`, `lraspp_trace_run_n100.json` | LRASPP segmentation: −44.7 → −1.2 mIoU with equalisation + 16 bits on four backbone layers |
| `yolov8n_*_n300.json` | YOLOv8n: auto mixed precision + the head concat on ARM, −0.99 mAP |
| `emulate_*.json` | Emulation screens (not TIDL) |
| `run_362*.json`, `mnv2_mnv3l_n512.json`, `unet_carvana_n50.json` | Earlier runs, kept for the record |
