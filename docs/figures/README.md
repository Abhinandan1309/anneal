# Figures

Made by [examples/figures/make_figures.py](../../examples/figures/make_figures.py) from committed
results. Numbers match [benchmark_grid.md](../benchmark_grid.md).

| File | Shows | Caption (short) |
|---|---|---|
| `channel_ranges.png` | Why INT8 breaks | One INT8 scale is shared by 1,152 channels of EfficientNet-B0. 81 channels get less than one INT8 level. After Anneal's equalisation, none do. |
| `method.png` | The fix | Scale each channel by s before the gate and divide it out after. The gate sees the same input, so the model's output is unchanged. No retraining. |
| `workflow.png` | Where Anneal fits | Anneal sits between the trained model and the vendor's toolchain. It hands over either a rewritten float model or its own INT8 model, and checks the result on the target. No retraining, no change to the vendor's tools. |
| `grid.png` | The results | 8 models × 4 toolchains. Left: the vendor's default INT8. Right: with Anneal. Top-1 change vs FP32, percentage points. |
| `speed_s24.png` | The cost | Galaxy S24 NPU. Anneal's INT8 is slower than plain INT8 but faster than FP16, and far more accurate than plain INT8. |
| `social_preview_dark.png` | GitHub social preview (1280×640) | Upload in the repo's Settings → Social preview. The bars are EfficientNet-B0's real per-channel INT8 levels before and after. |
| `segmentation.png` | What it looks like | LRASPP on TI's TDA4VM. TIDL's 8-bit masks break (9.2 mIoU); with Anneal they nearly match FP32 (53.2 vs 54.4). |

## Alt text

- `channel_ranges.png`: Two bar charts of INT8 levels per channel, log scale. Left, red: levels
  fall from 127 to below 1 for the smallest 81 channels. Right, green: every channel keeps
  several levels.
- `method.png`: Two rows of three boxes: conv A, gate, depthwise B. In the second row conv A is
  multiplied by s, the gate reads x'/s, and B is divided by s.
- `workflow.png`: Boxes left to right: trained model, Anneal, then the vendor quantizer or
  compiler, then the target. A dashed arrow returns measurements from the target to Anneal.
- `grid.png`: Two heatmaps, 8 models by 4 targets. The left is mostly red (−45 to −77 points);
  the right is mostly green (within 2 points).
- `speed_s24.png`: Grouped bars of latency for three models on the S24, labelled with accuracy
  change.
- `segmentation.png`: Four COCO photos with segmentation masks from FP32, TIDL 8-bit and TIDL
  with Anneal.

## Notes for the text

- AMD and TI columns are the vendors' own quantizers and emulators on a PC. S24 and T4 are real
  hardware. Say so next to the grid.
- On a T4 at batch 1, INT8 is slower than FP16 for these models. T4 latencies came from different
  Kaggle sessions, so they are not charted.
- Intel OpenVINO does not collapse, and Anneal does not help there.
- Weakest cells: EfficientNet-B1 on TI (−8.9) and MobileViT on TI (−4.2; TI's 16-bit mode −2.6).
- Segmentation example images were chosen by a fixed rule before the run: CC BY 2.0, one VOC
  class, largest object area, one image per class. Credits: `segmentation_credits.txt`.
- Do not show ImageNet or Imagenette photos: their licence does not allow redistribution.
