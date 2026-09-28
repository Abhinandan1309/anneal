# NVIDIA T4 (TensorRT)

A T4 GPU on Kaggle, TensorRT 10.16, batch 1: **measured on the device**. Imagenette validation
images, paired against FP32 (onnxruntime, CPU). Headline numbers are in the
[benchmark grid](../../docs/benchmark_grid.md).

## What is compared

| Label | Engine |
|---|---|
| `trt int8` | TensorRT's implicit INT8 (calibrator + FP16 fallback; deprecated by NVIDIA) |
| `modelopt int8` | NVIDIA ModelOpt ONNX PTQ → explicit QDQ, NVIDIA's recommended path |
| `modelopt int8 + eq (modelopt convs only)` | the same on Anneal's equalised model, ModelOpt quantizing convolutions only |
| `anneal qdq (sym)` | Anneal's own QDQ model (symmetric int8), made TensorRT-buildable (`trt_ready_qdq`) |
| `anneal qdq (sym 4 float)` | the same, with the 4 most sensitive tensors left in float |
| `... (no fp16)` | built without the FP16 flag |
| `trt fp16` | the FP16 reference |

## Running

`kaggle/run.py` is the Kaggle kernel: edit `MODELS`, `IMAGES` and `VARIANTS`, then
`kaggle kernels push -p examples/tensorrt/kaggle`; it clones this repository and runs
`run_trt.py`. `run_trt.py` runs anywhere TensorRT and a CUDA GPU are available.
`qdq_prefix_diff.py` bisects where a QDQ engine diverges from float.

## Findings

- ModelOpt INT8 collapses the gated models (−52 to −76pp; MobileNetV3-L −9.9); equalisation
  with ModelOpt on convolutions only brings them to between −3.5 and +0.6pp (EfficientViT-B0
  −70 → −33).
- Anneal's own QDQ: EfficientNet-B0 −0.1, B1 −3.5 and EfficientViT-B0 −12.8 (4 tensors in float,
  no FP16).
- Tensors left in float next to INT8 overflow in FP16: B0 −75.1pp with FP16, −0.1 without it
  (+10% latency).
- The implicit INT8 builds are not deterministic (EfficientViT-B0 −69.4 and −0.2 in two builds
  of the same model).
- ModelOpt patches onnxruntime's calibrators in-process; Anneal's QDQ is built in a fresh
  process for that reason.
- **Speed:** at batch 1 every INT8 engine here is slower than TensorRT FP16 (B1 2.5 ms against
  1.32 ms at −0.1pp). On a T4, FP16 is the better choice for these models.

## Results

`t4_imagenette1000.json`, `t4_modelopt_fixed_imagenette1000.json`, `t4_rebuilds_n1000.json` (B0,
B1, EfficientViT, MobileNetV3-L); `t4_v8_convs_only_n1000.json`, `t4_eqpos_n1000.json`,
`t4_modelopt_gates_grid_n1000.json` (ModelOpt variants); `t4_v10_grid_n1000.json` (the 8-model
grid); `t4_v9`, `t4_v11`–`t4_v14` (Anneal's QDQ, FP16 vs not); `modelopt_eq_prefix_diff.txt`.
