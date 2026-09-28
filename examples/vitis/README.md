# AMD XINT8 (Quark)

AMD Quark's XINT8 configuration, the arithmetic of AMD's Ryzen AI NPUs and DPUs: symmetric
power-of-two scales, per tensor for activations and weights. The quantized models run in
onnxruntime on a CPU (GitHub runner), so these are **emulated** numbers from AMD's own quantizer,
not measurements on an NPU. Imagenette validation images (1,000), paired against FP32.
Headline numbers are in the [benchmark grid](../../docs/benchmark_grid.md).

## Running

The `quark-lab` workflow (Actions → quark-lab → Run workflow) takes `models`, `images` and
`variants` (labels in `run_quark.py`'s docstring and `VARIANTS`); it runs `run_quark.py`.
`surrogate_float.py` checks the FP32 surrogate for the gate in float.

## Anneal's AMD recipe

For gated models, per-tensor equalisation with grid-aligned 1/s, an FP32 surrogate for XINT8's
Sigmoid → HardSigmoid swap, the gates in 16 bits, and measured bias correction. For ReLU models,
CLE (max scale 4) with bias correction.

- XINT8 collapses every gated model tested (−44.8pp on MobileNetV3-L, −64.7 to −80.8pp on the
  others) and loses 7.9pp on MobileNetV2.
- The recipe brings EfficientNet-B0, B2, B3, V2-S, MobileNetV3-S/L, MobileViT and FBNetV3 within
  about 2.3pp; EfficientNet-B1 −3.1 to −4.4 across three runs, LCNet −2.9.
- Bias correction matters: on B1, a gate-conv variant is −43.6pp without it and −3.5 with it.
- Noise-optimal scales (`equalize_derived`): B0 pure 8-bit −5.4 → −1.8, but B1 −8.0 → −11.0; ties
  with the gates in 16 bits.

## Results

`sweep_final_*.json` and `sweep_fixed_fbnet_b1_mnv3l_n1000.json` (the gated sweep, final code);
`sweep_controls_n1000.json`, `relu_cnn_ablation_n1000.json` (ResNet-50, MobileNetV2);
`derived_vs_grid_*.json` (noise-optimal scales); `quark_*.json`, `run*.json` (earlier steps:
grid, gate-conv, clip, bias correction).
