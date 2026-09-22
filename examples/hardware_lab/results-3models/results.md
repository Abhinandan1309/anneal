# Hardware lab: static INT8 recipes across CPUs

Calibrated on 64 Imagenette train images, scored against FP32 on the same validation images. Each cell: accuracy change (\* = McNemar p < 0.05) · speedup.

## efficientnet_b0

| machine | CPU | INT8 path | drift | U8S8 per-channel | U8S8 per-ch + reduce_range | S8S8 per-channel | U8S8 per-tensor | S8S8 per-tensor |
|---|---|---|---|---|---|---|---|---|
| macOS ARM64 | Apple M1 (Virtual) | arm-dotprod | 1.3% | -49.3pp* · 3.05x | -46.8pp* · 2.99x | -49.3pp* · 3.08x | -75.4pp* · 3.08x | -75.4pp* · 3.17x |
| Linux ARM64 | Neoverse-N2 | arm-dotprod | 0.1% | -49.3pp* · 2.15x | -46.8pp* · 2.34x | -49.3pp* · 2.36x | -75.4pp* · 2.35x | -75.4pp* · 2.33x |
| Linux X64 | AMD EPYC 7763 64-Core Processor | x86-avx2-16bit | 0.0% | -49.7pp* · 1.26x | -46.1pp* · 1.29x | -47.8pp* · 0.38x | -75.4pp* · 1.26x | -75.4pp* · 0.38x |
| Windows X64 | AMD EPYC 7763 64-Core Processor | x86-avx2-16bit | 2.8% | -51.2pp* · 1.25x | -47.3pp* · 1.25x | -47.8pp* · 0.38x | -75.4pp* · 1.14x | -75.4pp* · 0.38x |
| **predicted saturation (x86-avx2-16bit only)** | | | | 31 layer(s), worst 18.8% | none possible | 31 layer(s), worst 18.8% | 6 layer(s), worst 8.8% | 6 layer(s), worst 8.8% |

## mobilenet_v3_large

| machine | CPU | INT8 path | drift | U8S8 per-channel | U8S8 per-ch + reduce_range | S8S8 per-channel | U8S8 per-tensor | S8S8 per-tensor |
|---|---|---|---|---|---|---|---|---|
| macOS ARM64 | Apple M1 (Virtual) | arm-dotprod | 0.2% | -12.2pp* · 2.98x | -11.5pp* · 3.04x | -12.2pp* · 3.10x | -37.1pp* · 3.03x | -37.1pp* · 3.12x |
| Linux ARM64 | Neoverse-N2 | arm-dotprod | 0.5% | -12.2pp* · 1.87x | -11.8pp* · 1.87x | -12.2pp* · 1.93x | -37.2pp* · 1.90x | -37.2pp* · 1.97x |
| Linux X64 | AMD EPYC 7763 64-Core Processor | x86-avx2-16bit | 1.3% | -22.8pp* · 1.03x | -11.0pp* · 1.02x | -12.1pp* · 0.39x | -41.6pp* · 1.04x | -38.2pp* · 0.39x |
| Windows X64 | AMD EPYC 7763 64-Core Processor | x86-avx2-16bit | 1.7% | -22.8pp* · 0.89x | -11.0pp* · 0.94x | -12.1pp* · 0.39x | -41.6pp* · 0.96x | -38.2pp* · 0.39x |
| **predicted saturation (x86-avx2-16bit only)** | | | | 19 layer(s), worst 14.4% | none possible | 19 layer(s), worst 14.4% | 1 layer(s), worst 2.0% | 1 layer(s), worst 2.0% |

## resnet18

| machine | CPU | INT8 path | drift | U8S8 per-channel | U8S8 per-ch + reduce_range | S8S8 per-channel | U8S8 per-tensor | S8S8 per-tensor |
|---|---|---|---|---|---|---|---|---|
| macOS ARM64 | Apple M1 (Virtual) | arm-dotprod | 9.5% | +1.1pp · 4.16x | +0.2pp · 4.09x | +1.1pp · 4.79x | +1.0pp · 4.37x | +1.0pp · 4.58x |
| Linux ARM64 | Neoverse-N2 | arm-dotprod | 0.1% | +1.1pp · 3.61x | +0.2pp · 3.61x | +1.1pp · 3.92x | +1.0pp · 3.64x | +1.0pp · 4.03x |
| Linux X64 | AMD EPYC 7763 64-Core Processor | x86-avx2-16bit | 0.3% | -4.2pp* · 1.89x | +0.4pp · 1.89x | -4.3pp* · 1.14x | +1.3pp · 1.91x | +1.5pp* · 1.14x |
| Windows X64 | AMD EPYC 7763 64-Core Processor | x86-avx2-16bit | 0.3% | -4.2pp* · 1.82x | +0.4pp · 1.85x | -4.3pp* · 1.09x | +1.3pp · 1.79x | +1.5pp* · 1.09x |
| **predicted saturation (x86-avx2-16bit only)** | | | | 1 layer(s), worst 21.5% | none possible | 1 layer(s), worst 21.5% | 1 layer(s), worst 1.1% | 1 layer(s), worst 1.1% |

Drift is the change in FP32 latency between the start and end of each job; ⚠ marks over 10% drift or a machine-state warning, and that machine's speedups are not reliable. Accuracy is unaffected by timing noise.
