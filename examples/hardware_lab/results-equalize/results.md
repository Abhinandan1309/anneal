# Hardware lab: static INT8 recipes across CPUs

Calibrated on 64 Imagenette train images, scored against FP32 on the same validation images. Each cell: accuracy change (\* = McNemar p < 0.05) · speedup.

## efficientnet_b0

| machine | CPU | INT8 path | drift | U8S8 per-channel | P + stem | EQ + P + stem | EQ + P + stem + rr | EQ + P + stem + gates |
|---|---|---|---|---|---|---|---|---|
| macOS ARM64 | Apple M1 (Virtual) | arm-dotprod | 16.1% ⚠ | -49.1pp* · 3.90x | -4.8pp* · 3.78x | -0.5pp · 2.64x | -0.7pp · 3.01x | -1.6pp* · 2.68x |
| Linux X64 | AMD EPYC 7763 64-Core Processor | x86-avx2-16bit | 0.1% | -50.9pp* · 1.27x | -7.0pp* · 1.30x | -3.1pp* · 1.15x | -0.6pp · 1.18x | -2.8pp* · 1.04x |
| Linux ARM64 | Neoverse-N2 | arm-dotprod | 0.8% | -49.1pp* · 2.37x | -4.8pp* · 2.33x | -0.5pp · 2.11x | -0.7pp · 2.11x | -1.6pp* · 1.71x |
| Linux X64 | INTEL(R) XEON(R) PLATINUM 8573C | x86-vnni | 0.2% | -48.4pp* · 1.18x | -4.6pp* · 1.22x | -0.7pp · 1.04x | -0.5pp · 1.04x | -1.5pp* · 0.89x |
| Windows X64 | Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz | x86-vnni | 0.9% | -48.7pp* · 1.14x | -4.6pp* · 1.20x | -0.7pp · 0.99x | -0.5pp · 1.01x | -1.5pp* · 0.83x |
| Windows X64 | AMD EPYC 7763 64-Core Processor | x86-avx2-16bit | 4.3% | -51.3pp* · 1.24x | -7.0pp* · 1.30x | -3.1pp* · 1.17x | -0.6pp · 1.16x | -2.8pp* · 1.01x |
| **predicted saturation (x86-avx2-16bit only)** | | | | 31 layer(s), worst 18.8% | 43 layer(s), worst 19.5% | 46 layer(s), worst 27.1% | none possible | 45 layer(s), worst 25.4% |

## mobilenet_v3_large

| machine | CPU | INT8 path | drift | U8S8 per-channel | P + stem | EQ + P + stem | EQ + P + stem + rr | EQ + P + stem + gates |
|---|---|---|---|---|---|---|---|---|
| macOS ARM64 | Apple M1 (Virtual) | arm-dotprod | 3.2% | -11.8pp* · 3.03x | -2.5pp* · 3.10x | -1.0pp · 2.97x | -4.5pp* · 2.95x | -1.4pp* · 2.66x |
| Linux X64 | AMD EPYC 7763 64-Core Processor | x86-avx2-16bit | 0.3% | -24.8pp* · 1.03x | -3.6pp* · 1.06x | -2.6pp* · 1.02x | -4.8pp* · 1.03x | -3.0pp* · 1.04x |
| Linux ARM64 | Neoverse-N2 | arm-dotprod | 0.1% | -11.5pp* · 1.93x | -1.7pp* · 1.96x | -1.1pp* · 1.88x | -4.2pp* · 1.89x | -1.2pp* · 1.92x |
| Linux X64 | INTEL(R) XEON(R) PLATINUM 8573C | x86-vnni | 2.0% | -11.3pp* · 0.93x | -2.1pp* · 1.00x | -1.4pp* · 0.95x | -4.5pp* · 0.94x | -1.4pp* · 0.96x |
| Windows X64 | Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz | x86-vnni | 8.9% | -11.3pp* · 0.89x | -2.1pp* · 0.99x | -1.4pp* · 0.89x | -4.5pp* · 0.88x | -1.4pp* · 0.91x |
| Windows X64 | AMD EPYC 7763 64-Core Processor | x86-avx2-16bit | 1.5% | -24.8pp* · 1.03x | -3.6pp* · 1.13x | -2.6pp* · 1.09x | -4.8pp* · 1.02x | -3.0pp* · 1.01x |
| **predicted saturation (x86-avx2-16bit only)** | | | | 19 layer(s), worst 14.4% | 36 layer(s), worst 2.9% | 35 layer(s), worst 2.8% | none possible | 35 layer(s), worst 2.8% |

Drift is the change in FP32 latency between the start and end of each job; ⚠ marks over 10% drift or a machine-state warning, and that machine's speedups are not reliable. Accuracy is unaffected by timing noise.
