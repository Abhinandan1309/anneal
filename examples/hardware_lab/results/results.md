# Hardware lab: static INT8 recipes across CPUs

ResNet-18, calibrated on 64 Imagenette train images, scored against FP32 on the same validation images. Each cell: accuracy change (\* = McNemar p < 0.05) · speedup.

| machine | CPU | VNNI | ARM dotprod | latency drift | U8S8 per-channel | U8S8 per-ch + reduce_range | S8S8 per-channel | U8S8 per-tensor | S8S8 per-tensor |
|---|---|---|---|---|---|---|---|---|---|
| Windows AMD64 | AMD Ryzen 7 4800H with Radeon Graphics | no | no | 0.8% | -4.2pp* · 1.07x | +0.4pp · 1.06x | -4.3pp* · 0.83x | +1.3pp · 1.06x | +1.5pp* · 0.83x |
| macOS ARM64 | Apple M1 (Virtual) | no | yes | 5.7% | +1.1pp · 3.96x | +0.2pp · 3.18x | +1.1pp · 4.09x | +1.0pp · 3.29x | +1.0pp · 4.44x |
| Linux ARM64 | Neoverse-N2 | no | yes | 1.0% | +1.1pp · 3.64x | +0.2pp · 3.66x | +1.1pp · 4.07x | +1.0pp · 3.64x | +1.0pp · 4.08x |
| Linux X64 | AMD EPYC 7763 64-Core Processor | no | no | 0.6% | -4.2pp* · 1.94x | +0.4pp · 1.95x | -4.3pp* · 1.14x | +1.3pp · 1.95x | +1.5pp* · 1.14x |
| Windows X64 | Intel(R) Xeon(R) 6973P-C | yes | no | 2.2% | +1.0pp · 2.93x | +0.0pp · 2.92x | +1.3pp* · 1.04x | +0.9pp · 2.78x | +0.8pp · 0.97x |

Latency drift is the change in FP32 latency between the start and end of the job; above 10% (⚠) that machine's speedups are not reliable. Accuracy is unaffected by timing noise.
