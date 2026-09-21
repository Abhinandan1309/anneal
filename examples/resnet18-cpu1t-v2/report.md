# Anneal run `b5aaf0b7cef1`

## Setup

| | |
|---|---|
| Model | `torchvision:resnet18` |
| Target | `cpu-1t` (CPUExecutionProvider) |
| Threads | 1 intra-op, 1 inter-op |
| Machine | AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD |
| Platform | Windows-11-10.0.26200-SP0 |
| onnxruntime | 1.30.0 |
| Policy | `heuristic` |
| Eval set | `imagenette` (256 images) |
| Latency protocol | 10 warm-up discarded, 50 timed runs, batch 1 |
| Budget | 10 trials |

**Measurement stability:** the baseline re-measured at the end of the run drifted 2.5% (tolerance 10%), and no power or throttling problems were detected.

**Accuracy resolution: ±5.7pp** (95% Wilson interval at n=256). Two trials whose top-1 differs by less than roughly this much are tied, not ranked. When the difference is within noise, `agreement` — the fraction of images where a candidate predicts the same class as the baseline — is the sharper signal, because it is paired per-image rather than an aggregate.

## Frontier

![Pareto frontier](frontier.svg)

## All trials

| # | recipe | p50 ms | p99 ms | speedup | size MB | top-1 | Δpp | agreement |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `baseline` | 34.44 | 44.57 | 1.00x | 44.58 | 66.80% | +0.00 | — |
| 1 | `graph_optimize(level=all)` | 35.46 | 43.97 | 0.97x | 44.58 | 66.80% | +0.00 | 1.000 |
| 2 | `quantize_dynamic_int8(per_channel=True,reduce_range=False,weight_type=int8)` | 577.45 | 646.32 | 0.06x | 11.21 | 67.97% | +1.17 | 0.961 |
| 3 | `quantize_static_int8(activation_type=uint8,calib_samples=64,calibrate_method=minmax,per_channel=True,reduce_range=False)` | 33.08 | 39.69 | 1.04x | 11.28 | 64.45% | -2.34 | 0.852 |
| 4 | `quantize_dynamic_int8(per_channel=False,reduce_range=False,weight_type=int8)` | 577.40 | 697.91 | 0.06x | 11.20 | 67.97% | +1.17 | 0.965 |
| 5 | `quantize_static_int8(activation_type=uint8,calib_samples=64,calibrate_method=minmax,per_channel=True,reduce_range=True)` | 32.10 | 69.96 | 1.07x | 11.28 | 67.97% | +1.17 | 0.961 |
| 6 | `quantize_static_int8(activation_type=uint8,calib_samples=64,calibrate_method=minmax,per_channel=False,reduce_range=False)` | 32.00 | 34.86 | 1.08x | 11.20 | 67.97% | +1.17 | 0.953 |
| 7 | `quantize_dynamic_sensitive(per_channel=True,ranking=measured,skip_first_last=False,skip_top_k=1)` | 536.50 | 570.87 | 0.06x | 11.23 | 67.19% | +0.39 | 0.965 |
| 8 | `quantize_dynamic_sensitive(per_channel=True,ranking=measured,skip_first_last=False,skip_top_k=2)` | 510.49 | 611.96 | 0.07x | 12.92 | 67.19% | +0.39 | 0.965 |
| 9 | `quantize_dynamic_sensitive(per_channel=True,ranking=measured,skip_first_last=False,skip_top_k=4)` | 420.78 | 441.45 | 0.08x | 19.77 | 67.19% | +0.39 | 0.973 |
| 10 | `graph_optimize(level=all) -> quantize_static_int8(activation_type=uint8,calib_samples=64,calibrate_method=minmax,per_channel=True,reduce_range=False)` | failed | | | | | | |

## Pareto frontier

Non-dominated across (latency p50, size, top-1). No single row is 'best' — pick by whichever constraint binds you.

| # | recipe | p50 ms | p99 ms | speedup | size MB | top-1 | Δpp | agreement |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | `quantize_static_int8(activation_type=uint8,calib_samples=64,calibrate_method=minmax,per_channel=False,reduce_range=False)` | 32.00 | 34.86 | 1.08x | 11.20 | 67.97% | +1.17 | 0.953 |
| 4 | `quantize_dynamic_int8(per_channel=False,reduce_range=False,weight_type=int8)` | 577.40 | 697.91 | 0.06x | 11.20 | 67.97% | +1.17 | 0.965 |

## Recommended pick

**`quantize_static_int8(activation_type=uint8,calib_samples=64,calibrate_method=minmax,per_channel=False,reduce_range=False)`** — trial 6.

- 1.08x faster (p50)
- +1.17pp top-1
- 25% of baseline size

> Trial [3] broke accuracy with per-channel scales; per-tensor scales keep most weights well below full range.

## Portability caveats

A fast number on this machine is not automatically a fast number on the deployment target. These artifacts carry hardware-specific assumptions:

- **Trial 1** (`graph_optimize(level=all)`): level='all' bakes in NCHWc layout transforms specific to the CPU that ran the optimisation; this artifact is only valid on matching hardware. Use level='extended' for a portable file.
- **Trial 10** (`graph_optimize(level=all) -> quantize_static_int8(activation_type=uint8,calib_samples=64,calibrate_method=minmax,per_channel=True,reduce_range=False)`): level='all' bakes in NCHWc layout transforms specific to the CPU that ran the optimisation; this artifact is only valid on matching hardware. Use level='extended' for a portable file.

## Failed trials

Recorded rather than hidden — a transform that does not apply is a real property of this model and target.

- `graph_optimize(level=all) -> quantize_static_int8(activation_type=uint8,calib_samples=64,calibrate_method=minmax,per_channel=True,reduce_range=False)` — ValueError: Unable to get valid quantization scale for input 'reorder_token_1' when quantizing bias 'onnx::Conv_197' to int32.

## Reproducing

Every row above is a recipe. To rebuild one, apply its transform chain to the baseline model with the same parameters; the ledger JSON alongside this report records the exact parameters, the machine fingerprint and the measurement protocol used.
