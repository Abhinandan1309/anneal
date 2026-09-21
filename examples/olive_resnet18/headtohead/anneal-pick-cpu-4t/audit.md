# Audit: `quantize-static-int8-activation-type-uint8-calib-samples-64--3fa25575a7.onnx` vs `resnet18-fp32.onnx`

Target `cpu-4t`, 3925 evaluation images.

## Verdict

- SLOWER, not faster: 1.08x the original's median latency on cpu-4t.
- No detectable accuracy change at n=3925 (p = 0.19), but the data cannot rule out a loss of up to 0.17pp.
- 7.8% of predictions changed class — far more than the accuracy delta alone suggests, because regressions and fixes partly cancel.

## Measurements

| | original | candidate | ratio |
|---|---:|---:|---:|
| p50 latency | 14.12 ms | 15.27 ms | 0.92x faster |
| p99 latency | 15.47 ms | 16.08 ms | 0.96x faster |
| size | 44.58 MB | 11.20 MB | 25% |
| top-1 | 66.88% | 67.31% | +0.43pp |

## Paired accuracy test

- Images the original got right and the candidate got wrong: **65**
- Images the original got wrong and the candidate got right: **82**
- Images whose predicted class changed at all: **308** (7.8%)
- Accuracy delta: **+0.43pp**, 95% CI [-0.17, +1.04]
- Exact McNemar p-value: **0.187**

## Per class

| class | n | original | candidate | Δpp |
|---|---:|---:|---:|---:|
| gas pump | 419 | 69.2% | 66.1% | -3.1 |
| garbage truck | 389 | 62.2% | 61.2% | -1.0 |
| French horn | 394 | 71.6% | 71.1% | -0.5 |
| golf ball | 399 | 87.2% | 87.0% | -0.3 |
| church | 409 | 29.6% | 29.8% | +0.2 |
| chain saw | 386 | 62.7% | 63.2% | +0.5 |
| tench | 387 | 92.2% | 93.0% | +0.8 |
| parachute | 390 | 79.7% | 80.8% | +1.0 |
| English springer | 395 | 60.8% | 62.8% | +2.0 |
| cassette player | 357 | 53.8% | 59.1% | +5.3 |
