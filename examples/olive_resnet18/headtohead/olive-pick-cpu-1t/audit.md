# Audit: `model.onnx` vs `resnet18-fp32.onnx`

Target `cpu-1t`, 3925 evaluation images.

## Verdict

- No meaningful speedup on cpu-1t (1.01x).
- No detectable accuracy change at n=3925 (p = 0.23), but the data cannot rule out a loss of up to 1.23pp.
- That is wider than a 1.0pp budget: audit on more images before signing this off.
- 12.5% of predictions changed class — far more than the accuracy delta alone suggests, because regressions and fixes partly cancel.

## Measurements

| | original | candidate | ratio |
|---|---:|---:|---:|
| p50 latency | 32.89 ms | 32.72 ms | 1.01x faster |
| p99 latency | 37.97 ms | 40.38 ms | 0.94x faster |
| size | 44.58 MB | 11.18 MB | 25% |
| top-1 | 66.88% | 66.39% | -0.48pp |

## Paired accuracy test

- Images the original got right and the candidate got wrong: **122**
- Images the original got wrong and the candidate got right: **103**
- Images whose predicted class changed at all: **489** (12.5%)
- Accuracy delta: **-0.48pp**, 95% CI [-1.23, +0.26]
- Exact McNemar p-value: **0.23**

## Per class

| class | n | original | candidate | Δpp |
|---|---:|---:|---:|---:|
| English springer | 395 | 60.8% | 58.5% | -2.3 |
| chain saw | 386 | 62.7% | 60.6% | -2.1 |
| golf ball | 399 | 87.2% | 85.2% | -2.0 |
| gas pump | 419 | 69.2% | 67.3% | -1.9 |
| parachute | 390 | 79.7% | 78.7% | -1.0 |
| French horn | 394 | 71.6% | 71.6% | +0.0 |
| garbage truck | 389 | 62.2% | 62.7% | +0.5 |
| tench | 387 | 92.2% | 92.8% | +0.5 |
| cassette player | 357 | 53.8% | 55.5% | +1.7 |
| church | 409 | 29.6% | 31.5% | +2.0 |
