# Audit: `model.onnx` vs `resnet18-fp32.onnx`

Target `cpu-4t`, 3925 evaluation images.

## Verdict

- SLOWER, not faster: 1.38x the original's median latency on cpu-4t.
- No detectable accuracy change at n=3925 (p = 0.32), but the data cannot rule out a loss of up to 0.27pp.
- 8.1% of predictions changed class — far more than the accuracy delta alone suggests, because regressions and fixes partly cancel.

## Measurements

| | original | candidate | ratio |
|---|---:|---:|---:|
| p50 latency | 14.03 ms | 19.42 ms | 0.72x faster |
| p99 latency | 18.48 ms | 25.07 ms | 0.74x faster |
| size | 44.58 MB | 11.20 MB | 25% |
| top-1 | 66.88% | 67.21% | +0.33pp |

## Paired accuracy test

- Images the original got right and the candidate got wrong: **67**
- Images the original got wrong and the candidate got right: **80**
- Images whose predicted class changed at all: **319** (8.1%)
- Accuracy delta: **+0.33pp**, 95% CI [-0.27, +0.94]
- Exact McNemar p-value: **0.322**

## Per class

| class | n | original | candidate | Δpp |
|---|---:|---:|---:|---:|
| gas pump | 419 | 69.2% | 65.9% | -3.3 |
| French horn | 394 | 71.6% | 69.8% | -1.8 |
| garbage truck | 389 | 62.2% | 61.4% | -0.8 |
| golf ball | 399 | 87.2% | 87.2% | +0.0 |
| church | 409 | 29.6% | 29.8% | +0.2 |
| tench | 387 | 92.2% | 93.0% | +0.8 |
| chain saw | 386 | 62.7% | 63.5% | +0.8 |
| parachute | 390 | 79.7% | 81.0% | +1.3 |
| English springer | 395 | 60.8% | 62.5% | +1.8 |
| cassette player | 357 | 53.8% | 58.8% | +5.0 |
