# Saturation analyser vs. the hardware lab

*x86-only loss* = mean accuracy change on x86 CPUs without VNNI minus the mean on CPUs with 32-bit INT8 accumulation (VNNI x86, ARM). Negative means the recipe loses accuracy only where saturation is possible. Beyond 1.5pp counts as observed.

| model | recipe | on 32-bit CPUs | on saturating x86 | x86-only loss | predicted | verdict |
|---|---|---:|---:|---:|---|---|
| efficientnet_b0 | U8S8 per-channel | -49.3pp | -50.4pp | -1.1pp | 32 layer(s), worst 19% | false alarm |
| efficientnet_b0 | U8S8 per-ch + reduce_range | -46.8pp | -46.7pp | +0.1pp | impossible | correct all-clear |
| efficientnet_b0 | S8S8 per-channel | -49.3pp | -47.8pp | +1.6pp | 32 layer(s), worst 19% | not modelled (S8S8) |
| efficientnet_b0 | U8S8 per-tensor | -75.4pp | -75.4pp | +0.0pp | 6 layer(s), worst 9% | false alarm |
| efficientnet_b0 | S8S8 per-tensor | -75.4pp | -75.4pp | +0.0pp | 6 layer(s), worst 9% | not modelled (S8S8) |
| mobilenet_v3_large | U8S8 per-channel | -12.2pp | -22.8pp | -10.5pp | 19 layer(s), worst 14% | hit |
| mobilenet_v3_large | U8S8 per-ch + reduce_range | -11.7pp | -11.0pp | +0.6pp | impossible | correct all-clear |
| mobilenet_v3_large | S8S8 per-channel | -12.2pp | -12.1pp | +0.1pp | 19 layer(s), worst 14% | not modelled (S8S8) |
| mobilenet_v3_large | U8S8 per-tensor | -37.2pp | -41.6pp | -4.4pp | 1 layer(s), worst 2% | miss |
| mobilenet_v3_large | S8S8 per-tensor | -37.2pp | -38.2pp | -1.0pp | 1 layer(s), worst 2% | not modelled (S8S8) |
| resnet18 | U8S8 per-channel | +1.1pp | -4.2pp | -5.3pp | 1 layer(s), worst 21% | hit |
| resnet18 | U8S8 per-ch + reduce_range | +0.2pp | +0.4pp | +0.2pp | impossible | correct all-clear |
| resnet18 | S8S8 per-channel | +1.1pp | -4.3pp | -5.4pp | 1 layer(s), worst 21% | not modelled (S8S8) |
| resnet18 | U8S8 per-tensor | +1.0pp | +1.3pp | +0.3pp | 1 layer(s), worst 1% | correct all-clear |
| resnet18 | S8S8 per-tensor | +1.0pp | +1.5pp | +0.5pp | 1 layer(s), worst 1% | not modelled (S8S8) |

**Tally (U8S8 recipes):** hit: 2, miss: 1, false alarm: 2, correct all-clear: 4, not modelled: 6

S8S8 recipes are listed but not scored: the lab shows the S8S8 kernel on non-VNNI x86 does not consistently use the saturating path the analyser assumes.
