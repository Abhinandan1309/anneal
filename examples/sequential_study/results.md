# Sequential acceptance: simulation study (1000 runs per scenario)

Budget 1.0pp, alpha = 0.05, full set 3925 images.

| scenario | true Δ | truth | fixed-256 error | sequential error | undecided | median n | p90 n | fixed-n for 90% power |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| static INT8, full-range per-channel | -4.20pp | reject | 6.6% | 0.0% | 0.0% | 724 | 1802 | 909 |
| static INT8, reduce_range | +0.39pp | accept | 3.8% | 0.0% | 0.0% | 600 | 1383 | 778 |
| Olive default static INT8 | +0.33pp | accept | 13.6% | 0.0% | 7.8% | 1406 | 3561 | 1,810 |
| near budget, acceptable | -0.50pp | accept | 41.4% | 1.4% | 91.0% | 3925 | 3925 | 32,535 |
| near budget, unacceptable | -1.50pp | reject | 42.5% | 1.0% | 90.5% | 3925 | 3925 | 34,179 |
| exactly on budget | -1.00pp | boundary | — | — | 95.0% | 3925 | 3925 | — |

On the boundary scenario there is no correct answer; its accept rate under each rule shows how each behaves when the evidence cannot decide:

- fixed-256 accepts 48.5%, fixed-3925 accepts 50.6%, sequential accepts 2.8% and stays undecided 95.0%.
