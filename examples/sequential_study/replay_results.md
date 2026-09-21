# Sequential test replayed on real outcomes (1000 random orders each)

ResNet-18 on all 3925 Imagenette validation images. Budget 1.0pp, alpha = 0.05. The reference is the full-set decision.

| candidate | full-set change | reference | sequential agrees | contradicts | undecided | median n | p90 n | fixed-256 contradicts |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| static INT8, full-range per-channel | -4.31pp | reject | 99.9% | 0.1% | 0.0% | 612 | 1340 | 3.4% |
| static INT8, reduce_range | +0.54pp | accept | 100.0% | 0.0% | 0.0% | 630 | 1356 | 4.4% |
| Olive default static INT8 | +0.33pp | accept | 100.0% | 0.0% | 0.0% | 1463 | 2785 | 13.1% |
