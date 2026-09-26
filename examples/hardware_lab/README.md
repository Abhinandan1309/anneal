# Hardware lab

Runs the static INT8 recipe study ([`../static_recipe_ab.py`](../static_recipe_ab.py)) on
several kinds of GitHub-hosted machine — x86 Linux, ARM64 Linux, Windows, macOS — so that a
finding made on one laptop can be checked on other silicon, publicly and repeatably.

The question it was built for: on a Zen 2 laptop (no VNNI), full-range per-channel static
INT8 lost ~4pp on ResNet-18 and `reduce_range` fixed it. Is that an artefact of x86 without
VNNI, as onnxruntime's documentation would suggest, or does it happen everywhere?

## Results (first run, 22 Sep 2026)

[`results/results.md`](results/results.md) has the full table. In short: full-range
per-channel static INT8 loses 4.2pp on exactly the x86 CPUs without VNNI (a Zen 2 laptop and
an AMD EPYC 7763) and nothing on an Intel Xeon with AVX-512 VNNI, an ARM Neoverse-N2 or an
Apple M1. S8S8 is the slowest recipe on every x86 chip and the fastest on both ARM chips.
Per-machine raw data is in [`results/`](results/).

Caveats: one run per machine, on shared cloud hardware whose CPU is assigned, not chosen;
latency drift was 0.6–5.7%, all under the 10% tolerance.

> **Update (26 Sep 2026).** Two points above have been superseded.
>
> - **`reduce_range` is no longer the recommended fix for ReLU CNNs.** The per-layer analyser
>   later found that the saturation on ResNet-18 sits in the stem convolution, and keeping only
>   the stem in float removes it (−4.20pp → +1.27pp on the laptop). On ImageNet (ResNet-50,
>   10,000 images, the same non-VNNI laptop, real kernels), symmetric 99.999 percentile + float
>   stem loses 0.04pp, while the same calibration with `reduce_range` instead of the float stem
>   loses 0.63pp ([`../advise/resnet50_ablation.json`](../advise/resnet50_ablation.json)). The
>   advisor's plain-CNN rule is now symmetric 99.999 percentile + float stem, with no
>   `reduce_range` (commit 906393b). `reduce_range` is still advised for SiLU networks
>   (EfficientNet) on non-VNNI x86.
> - **Later lab runs** are not described on this page:
>   [`results-equalize/results.md`](results-equalize/results.md) (EfficientNet-B0 and
>   MobileNetV3-Large with equalisation, six machines) and
>   [`results-3models/`](results-3models/) (the saturation analyser's predictions against three
>   models, with [`scorecard.md`](results-3models/scorecard.md)). In the equalisation run, the
>   Apple M1 job drifted 16.1%, so its speedups are unreliable.
>
> The first-run finding itself stands: the ResNet-18 loss appears on exactly the x86 CPUs without
> VNNI.

## Running it

On GitHub: **Actions → hardware-lab → Run workflow**. Or from a terminal:

```bash
gh workflow run hardware-lab.yml -f eval_limit=1024 -f target=cpu-1t
```

Each machine records its CPU and INT8-relevant instruction-set features (VNNI on x86, the
dot-product extension on ARM), runs the five-recipe study, and re-times FP32 at the end. A
final job combines everything into one table, shown in the run's summary page and uploaded
as the `hardware-lab-summary` artifact.

Locally, on any machine:

```bash
pip install py-cpuinfo
python examples/hardware_lab/run_lab.py --eval-limit 1024 --out lab-local.json
python examples/hardware_lab/summarize.py lab-*.json
```

## Reading the results

- **Accuracy** is exact for each machine's kernels and unaffected by timing noise.
- **Speedups** come from shared cloud machines. A machine whose FP32 latency drifted more
  than 10% between the start and end of its job is marked ⚠; treat its speedups as
  unreliable.
- You do not choose the exact CPU a hosted runner gets; the job reports which one it was.
