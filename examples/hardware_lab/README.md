# Hardware lab

Runs the static INT8 recipe study ([`../static_recipe_ab.py`](../static_recipe_ab.py)) on
several kinds of GitHub-hosted machine — x86 Linux, ARM64 Linux, Windows, macOS — so that a
finding made on one laptop can be checked on other silicon, publicly and repeatably.

The question it was built for: on a Zen 2 laptop (no VNNI), full-range per-channel static
INT8 lost ~4pp on ResNet-18 and `reduce_range` fixed it. Is that an artefact of x86 without
VNNI, as onnxruntime's documentation would suggest, or does it happen everywhere?

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
