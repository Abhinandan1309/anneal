# Anneal

**An agent that optimises neural networks for the hardware they'll actually run on.**

Anneal proposes a model transform, applies it, benchmarks it on a real runtime, and
conditions its next proposal on what it measured. It returns a Pareto frontier across
latency, accuracy and size, a ledger recording every trial including the ones that
failed, and a script that rebuilds any point on the frontier.

Nothing in that ledger is an estimate.

```bash
anneal run --model torchvision:resnet18 --target cpu-1t --eval imagenette --budget 10
```

---

## Why this exists

Deploying a model to constrained silicon is still artisanal. An engineer tries INT8,
checks accuracy, tries per-channel scales, recompiles, measures, tries sparing a layer,
measures again. It is a search — a slow one, run by hand, usually abandoned after three
or four attempts, with no record of what was tried.

Two things make it worth automating:

1. **The answers are counter-intuitive and target-specific.** In the run below, textbook
   INT8 dynamic quantization makes ResNet-18 **13.4x slower** on an ordinary x86 CPU. No
   amount of reasoning from first principles gets you there. You have to run it.
2. **"Faster" is not a scalar.** Faster at what accuracy, what size, what tail latency? A
   tool that returns one model has already made a decision that belongs to the engineer.

---

## The loop

```
    ┌──────────────────────────────────────────────┐
    │  policy — what should we try next?           │
    │  heuristic curriculum, or Claude             │
    └───────────────────┬──────────────────────────┘
                        │ proposal
                        ▼
    ┌──────────────────────────────────────────────┐
    │  transform — quantize / fuse / spare layers  │
    │  produces a real .onnx on disk               │
    └───────────────────┬──────────────────────────┘
                        │ candidate
                        ▼
    ┌──────────────────────────────────────────────┐
    │  MEASURE on the real runtime                 │
    │  warm-up discarded · p50/p90/p99             │
    │  top-1 on real images · baseline agreement   │
    └───────────────────┬──────────────────────────┘
                        │ measurement
                        ▼
    ┌──────────────────────────────────────────────┐
    │  ledger → Pareto frontier                    │
    └───────────────────┬──────────────────────────┘
                        │
                        └──► back to the policy
```

The policy never sees anything but measured numbers.

---

## A real run

ResNet-18, single-threaded CPU, 256 Imagenette images scored with a full 1000-way argmax,
10 warm-up iterations discarded and 50 timed runs per trial, 10-trial budget, 1pp accuracy
budget. Full artifacts in [`examples/resnet18-cpu1t/`](examples/resnet18-cpu1t/).

![Pareto frontier](examples/resnet18-cpu1t/frontier.svg)

| # | recipe | p50 ms | p99 ms | speedup | size MB | top-1 | agreement |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | baseline fp32 | 46.74 | 60.52 | 1.00x | 44.58 | 66.80% | — |
| 1 | graph fusion | 43.65 | 51.27 | **1.07x** | 44.58 | 66.80% | 1.000 |
| 2 | dynamic INT8, per-channel | 627.90 | 772.16 | **0.07x** | 11.21 | 67.97% | 0.961 |
| 3 | static INT8 (QDQ) | 35.06 | 44.51 | **1.33x** | 11.28 | 64.06% | 0.844 |
| 4 | dynamic INT8, per-tensor | 673.75 | 841.71 | 0.07x | 11.20 | 67.97% | 0.965 |
| 5 | selective INT8, spare k=1 | 562.16 | 691.86 | 0.08x | 12.89 | 67.58% | 0.961 |
| 6 | selective INT8, spare k=2 | 574.44 | 883.40 | 0.08x | 13.00 | 66.80% | 0.961 |
| 7 | selective INT8, spare k=4 | 549.71 | 746.11 | 0.09x | 20.17 | 67.58% | 0.961 |
| 8 | fusion → static INT8 | *failed* | | | | | |
| 9 | static INT8, entropy calib | 35.45 | 46.56 | 1.32x | 11.28 | 64.06% | 0.844 |
| 10 | selective INT8, spare stem+head | 578.05 | 770.82 | 0.08x | 12.92 | 67.19% | 0.965 |

### What the run actually found

**The textbook first move is a disaster here.** Dynamic INT8 — the transform every
quantization tutorial opens with — makes the model 13.4x *slower*. `anneal profile`
itemises exactly why:

```console
$ anneal profile resnet18-fp32.onnx --against dyn-int8.onnx --target cpu-1t

| op                          | before ms | after ms | delta ms |
|-----------------------------+-----------+----------+----------|
| ConvInteger (new)           |      0.00 |  1695.74 | +1695.74 |
| Add (new)                   |      0.00 |     3.44 |    +3.44 |
| Relu (new)                  |      0.00 |     1.76 |    +1.76 |
| DynamicQuantizeLinear (new) |      0.00 |     2.23 |    +2.23 |
| Conv (gone)                 |     86.45 |     0.00 |   -86.45 |
```

In FP32, onnxruntime fuses Conv+Add+Relu into one NCHWc-layout kernel: 86ms. Quantizing
replaces it with `ConvInteger`, which has no such kernel — 1696ms — and `Add` and `Relu`
reappear as separate nodes because `ConvInteger` cannot absorb them. The 13.4x is not a
mystery; it is one missing kernel.

**The policy formed a hypothesis and refuted it.** Seeing trial 2's regression, the
heuristic proposed per-tensor scales as the likely cause. Trial 4 measured it: 673.75ms,
no better. The hypothesis was wrong and the ledger says so.

**Accuracy alone would have hidden the real risk.** Trial 3 looks like a 2.73pp accuracy
drop. Its `agreement` of 0.844 says **15.6% of predictions changed** — a far larger
behavioural shift than the aggregate suggests, because errors partly cancel. (Validation on
the full set, below, shows the real drop is larger still.) Meanwhile
trial 2's *gain* of +1.17pp is noise (see below). Agreement is paired per-image; accuracy
deltas are two aggregates subtracted.

**A failure is recorded, not hidden.** Trial 8 — fusion then static quantization — failed
with `Unable to get valid quantization scale for input`. The NCHWc-transformed graph is
not statically quantizable. That is a real property of this toolchain and it stays in the
ledger.

**The honest answer is unexciting, which is the point.** Under a 1pp accuracy budget on
this target, the recommended pick is graph fusion at **1.07x** — because the 1.33x from
static INT8 costs more accuracy than the budget allows. A tool that returned "1.33x
speedup!" would have been lying by omission.

### Confirming the finalists on the full eval set

The search scores every trial on 256 images, which is cheap enough to run eleven times
and resolves only ±5.7pp. `anneal validate` re-scores just the frontier on all 3,925
Imagenette validation images (±1.47pp), skipping candidates already disqualified on
latency.

```console
$ anneal validate examples/resnet18-cpu1t/ledger.json --eval imagenette
```

| # | recipe | speedup | top-1 (n=256) | top-1 (n=3925) | Δpp vs baseline | agreement |
|---:|---|---:|---:|---:|---:|---:|
| 0 | baseline fp32 | 1.00x | 66.80% | 66.88% | — | — |
| 1 | graph fusion | 1.07x | 66.80% | 66.88% | 0.00 | 1.000 |
| 3 | static INT8 | 1.33x | 64.06% | **62.60%** | **−4.28** | **0.824** |

**The small eval set was optimistic.** The search saw static INT8 cost 2.73pp; the full set
says 4.28pp — the damage was understated by more than a third. At n=3,925 that drop is well
outside the ±1.47pp interval, and 17.6% of predictions change. Fusion is confirmed exactly
lossless: identical accuracy, every prediction unchanged. The recommendation from the
search stands, now on evidence rather than on a sample too small to separate the options.

This is why `validate` exists as a separate step. Search cheap, confirm the shortlist
properly, and never quote the search's accuracy numbers as the answer.

### Does the cheap sensitivity proxy actually work?

Selective quantization chooses which layers to keep in FP32 using a free weight-space
proxy: the relative L2 error INT8 introduces into each layer's weights. `anneal sensitivity
--measured` tests that proxy against ground truth by quantizing each of the 21 layers *alone*
and counting how many predictions change.

```console
$ anneal sensitivity resnet18-fp32.onnx --measured --eval-limit 256 --top 21
```

| layer | proxy rank | predictions changed |
|---|---:|---:|
| `/conv1/Conv` (stem) | **21 of 21** | **3.5%** |
| `/layer3/layer3.0/conv2/Conv` | 1 | 1.6% |
| `/layer1/layer1.0/conv1/Conv` | 2 | 1.6% |
| `/layer4/layer4.0/conv2/Conv` | 3 | 1.6% |
| `/layer2/layer2.0/conv1/Conv` | 15 | 1.6% |
| … 16 more layers | | 0.0 – 1.2% |

**Spearman ρ = +0.33. The proxy is weak.** It finds 3 of the 5 most damaging layers, but it
ranks the single most sensitive layer — the stem convolution, which does more than twice
the damage of any other — dead last. That is not bad luck: the stem's sensitivity comes
from quantizing raw-pixel activations with a wide dynamic range, and a proxy that only looks
at weights cannot see activations at all.

What this means for the tool, stated plainly: the proxy is a usable prior for the middle of
the network and blind at the input. So selective quantization now ranks by measurement by
default (`ranking='measured'`), sweeping once per model per run, or reusing a saved sweep
via `anneal run --sensitivity sweep.json`. The proxy remains available as
`ranking='proxy'` for runs with no eval set.

Did that change help? [`examples/ranking_ab.py`](examples/ranking_ab.py) spares one layer
under each ranking and counts changed predictions on 1,024 images:

| spare 1 layer | chosen by | predictions changed vs FP32 |
|---|---|---:|
| `/layer3/layer3.0/conv2/Conv` | proxy | 6.15% (63 images) |
| `/conv1/Conv` (stem) | measurement | **5.08% (52 images)** |

Measured ranking changes 11 fewer predictions — the right direction, and consistent with the
sweep. But 11 images is roughly one standard deviation for a paired count this size, so this
is **suggestive, not conclusive**. The accuracy difference between the two (68.55% vs 67.97%)
is inside the ±2.9pp interval and means nothing. A larger `k` or eval set would be needed to
claim a real improvement.

A resolution caveat: at n=256 each image is 0.39pp, so the 0.4–1.6% tail is a handful of
images per layer and largely tied. The stem result — 9 images, more than double the next
layer — is the one clear signal. The correlation itself should be read as "weak", not as a
precise ρ.

### The same recipes on a different target

If optimisation results were a property of the model, this table would be boring.

```console
$ anneal compare examples/resnet18-cpu1t/ledger.json --target cpu-4t
```

| # | recipe | cpu-1t p50 | cpu-1t | cpu-4t p50 | cpu-4t | verdict |
|---:|---|---:|---:|---:|---:|---|
| 0 | baseline fp32 | 46.74 | 1.00x | 30.19 | 1.00x | holds |
| 3 | static INT8 | 35.06 | 1.33x | 27.52 | **1.10x** | shifts |
| 1 | graph fusion | 43.65 | 1.07x | 29.77 | **1.01x** | **flips** |
| 2 | dynamic INT8 | 627.90 | 0.07x | 539.45 | 0.06x | shifts |

**The recommended pick from the cpu-1t run is worth essentially nothing on cpu-4t.**
Fusion's 1.07x becomes 1.01x, because four threads already extract the parallelism that
fusion was buying. Static INT8's advantage shrinks from 1.33x to 1.10x for the same
reason. Same model, same recipes, different answer — from a change of *thread count*, not
even a change of silicon.

This is the entire thesis, and it is why the edge targets in `anneal targets` refuse to
fake it.

---

## What makes the measurements trustworthy

This took the most care, because an optimisation tool that reports flattering numbers is
worse than no tool at all.

**Warm-up is discarded.** onnxruntime allocates arenas and selects kernels on the first
few calls. Timing those measures the allocator.

**Percentiles, not means,** by nearest rank — so every reported latency is an observation
that actually happened, not an interpolation between two of them. An edge deadline cares
about p99.

**Silent provider fallback is a hard error.** onnxruntime quietly runs on CPU when the
execution provider you asked for is unavailable. Anneal compares the providers actually
used against those requested and refuses to label the result a `cuda` measurement if CPU
ran it. This is the most common source of fictional edge benchmarks.

**Candidates are compared to the baseline, not just to ground truth.** Every trial records
`top1_agreement` and `logit_cosine`. See trial 3 above for why this is not optional.

**The report states its own resolution.** At n=256 the 95% Wilson interval is ±5.7pp, so
several "differences" in the table above are ties. The report says this in its own header
rather than letting two decimal places imply precision that is not there.

**Failures stay in the ledger** with their error text — signal for the policy, honesty for
the reader.

**Accuracy comes from real images, scored honestly.** Imagenette with a *full 1000-way*
argmax. Restricting the softmax to the ten classes present would inflate top-1 by several
points and make every quantization result look safer than it is. The synthetic eval set
exists for plumbing tests and is labelled meaningless everywhere it appears.

---

## Two policies, on purpose

| | `--policy heuristic` | `--policy claude` |
|---|---|---|
| Needs an API key | no | yes |
| Deterministic | yes | no |
| Sees | the measured ledger | the same measured ledger |

`HeuristicPolicy` is a curriculum distilled from how edge engineers actually work: probe
each transform family, then react. Candidate came out slower than FP32? Retry per-tensor
before writing off the transform. Accuracy broke? Escalate selective quantization with
increasing `k`. Accuracy held? Push harder on speed. Every trial above came from it.

`ClaudePolicy` hands Claude the same rendered ledger each turn and lets it call a
`propose_transform` tool. Each turn is stateless — the ledger is re-rendered rather than
accumulated as conversation — so context stays bounded and each decision is auditable
alone.

Shipping both is deliberate. **An agentic system that cannot be compared against a
competent non-agentic baseline is a demo, not an engineering result.**

> **Disclosure:** the Claude policy is implemented and unit-tested against a stubbed
> client, but it has **not** been run against the live API — no key was available in the
> environment where this was built. Every measured number in this README came from
> `--policy heuristic`. Treat the LLM path as reviewed code, not as a validated result.

---

## Install

```bash
git clone https://github.com/Abhinandan1309/anneal
cd anneal
pip install -e ".[torch]"    # torch extra only needed to export torchvision models
```

For the Claude policy: `pip install -e ".[llm]"` and set `ANTHROPIC_API_KEY`.

## Commands

```bash
anneal run          # the search
anneal validate     # re-score the frontier on a much larger eval set
anneal compare      # re-measure the frontier on a different target
anneal profile      # attribute runtime to operators; --against to diff two models
anneal sensitivity  # rank layers by INT8 damage; --measured for the real sweep
anneal export       # emit a standalone script that rebuilds a frontier point
anneal report       # re-render from a ledger
anneal targets      # what can actually run here
anneal transforms   # the action space
```

Typical session:

```bash
# search cheap
anneal run --model torchvision:resnet18 --target cpu-1t --eval imagenette \
           --budget 10 --max-accuracy-drop 1.0

# confirm the finalists properly
anneal validate runs/latest/ledger.json --eval-limit 2048

# check it survives the deployment target
anneal compare runs/latest/ledger.json --target cpu-4t

# ship it
anneal export runs/latest/ledger.json --out deploy_recipe.py
```

---

## Real silicon

CPU targets run anywhere. Edge targets are *declared* and raise an actionable error rather
than being silently absent:

```console
$ anneal targets
| name          | status          | description                                 |
|---------------|-----------------|---------------------------------------------|
| cpu-1t        | available       | Single-threaded CPU.                        |
| cpu-4t        | available       | 4-thread CPU.                               |
| cuda          | runtime missing | NVIDIA GPU via the CUDA execution provider. |
| tensorrt      | runtime missing | NVIDIA GPU/Jetson via TensorRT.             |
| jetson-orin   | adapter needed  | Jetson Orin, TensorRT INT8/FP16 tactics.    |
| tda4vm        | adapter needed  | TI TDA4VM (J721E) C7x/MMA via TIDL.         |
| coral-edgetpu | adapter needed  | Google Coral Edge TPU. INT8-only.           |
```

Adding one means implementing `Target` and calling `register_target()`. The honest caveat,
stated in the code: **the sensitivity ranking transfers between targets; the absolute
latencies do not.** Develop the recipe on CPU, confirm it on the board.

---

## Layout

```
src/anneal/
  core/
    artifact.py     ModelArtifact + lineage. Every candidate is a real file with provenance.
    targets.py      What you are optimising *for*.
    transforms.py   The action space, and the weight-space sensitivity proxy.
    measure.py      The benchmark harness. The reason any of this is trustworthy.
    profile.py      Operator-level attribution. Why, not just how much.
    sensitivity.py  The expensive measured sweep, and the proxy's score against it.
    dataset.py      Imagenette, scored 1000-way. Labelled synthetic fallback.
    ledger.py       Append-only trial record + Pareto frontier.
  agent/
    policy.py       HeuristicPolicy and ClaudePolicy.
    prompts.py      Ledger rendering and tool schemas.
    loop.py         propose → apply → measure → record.
  report.py         Rich tables, Markdown, dependency-free SVG.
  cli.py
```

```bash
pytest tests/ -q
```

---

## Limitations

Stated plainly, because the alternative is letting someone find them in a review:

- **These are CPU numbers,** measured on the machine named in the report fingerprint. They
  do not transfer to a Jetson or a TDA4VM — as the `compare` table shows, they barely
  transfer across a thread count. What transfers is the method.
- **The search's eval set is small.** 256 images resolves ±5.7pp, and in the run above it
  understated static INT8's damage by more than a third. `anneal validate` exists because of
  this, and the report prints its own resolution.
- **The layer-sensitivity proxy is weak** (ρ = +0.33 against the measured sweep) and blind to
  the stem layer. Selective quantization now ranks by measurement by default, but the evidence
  that this protects accuracy better is so far a single, marginal A/B (52 vs 63 changed
  predictions of 1,024).
- **The action space is quantization and graph optimisation.** No pruning, distillation, or
  NAS. Those are real transforms with real payoffs and they are absent; the registry is the
  extension point.
- **The Claude policy has not been run against the live API.** See the disclosure above.
- **`graph_optimize(level='all')` produces a non-portable artifact** — onnxruntime's NCHWc
  transformer bakes in the optimising CPU's layout and SIMD width. Anneal records this on
  the artifact and warns in the report. Use `level='extended'` if the file must travel.

---

## Licence

MIT.
