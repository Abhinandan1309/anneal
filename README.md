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

**…and it was also incomplete.** Auditing Microsoft Olive's output later showed that
recipes already in Anneal's action space — static INT8 with `reduce_range`, or with
per-tensor scales — keep full accuracy. This search never tried them on the plain graph. The
policy now does, and a rerun with the fix, on AC power with a 2.5% baseline drift
([`examples/resnet18-cpu1t-v2/`](examples/resnet18-cpu1t-v2/)), picks static INT8 per-tensor:
**1.08x at no accuracy loss** (audited on all 3,925 images: +0.43pp, any loss ≤ 0.17pp). In
the rerun, fusion measures 0.97x, so the 1.07x above was within run-to-run noise. On one
thread the speed prize for INT8 on this CPU is small; the honest headline is how small, and
how certain. The run above is committed as it happened.

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

### A second model: MobileNetV3-Large says "don't"

Same protocol, same target, same budget. Full artifacts in
[`examples/mobilenetv3-cpu1t/`](examples/mobilenetv3-cpu1t/).

| # | recipe | p50 ms | speedup | size MB | top-1 | agreement |
|---:|---|---:|---:|---:|---:|---:|
| 0 | baseline fp32 | 7.03 | 1.00x | 20.9 | 71.48% | — |
| 1 | graph fusion | 7.25 | 0.97x | 20.9 | 71.48% | 1.000 |
| 2 | dynamic INT8, per-channel | 105.46 | 0.07x | 5.5 | 41.41% | 0.453 |
| 3 | static INT8 (QDQ) | 8.25 | **0.85x** | 5.7 | **42.19%** | **0.480** |
| 5 | static INT8, per-tensor | 8.77 | 0.80x | 5.5 | 20.31% | 0.223 |
| 6 | selective INT8, spare k=1 (measured) | 97.63 | 0.07x | 5.5 | 49.22% | 0.602 |
| 7 | selective INT8, spare k=2 (measured) | 100.96 | 0.07x | 5.5 | 50.00% | 0.625 |
| 8 | selective INT8, spare k=4 (measured) | 110.75 | 0.06x | 5.5 | 57.42% | 0.738 |

**Nothing helps, and the tool says so.** Static INT8 — the transform that bought 1.33x on
ResNet-18 — is *slower* here, and top-1 collapses from 71.5% to 42.2% with more than half
of all predictions changed. Fusion is within noise of the baseline. The recommended pick is
the unmodified FP32 model. MobileNetV3's depthwise convolutions, hard-swish activations and
squeeze-excite blocks are known to resist post-training quantization; it needs
quantization-aware training, which is outside this action space. The value here is the
negative result arriving in minutes, with the evidence attached, instead of after a week of
hand-tuning.

**The measured ranking finds the fragile layers on its own.** The four layers it chose to
spare are all depthwise convolutions — the layer type the literature singles out as
quantization-sensitive — and accuracy climbs monotonically as more are spared (49.2% →
50.0% → 57.4%, agreement 0.60 → 0.74). Still nowhere near FP32, and still on the slow
`ConvInteger` path, but it is the ranking doing its job on an unfamiliar architecture.

**The ledger caught a bug in the policy.** Trial 9 is `fusion → fusion`: the heuristic's
"stack the fastest candidate on the fused graph" rule assumed the fastest candidate would be
a quantization. On MobileNetV3 it was fusion itself, so the rule proposed fusing twice. The
transform refused it and the trial was recorded as a failure — which is how it was noticed —
and the rule now only stacks quantizations. This run predates the fix and is committed as it
happened.

---

## Auditing other tools' output

Anneal's search is one way to get an optimised model. Microsoft Olive, Intel Neural
Compressor, NNCF and TensorRT are others, with far larger teams and action spaces. Anneal
does not try to out-optimise them. `anneal audit` checks **any** optimised model against its
original, on the target it will run on:

```console
$ anneal audit original.onnx candidate.onnx --target cpu-4t --eval-limit 4000 --profile
```

It compares the two models **image by image**: regressions (original right, candidate
wrong), fixes (the reverse), an exact McNemar test on those, a paired 95% interval on the
accuracy change — including how large a loss the data *cannot* rule out — plus how many
predictions changed class, a per-class breakdown, p50/p99 speedup, and which operators the
latency moved to. It exits non-zero on a significant loss beyond budget or a slowdown, so
it can gate CI.

**Olive head-to-head on ResNet-18** (full write-up in
[`examples/olive_resnet18/`](examples/olive_resnet18/)):

- Olive's default static quantization, by Olive's own report, came out **1.70x slower**,
  and was written out as the result because the workflow set no search objective.
- The audit agrees it is slower on this target (0.72x on 4 threads), and establishes
  something Olive's 256-image evaluation could not: accuracy is **genuinely unchanged**
  (+0.33pp on 3,925 images, McNemar p = 0.32, any loss ≤ 0.27pp) — while **8.1% of
  predictions change class**.
- Olive's recipe kept accuracy where Anneal's own static INT8 lost 4.28pp. A controlled
  experiment traced it to **full-range per-channel weights**, not activation signedness as
  I first guessed: adding `reduce_range` restores full accuracy. The heuristic policy now
  tries that fix first.
- **With its search enabled, Olive finds the same fix.** Audited identically, the two tools'
  final picks each win on the thread count they were chosen for, within measurement noise.
  The difference is in the accuracy: Anneal's pick is confirmed within the 1pp budget (any
  loss ≤ 0.17pp); Olive's cannot be — it was selected for "+1.56pp" on 256 images, which
  is four images, and on 3,925 images its possible loss extends to 1.23pp. Picking the
  best-looking of many candidates on a small sample reliably picks a lucky one.

**What the audit found was mostly wrong with Anneal, not with Olive** — and where it did
find something about Olive, it was about statistics, not optimisation. That is the point of
an auditor that does not care which tool produced the model.

> **Correction:** an earlier version of this section reported `reduce_range` at 1.52x FP32
> speed. That was measured on battery with battery saver engaged, which slowed FP32 by ~25%
> and INT8 hardly at all. On AC power the same recipe is 1.05x on one thread. Details in
> [`examples/olive_resnet18/`](examples/olive_resnet18/).

---

## Deciding with fewer images: anytime-valid sequential testing

Evaluation is the expensive part of every trial, and the cheap shortcut is unreliable: the
256-image eval set the search uses understated static INT8's damage by more than a third.
Checking a fixed-sample test after every batch and stopping when it "looks decided" is not a
fix — that quietly inflates the error rate.

`anneal audit --sequential` scores each image +1 (candidate fixed it), −1 (broke it) or 0,
whose mean is exactly the accuracy change, and runs two **betting tests** against the budget
(aGRAPA bet sizing, [Waudby-Smith & Ramdas, JRSS-B 2023](https://arxiv.org/abs/2010.09686)).
By Ville's inequality each decision's error stays below α **however early it stops**, so it
stops as soon as the evidence is decisive and says *undecided* when it is not.

Replayed on **real** per-image outcomes — every model's result on all 3,925 validation
images, streamed in 1,000 random orders each
([`examples/sequential_study/replay_results.md`](examples/sequential_study/replay_results.md)):

| candidate | full-set answer | sequential agrees | median images used | fixed 256 images wrong |
|---|---|---:|---:|---:|
| static INT8, full-range per-channel (−4.31pp) | reject | 99.9% | 612 | 3.4% |
| static INT8, reduce_range (+0.54pp) | accept | 100% | 630 | 4.4% |
| Olive's default static INT8 (+0.33pp) | accept | 100% | 1,463 | 13.1% |

A simulation study at measured rates
([`examples/sequential_study/results.md`](examples/sequential_study/results.md)) adds the
hard cases: within half a point of the budget the 256-image rule is **wrong 41–43% of the
time, silently**, while the sequential test stays within its 5% error bound and reports
*undecided* — correctly, since settling those would take ~33,000 images.

What this does and does not claim: on clear cases it uses about **5× fewer images than the
full set**, and about as few as a fixed test that was sized using advance knowledge of the
effect — the difference is that it needs no such knowledge and keeps its error guarantee at
any stopping point. It is not magic near the budget; it is honest there. The method is
from the statistics literature; applying it to model-optimisation acceptance is the part
this project contributes. The guarantee assumes images arrive in random order, which
Anneal's eval sets do.

---

## Related work

| | what it does | where Anneal differs |
|---|---|---|
| [Microsoft Olive](https://github.com/microsoft/Olive) | Hardware-aware optimisation workflows for ONNX/PyTorch, with a search over pass parameters and an evaluator. | Much broader, and vendor-integrated. Anneal's distinct part is the audit: paired significance testing, prediction-change counts, operator attribution, cross-target checks. |
| [Intel Neural Compressor](https://github.com/intel/neural-compressor) | Accuracy-driven quantization tuning with a tolerance loop. | Tunes to an accuracy target; Anneal reports the statistical resolution of that accuracy and what it cannot rule out. |
| [OpenVINO NNCF](https://github.com/openvinotoolkit/nncf), [Qualcomm AIMET](https://github.com/quic/aimet) | Compression and quantization, including quantization-aware training. | They can fix what post-training quantization breaks (MobileNetV3 here); Anneal cannot and says so. |
| TensorRT, TI TIDL tools | Vendor compilers that choose kernels for their own silicon. | Anneal declares these targets but does not implement them yet. |

This comparison reflects those projects as I understand them; check their current docs
before relying on it.

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

**The machine is checked, not trusted.** A rerun of the ResNet-18 search once reported a
recipe at 0.36x FP32 speed that had measured 1.5x before. Neither number was right. The
laptop had been on battery for both: in the first session battery saver slowed FP32 more
than INT8, inflating the speedup to 1.5x; in the second the battery fell to 21% mid-run and
the CPU dropped from 2.9 to 1.7 GHz, deflating it. On AC power the recipe is 1.05x. Throttling
does not just slow everything down — it changes *ratios*, so "compare within a session" is
not a sufficient safeguard. Anneal now reads AC/battery state, battery saver and clock ceilings before
measuring and warns loudly; re-times the baseline at the end of every search and marks the
whole run latency-untrustworthy if it drifted more than 10%; and `anneal audit` times the
original model before *and* after the candidate (A-B-A) to catch the same drift between two
models.

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
anneal audit        # check any optimised model (from any tool); --sequential stops early
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
- **The original ResNet-18 and MobileNetV3 search runs predate three fixes:** calibration on
  held-out images, the reduce_range retry, and the machine-state check. ResNet-18 has been
  re-run with all three (`examples/resnet18-cpu1t-v2/`); MobileNetV3 has not. The originals
  are kept as they ran rather than silently replaced.
- **Four-thread latencies on this laptop are not reliable.** Even on AC power, with low drift
  *within* each run, the same model measured 0.92x in one audit and 1.19x in the next.
  One-thread figures agree across runs to within a few percent, so claims here rest on them.
  The within-run drift check does not catch variation *between* runs; repeated,
  interleaved measurement would, and is not implemented yet.
- **Latencies measured before the environment check existed may be distorted.** Power state
  was not recorded for them, and one set is known to have run on battery (see the
  correction above). Figures from the first ResNet-18 and MobileNetV3 runs, the cross-target
  table and the earlier Olive audit fall in that period.
- **The saturation explanation is unconfirmed.** `reduce_range` fixing per-channel static
  INT8 on a Zen 2 CPU fits onnxruntime's documented non-VNNI saturation issue, but I have not
  run the same experiment on a VNNI CPU to confirm it.
- **`graph_optimize(level='all')` produces a non-portable artifact** — onnxruntime's NCHWc
  transformer bakes in the optimising CPU's layout and SIMD width. Anneal records this on
  the artifact and warns in the report. Use `level='extended'` if the file must travel.

---

## Licence

MIT.
