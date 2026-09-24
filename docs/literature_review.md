# Literature review

Written in September 2026 to check what in Anneal is new and what is known, and to find the
published baselines it should be compared against. Three parts, researched separately:

1. [Equalisation across gated activations](#part-1--equalisation-across-gated-activations): prior art for `anneal.core.equalize`.
2. [Post-training quantization of efficient CNNs](#part-2--post-training-quantization-of-efficient-cnns): published W8A8 results and where Anneal's sit.
3. [Hardware-level INT8 and hardware-aware tools](#part-3--hardware-level-int8-and-hardware-aware-tools): what vendors document about x86 saturation, and how other tools measure.

## Summary of verdicts

| Anneal component | status | closest prior work |
|---|---|---|
| Inverse scale on the gate input of a self-gated activation, x·σ(x/s) | **Known.** Not a contribution. | I-LLM (arXiv:2405.17849); MambaQuant (arXiv:2501.13484) |
| The same construction applied to CNN static INT8 PTQ (SiLU and Hardswish into depthwise convs), with closed-form scales | **Not found.** An application and extension. | HPTQ / Sony MCT: equalises across ReLU only and names the Swish gap |
| Deliberately negative equalisation scales, to use an asymmetric range | **Not found** in the equalisation literature. | Qualcomm CLE patent requires non-negative scales; OS+ shifts instead of mirroring |
| The mechanism of x86 u8s8 saturation (VPMADDUBSW) and the 7-bit fix | **Documented** by Intel, oneDNN, FBGEMM, onnxruntime and PyTorch. | NNCF halves the weight range of the first layer only |
| How much saturation costs in accuracy (−8 to −18pp on 9 CNNs, ~0 on 32-bit hardware) | **No figures found.** Vendors describe the effect as insignificant or give no numbers. | — |
| Per-layer prediction of saturation without the affected CPU | **Not found.** OpenVINO's documentation calls it impossible to predict. | an experimental onnxruntime build flag (runtime, one kernel) |
| EfficientNet-B0 static INT8 collapse | **Known.** NVIDIA reports 76.85 → 22.3% with max calibration (Wu et al. 2020). | best published training-free results −3.0pp (HPTQ), −4.8pp (NVIDIA) |

Caveats that apply to Anneal's own numbers:
- **Imagenette is not ImageNet.** Anneal's results are on Imagenette, a 10-class subset scored 1000-way, which is easier than ImageNet-1k. The drops are not directly comparable.
- **SQNR does not predict top-1 drop.** A September 2026 preprint finds that per-layer SQNR does not predict the drop, so `anneal imbalance` is a diagnostic, not an accuracy predictor.

---

# Part 1 — Equalisation across gated activations


Technique under review (`src/anneal/core/equalize.py`): for `x = ConvA(.); y = x * g(x); z = DWConvB(y)`
with `g` in {Sigmoid, HardSigmoid}, set `ConvA_c *= s_c`, feed the gate `x'/s_c`, and set `DWConvB_c /= s_c`.
The float function is unchanged. Per-channel weight quantization absorbs `s`. `s_c` may be negative, which
mirrors channels in SiLU's negative lobe into the positive side of an asymmetric uint8 range. The rewrite
runs before static per-tensor INT8 PTQ of CNNs (EfficientNet-B0, MobileNetV3).

Searched September 2026. Full text was read (pypdf extraction) for every arXiv paper listed below. MCT and
NNCF were checked in their source code. The Qualcomm and Perceive/Amazon patents were checked through
Google Patents.

---

## 1. Cross-layer equalisation (CNN PTQ)

**DFQ / CLE, Nagel et al. 2019, https://arxiv.org/abs/1906.04721.** Cross-layer range equalisation relies
on positive scaling equivariance, f(sx) = s f(x) for s >= 0. The paper states this for ReLU and PReLU and
extends it to piecewise-linear functions by reparameterising their breakpoints. ReLU6 is *replaced by ReLU*
before equalisation. Nothing is said about Swish or sigmoid, so they are implicitly out of scope. Scales are
non-negative by construction. The paper equalises weight ranges; activation ranges are handled only
indirectly, through high-bias absorption.

**Same, Same But Different, Meller et al. 2019 (ICML), https://arxiv.org/abs/1902.01917.** This is the
concurrent "inversely proportional factorisation" paper. It scales output channel i of L1 by c_i > 0 and
the matching input weights of L2 by 1/c_i. The activation must be homogeneous; the paper lists ReLU, PReLU
and linear, and says ReLU6 "require[s] special treatment". It includes activation-driven one-step
equalisation, so its objective is close to ours (filling activation ranges), but it covers homogeneous
activations and positive factors only.

**Qualcomm white paper, Nagel et al. 2021, https://arxiv.org/abs/2106.08295.** Section 3 restates CLE for
"homogeneous function[s] of degree one" and piecewise-linear functions (for example ReLU6). For sigmoid and
Swish it only says they "require more dedicated support", meaning an extra quantizer around the
non-linearity. It gives no equalisation for them. Its PTQ tables use EfficientNet-*lite*, the ReLU6 variant.

**AIMET, Siddegowda et al. 2022, https://arxiv.org/abs/2201.08442.** CLE "exploit[s] the scale equivariance
property of certain activation functions (e.g. ReLU, PReLU)". The CLE API replaces ReLU6 with ReLU, and the
paper warns: if float accuracy drops after that replacement, "do not apply CLE". It offers no path for SiLU
or Hardswish.

**Qualcomm patent US12242956B2 (priority 2019-03-22), https://patents.google.com/patent/US12242956B2/en.**
The patented form of CLE. It says the equality may not hold "if f(.) is a non-linear function such as a
sigmoid activation or tanh", and it specifies S_ii as a "nonnegative scaling factor". This is explicit
prior-art language that negative scales and sigmoid-type activations are outside that method.

**Perceive/Amazon patent US11847568B2 (priority 2019-07-30), https://patents.google.com/patent/US11847568B2/en.**
It says of sigmoid, tanh, ELU and Swish that a rescaled input "cannot necessarily be compensated for by
modifying the scale/shift of fanout components". It falls back to affine approximation or LUT
reprogramming. Its scales are positive.

**HPTQ, Habi et al. 2021 (Sony), https://arxiv.org/abs/2109.09113.** HPTQ applies activation equalisation
(scales from post-activation statistics) only where positive scale equivariance holds, that is, for
piecewise-linear activations. For Swish, PReLU and HSwish it instead uses Shift Negative Correction (SNC):
add a constant shift so the output is non-negative, then fold the correction into the next layer's bias.
On EfficientNet-B0 (Keras, ImageNet) it reports 77.2 FP, 74.3 with activations only quantized, and 74.2 with
A+W quantized. It attributes the loss to "the fact that activation equalization is not applied for these
activations" (Swish). With SNC, MobileNetV1-Swish improves from 60.98 to 71.15. Note that HPTQ uses
*symmetric, power-of-two* thresholds, not asymmetric uint8.

**Sony MCT (source, github.com/SonySemiconductorSolutions/mct-model-optimization).**
`substitutions/scale_equalization.py` matches only `Conv -> ReLU -> Conv`, using closed-form ReLU moment
corrections; the scale is `min(1/std, 1)`, which is positive. `shift_negative_activation.py` matches PReLU,
ELU, Hardswish, SiLU and GELU, and adds a scalar *shift*, not a per-channel scale. MCT has no scaling across
SiLU or Hardswish and no signed scale.

**NNCF ChannelAlignment (OpenVINO, source).** It works on Conv -> Conv pairs *with no activation between
them*, adjusting weights and biases so that activation quantile medians centre on zero. It does not apply to
the gated case.

## 2. Scale migration in LLM quantization

**SmoothQuant, Xiao et al. 2022, https://arxiv.org/abs/2211.10438.** Per-channel `X diag(s)^-1 . diag(s) W`
is folded into the preceding LayerNorm or linear layer. It is only applied where the preceding op is linear,
and never across the SiLU/GELU non-linearity.

**Outlier Suppression+, Wei et al. 2023, https://arxiv.org/abs/2304.09145.** OS+ observes that channels sit
asymmetrically: one OPT-66B channel spans (-97, -58) while another spans (5.7, 43). Its fix is
*channel-wise shifting* (subtract the channel midpoint) followed by positive scaling, both folded into the
LayerNorm and the next linear layer. It solves the same "a channel on the wrong side of zero" problem that
our negative s addresses, but with an additive shift, not a sign flip, and only at linear/LN sites.

**AWQ, Lin et al. 2023, https://arxiv.org/abs/2306.00978.** Positive activation-aware per-channel scales on
weight-only quantization. In the FFN, the scale on the down-projection input is absorbed into the *up*
(linear) branch of the GLU, not the gate.

**OmniQuant, Shao et al. 2023, https://arxiv.org/abs/2308.13137.** A learnable equivalent transformation
(scale plus shift, following OS+) applied to [ln1, qkv], [v, out], [Q, K] and [ln2, fc1]. It explicitly
*excludes* the second FFN linear "due to the high sparsity of features after the non-linear layer". It does
not transform across the activation.

**SmoothQuant+, Pan et al. 2023, https://arxiv.org/abs/2312.03788.** SmoothQuant-style positive smoothing
for W4A16. It does not cross non-linearities.

**QuaRot, Ashkboos et al. 2024, https://arxiv.org/abs/2404.00456, and SpinQuant, Liu et al. 2024,
https://arxiv.org/abs/2405.16406.** Orthogonal rotations (randomised Hadamard, i.e. random ±1 diagonal
times Hadamard, or learned Cayley rotations) applied to the residual stream, plus an *online* Hadamard
before down_proj. The ±1 signs are random and exist only to make the rotation incoherent. They are not
chosen per channel to fit a range, and they never pass through the SiLU gate.

**DuQuant, Lin et al. 2024, https://arxiv.org/abs/2406.01721, and FlatQuant, Sun et al. 2024,
https://arxiv.org/abs/2410.09426.** Rotation or permutation, and learned affine transforms, respectively.
For down_proj, FlatQuant merges its scaling vector "to W_u", the up (linear) branch. Neither method
transforms the gate input.

**Smooth-SwiGLU, Fishman et al. 2024 (ICLR 2025), https://arxiv.org/abs/2409.12517.** Per-channel s_i is
applied to the *linear* branch (w1) of SwiGLU = (x^T w1) Swish(x^T w2) and inverted in w3. It is used for
FP8 training stability. It does not touch the Swish input.

**I-LLM, Hu et al. 2024 (Houmo AI), https://arxiv.org/abs/2405.17849. This is the closest prior art for
part (a).** Its "NonLinear Act-Smooth" (a case of FSBR) decomposes SiLU(x1) = x1 . sigma(x1). It smooths
x1' = x1 * s by setting W' = W * s, and defines **sigma'(x1') = sigma(x1' / s)**: an explicit inverse scale
on the sigmoid input, so the gate is unchanged. The compensating 1/s is absorbed into the *other* GLU
branch (V' = V / s), not into a following layer. The s are learned during block reconstruction, and no sign
or negativity is discussed (a search of the text for "negative" found no hits). It targets LLM SwiGLU for
integer-only inference, not CNNs.

**MambaQuant, Xu et al. 2025 (ICLR 2025), https://arxiv.org/abs/2501.13484.** Defines "Smooth SiLU"
S-SiLU(x, s) = x . sigma(s . x), citing Hu et al. 2024. The gate projection gets W'_g = W_g / s and the
output projection gets W'_o = s . W_o, which is exactly the "scale the producer, inverse-scale the gate
input, and inverse-scale the consumer" pattern. It is applied in Mamba blocks, where the consumer is a
dense linear layer after an element-wise product with y_ssm, and s is positive (a smoothing factor).

**MobileQuant, Tan et al. 2024, https://arxiv.org/abs/2408.13933.** It states that equivalent weight
transformations "cannot propagate beyond non-linear operators, e.g. ... SiLU/GELU", and restricts itself to
linear-linear pairs. This supports the view that the trick is not widely known, but it is not the most
recent word on the subject.

## 3. Negative / signed scales

**Signed Symmetric Quantization for Few-Bit Integers, Colbert et al. (AMD), June 2026,
https://arxiv.org/abs/2607.08779.** Treats the sign of the quantizer scale as a free parameter, placing the
dominant outlier on the extra negative code of a signed grid. It claims "no prior work treats the sign of
the scale factor as an explicit degree of freedom". It applies to *weight* groups only and is local, with
no cross-layer transformation.

**Optimal PTQ Scales and Where to Find Them, Amboage et al. (AMD), June 2026,
https://arxiv.org/abs/2606.10890.** Notes that "for asymmetric quantization grids ... the optimal scale may
be negative". It handles weight-only scale search and has no cross-layer or activation-range component.

**OS+ (above)** tackles the one-sided channel problem with shifts. **QuaRot/SpinQuant/DuQuant (above)**
use ±1 entries for incoherence, drawn at random or learned inside an orthogonal matrix, not per-channel
signs chosen from calibrated ranges. **DFQ, the Qualcomm patent and Meller et al.** require non-negative or
positive factors, and the Qualcomm patent says so explicitly.

## 4. EfficientNet-B0 PTQ reference points

- Wu et al. 2020 (NVIDIA), https://arxiv.org/abs/2004.09602: per-channel weights, per-tensor activations,
  ImageNet, FP32 76.85. Max calibration gives **22.3**; entropy gives 72.06; percentiles give 70.87 (99.9%),
  68.33, 51.88 and 42.49. Per-tensor weights with folded BN give 12.93. QAT is needed to recover full
  accuracy. Our "24% to 76%" baseline matches their max-calibration collapse.
- HPTQ (above): 77.2 FP to 74.2 INT8 (symmetric PoT, with SNC, Keras B0). Without their threshold search,
  activation-only accuracy is 13.6 (NC/MSE). The authors say the missing Swish equalisation is the likely
  remaining cause.
- The TF EfficientNet-Lite blog (2020) reports a 75 to 46 drop and the Swish-to-ReLU6 swap as the fix
  (architectural, not PTQ).

---

## 5. Novelty verdict (conservative)

**(a) Gate-side inverse scale for self-gated activations (x . g(x) with g fed x'/s): PRIOR ART EXISTS.**
- I-LLM (Hu et al. 2024, arXiv:2405.17849) defines sigma'(x1') = sigma(x1'/s) to smooth across SiLU in
  SwiGLU.
- MambaQuant (Xu et al. 2025, arXiv:2501.13484) defines S-SiLU(x, s) = x . sigma(s x), with the producer
  scaled by 1/s and the consumer linear by s. This is structurally identical to our producer, gate and
  consumer rewrite.

Our variant differs in several ways: it compensates in a following *depthwise conv* rather than the other
GLU branch or an out-projection; it covers HardSigmoid/Hardswish (by splitting the fused op); it chooses s
in closed form from calibrated ranges rather than by learning; and it targets CNNs. These are applications
or extensions, not a new mechanism. We should not claim the gate-side 1/s as novel. We should cite I-LLM
and MambaQuant.

**(b) Negative per-channel scales to use an asymmetric activation range: NOT FOUND as such; closely related
work exists.**
- Not found after searching the DFQ, Meller, AIMET, white paper and Qualcomm patent (positive or
  non-negative only, and the patent says so explicitly); HPTQ and MCT (SNC is a scalar shift); OS+ (shift);
  SmoothQuant, AWQ, OmniQuant and FlatQuant (positive scales); QuaRot, SpinQuant and DuQuant (random or
  learned ±1 inside orthogonal rotations, for incoherence, not range fitting); I-LLM and MambaQuant (no sign
  discussion); plus web searches for sign flips, mirrored channels and negative equalisation scales.
- Closely related: Colbert et al. 2026 (arXiv:2607.08779) and Amboage et al. 2026 (arXiv:2606.10890) use a
  *negative quantizer scale* for weights on asymmetric or signed grids. That is a per-group quantizer
  parameter, not a function-preserving cross-layer sign flip of activation channels. OS+ addresses the same
  one-sided-channel symptom by shifting.
- Caveat: an orthogonal-transform method that learns a general orthogonal or affine matrix (SpinQuant,
  FlatQuant) contains per-channel sign flips as a special case, and learned LET scales (OmniQuant) are not
  explicitly sign-constrained. These methods never apply such flips *through a gate*. The careful claim is
  therefore: "we found no prior method that *deliberately chooses* negative per-channel equalisation scales
  to move channels into the populated side of an asymmetric activation range". Note also that the sign flip
  is exact only because of the gate-side 1/s, which is itself prior art (a).

**(c) The combination applied to static per-tensor INT8 CNN PTQ (EfficientNet/MobileNetV3): NOT FOUND
after searching** the CNN literature and toolkits above (DFQ, Meller, AIMET, the white paper, HPTQ, MCT,
NNCF) and web searches for SiLU-, Swish- or Hardswish-aware CLE and for YOLO/EfficientNet SiLU
equalisation. The CNN literature consistently says CLE is limited to piecewise-linear activations, and
HPTQ names the missing Swish equalisation as a cause of EfficientNet's loss. The defensible claim is: "to
our knowledge, the first application of gate-side-compensated equalisation (after I-LLM and MambaQuant) to
CNN PTQ, extended with signed scales and to Hardswish, closing EfficientNet-B0 static INT8 to within ~0.7pp".
This is an engineering and application contribution, not a new equivalence.

---

# Part 2 — Post-training quantization of efficient CNNs


Scope: post-training quantization (PTQ) of EfficientNet-B0/-Lite, MobileNetV2/V3 and MnasNet
at W8A8. Also covered: why these networks quantize badly, how calibration choices matter, and
why per-channel activation scales are not an option in integer kernels. This file was compiled on
2026-09-24. Unless marked otherwise, every number below was read from the paper's own PDF
(extracted with pypdf) or from the vendor source/doc linked in the bibliography. "(mem.)" marks
the few items I did not re-verify from the primary source.

Anneal's reference numbers (README, "When INT8 breaks on every CPU"), from Imagenette val
(3,925 images, 10 ImageNet classes, scored 1000-way), onnxruntime QDQ, U8S8, per-channel
weights and per-tensor activations:

| model | FP32 | MinMax (emul / x86 no-VNNI) | best Anneal recipe (emul / laptop) | drop, best |
|---|---|---|---|---|
| EfficientNet-B0 | 76.6 | 26.5 / 24.2 | 75.6 / 75.9 (reduce_range 75.8 emul) | **−0.8 to −1.0** |
| MobileNetV3-L | 70.9 | 59.7 / 45.9 | 69.2 / 67.5 | **−1.7** emul, −3.4 laptop |

Anneal zoo, per the coordinator: with MinMax, EfficientNet-B1 loses −76pp and EfficientNetV2-S
loses −25pp **even emulated**. ResNet-50, MobileNetV2, RegNet, ShuffleNet and MnasNet lose ~0pp
emulated, but 8–18pp on x86 without VNNI. So the failure that shows up on every CPU is specific
to the EfficientNet family. The x86-only failure is the known U8S8 `VPMADDUBSW` saturation
issue, which onnxruntime's own docs describe.

---

## 1. Published W8A8 results on efficient CNNs (ImageNet-1k val, top-1)

"PC-W" = per-channel weights; "PT-A" = per-tensor activations; sym/asym = activation range.
Drops are FP32 minus INT8 in percentage points, using each paper's own FP32 baseline.

| method (paper) | model | FP32 → W8A8 | drop | PC-W? | PT-A? (range) | data? | training/gradients? |
|---|---|---|---|---|---|---|---|
| Naive per-layer, TF (Krishnamoorthi 2018, Tab.3) | MobileNetV1 / V2 | 70.9→0.1 / 71.9→0.1 | **~−71** | no | yes (asym) | calib | none |
| Per-channel, TF (Krishnamoorthi 2018, Tab.3) | MobileNetV1 / V2 | 70.9→70.8 / 71.9→70.0 (asym PC) | −0.1 / −1.9 | yes | yes (asym) | calib | none |
| TF8 "gemmlowp" (Sheng 2018, Tab.1) | MobileNetV1 | 70.50→1.80 | −68.7 | no | yes | calib | none |
| Basic 8-bit (Finkelstein 2019, Tab.1) | MobileNetV1 / V2 / V2-1.4 | 71.02 / 71.8 / 74.95 | −7.90 / −16.44 / −6.42 | no (sym PT) | yes | calib | none |
| + Bias fine-tune (Finkelstein 2019) | same | — | −1.03 / −1.2 / −0.85 | no | yes | small set | bias-only tuning |
| DFQ = CLE + bias absorb + bias corr. (Nagel 2019, Tab.1/2/5) | MobileNetV2 | 71.72→71.19 (orig. 0.12) | −0.53 | **no** | yes (asym) | **none** | **none** |
| DFQ (Nagel 2019, Tab.5) | MobileNetV1 | 70.8→70.5 | −0.3 | no | yes | none | none |
| Per-channel only (Nagel 2019, Tab.1) | MobileNetV2 | 71.72→70.65 | −1.07 | yes | yes | calib | none |
| ZeroQ (Cai 2020, Tab.Ib) | MobileNetV2 | 73.03→72.91 | −0.12 | yes (mem.) | yes (asym) | **none (synthetic)** | none |
| Max calib. (Wu/NVIDIA 2020, Tab.5) | **EfficientNet-B0** | 76.85→**22.3** | **−54.6** | yes | yes (**sym**) | 1024 imgs | none |
| Entropy, best PTQ (Wu 2020, Tab.5/7) | EfficientNet-B0 | 76.85→72.06 | −4.8 | yes | yes (sym) | 1024 | none |
| 99.99% / 99.999% pct (Wu 2020, Tab.5) | EfficientNet-B0 | →68.33 / 51.88 | −8.5 / −25.0 | yes | yes (sym) | 1024 | none |
| Entropy, 10 layers left in FP (Wu 2020, Tab.6) | EfficientNet-B0 | →76.35 | −0.5 | yes | partial | 1024 | none |
| Best of max/entropy/pct (Wu 2020, Tab.7) | EfficientNet-B3 | 81.61→80.28 | −1.3 | yes | yes (sym) | 1024 | none |
| Best (Wu 2020, Tab.7) | MobileNetV1 / V2 | 71.88→70.39 / 71.14 | −1.5 / −0.7 | yes | yes (sym) | 1024 | none |
| QAT (Wu 2020, Tab.7) | EfficientNet-B0 | →76.95 | +0.1 | yes | yes | full | **QAT** |
| Standard pipeline: CLE+MSE+AdaRound/BC (Nagel 2021, Tab.6) | MobileNetV2 | 71.72→70.99 (PT) / 71.16 (PC) | −0.73 / −0.56 | both | yes (asym) | 500–1000 | AdaRound optional |
| same (Nagel 2021, Tab.6) | **EfficientNet-lite** | 75.42→75.25 (PT) / 75.39 (PC) | −0.17 / −0.03 | both | yes (asym) | 500–1000 | AdaRound optional |
| HPTQ (Habi 2021, Tab.6) | **EfficientNet-B0 (swish)** | 77.2→74.216 | **−3.0** | yes | yes (sym, PoT, +SNC) | 500 | none |
| HPTQ, activations only, no-clip baseline (Tab.8) | EfficientNet-B0 | 77.2→13.56 | −63.6 | — | yes | 500 | none |
| HPTQ (Tab.6) | EfficientNet-B0 **ReLU6** retrained | 77.65→77.09 | −0.56 | yes | yes | 500 | none |
| HPTQ (Tab.2/6) | MobileNetV2 / V1 | 71.81→71.46 / 70.56→70.42 | −0.35 / −0.14 | yes | yes (sym PoT) | 500 | none |
| MQBench, FBGEMM backend, best calib (Li 2021, Tab.10) | MobileNetV2 | 72.6→71.2 | −1.4 | yes | yes (asym) | calib | none |
| MQBench, FBGEMM, MinMax | MobileNetV2 | 72.6→70.9 | −1.7 | yes | yes (asym) | calib | none |
| MQBench, SNPE (per-tensor W), best | MobileNetV2 | 72.6→71.1 | −1.5 | **no** | yes (asym) | calib | none |
| MQBench, FBGEMM, best (MSE/MSE) | **EfficientNet-Lite0** | 75.3→73.7 | −1.6 | yes | yes (asym) | calib | none |
| MQBench, SNPE, KLD / MinMax | EfficientNet-Lite0 | 75.3→48.1 / 71.0 | −27.2 / −4.3 | no | yes (asym) | calib | none |
| TFLite int-only PTQ (Liu, TF blog 2020) | EfficientNet-Lite0 **with swish** | 75→46 | **−29** | (TFLite spec: yes) | yes (asym) | calib | none |
| same, after swish→ReLU6 | EfficientNet-Lite0 | 75.1→74.4 | −0.7 | yes | yes | calib | none |
| MobileNetV3 paper, "quantized" (Howard 2019, Tab.4) | MobileNetV3-L / MobileNetV2 | 75.2→73.8 / 72.0→70.9 | −1.4 / −1.1 | TFLite | TFLite | ? | not stated (PTQ vs QAT unclear) |
| torchvision reference (QAT, not PTQ) | MobileNetV3-L / MobileNetV2 | 74.04→73.00 / 71.88→71.66 | −1.0 / −0.2 | yes | yes | full | **QAT, ~10 epochs** |

### Reconstruction-based PTQ (AdaRound, BRECQ, QDrop, PD-Quant, Genie): no W8A8 numbers on these nets

These papers treat W8A8 as already solved. None of them reports a W8A8 row for MobileNetV2,
MnasNet or EfficientNet. AdaRound's Table 7 cites DFQ's 71.2 for MobileNetV2 8/8 as prior art.
BRECQ writes that "4-bit and 8-bit quantization nearly do not drop the final accuracy" in its
mixed-precision section. QDrop's only W8A8 row is Min-Max on ResNet-18 (70.94 vs 71.06). They
report low-bit results instead. All numbers are ImageNet, with first and last layers kept at
8-bit:

| method | MobileNetV2 (FP 72.49) W4A4 | MnasNet-2.0 (FP 76.68) W4A4 | granularity | data | optimisation |
|---|---|---|---|---|---|
| AdaRound (Nagel 2020; numbers from QDrop Tab.3) | 61.52 (64.33 w/ BRECQ setting) | 68.86 | PC-W, PT-A (LSQ-style) | 1024 imgs | per-layer rounding opt., 10k–20k iters |
| AdaRound (own paper, W4A8 + CLE) | 69.25 (FP 71.72) | — | per-tensor W | 1024 | same |
| BRECQ (Li 2021) | 66.57 | 73.56 | PC-W, PT-A | 1024 | block reconstruction, 20k iters/block |
| QDrop (Wei 2022) | 68.84 | 73.71 | PC-W, PT-A | 1024 | BRECQ + random act-quant drop |
| PD-Quant (Liu 2023; FP 72.62 / 76.52) | 68.19 | 73.26 | PC-W, PT-A | 1024 | + prediction-difference loss, 20k iters |
| Genie-M, real data (Jeon 2023) | 69.23 | — (MnasNet-1.0: 68.29, FP 73.52) | PC-W asym, PT-A sym | 1024 | QDrop-style |
| Genie, **data-free** (Jeon 2023) | 68.38–68.70 | MnasNet-1.0: 66.94–67.45 | same | synthetic 1K | generator + reconstruction |

None of these evaluates EfficientNet-B0 with SiLU. The swish/SE variant is the one Anneal
targets, and the literature routinely replaces it with EfficientNet-Lite (MQBench does this
explicitly, "replaces swish activation to ReLU6 for better integer numeric support"), with
ReLU-family CNNs, or with RegNet.

### Data and training requirements (relevant because Anneal is closed-form, no gradient steps)

- **No data, no training:** DFQ (CLE, bias absorption, analytic bias correction), ZeroQ (uses
  synthetic data but no fine-tuning), and Genie (synthetic data, but it does optimise).
- **Calibration data, no gradients:** min-max, percentile, entropy/KL, MSE range setting
  (Krishnamoorthi, Wu, MQBench, HPTQ). HPTQ is the closest in spirit to Anneal: rule-based,
  about 500 images, and per-channel scaling across activations. Its "Max Channel Equalization",
  however, requires the activation to be ReLU, ReLU8 or PReLU (piece-wise linear), so it
  **skips swish**. HPTQ itself names that as the likely cause of EfficientNet's −3pp.
- **Calibration data plus gradient optimisation:** AdaRound, BRECQ, QDrop, PD-Quant, Genie-M
  (roughly 10k–20k iterations per layer or block; BRECQ quotes under 1 GPU-hour for MobileNetV2).
- **Full training:** QAT (Wu 2020 EfficientNet-B0 +0.1; torchvision MobileNetV2/V3; Sheng 2018
  architecture change).

---

## 2. Why these networks quantize badly

1. **Depthwise convolutions have few weights per output channel, and BN folding scatters their
   ranges.** Krishnamoorthi (2018) saw per-layer weights collapse MobileNetV1/V2 to 0.1%, which
   he attributes to "batch normalization, which causes extreme variation in dynamic range across
   convolution kernels". Nagel (2019, Fig. 2) shows per-output-channel ranges in MobileNetV2's
   first depthwise layer spanning more than 10x. Wu (2020, Tab.3) gives the sharpest single
   number: **EfficientNet-B0 weights-only, per-tensor after BN folding = 12.93%** (per-channel
   76.72%). HPTQ (Tab.11) has per-tensor weights at 2.5% for EfficientNet-B0 and 0.4% for
   MobileNetV2. Per-channel weights solve this part. That matches Anneal's finding that weight
   SQNR is at least 39 dB everywhere and that weights are not the cause.
2. **Activation-side channel mismatch.** Anneal's cause lives here, and the literature says less
   about it. Yun & Wong (2021) analyse MobileNet (ImageNet MobileNetV1 at UINT8: 71.04→3.00) and
   attribute the damage to "a mismatch between each individual channel's dynamic range and the
   entire tensor's range" at depthwise layers, with error accumulating through the network.
   Nagel 2019/2021 note that BN folding and high biases leave activation ranges imbalanced. CLE
   and bias absorption fix that only through ReLU-like (positively homogeneous) activations.
3. **Unbounded, signed activations (swish/SiLU, h-swish).** Krishnamoorthi (2018) says activations
   were easy *because* of ReLU6 ("restricts the activations to be in a fixed range (0,6)") and BN
   without scaling. EfficientNet removes both properties. The TFLite team measured an output
   tensor range of −168 to 204 in EfficientNet-Lite0 with swish, and PTQ dropped it from 75% to
   46%; switching to ReLU6 restored 74.4%. Anneal's stem channel spans 169, which is the same
   phenomenon. HPTQ (Tab.9): a swish MobileNetV1 drops to 60.98 under signed symmetric
   quantization, and to 71.15 with Shift-Negative-Correction (FP 73.52).
4. **Dead / constant channels.** Sheng et al. (2018) found channels in MobileNetV1 whose outputs
   are all zero, so their BN variance is 0. That produces huge folded weights. Fixing the
   zero-variance issue alone moved TF8 accuracy from 1.80% to 45.73%. Removing BN+ReLU6 from the
   depthwise layers and switching to ReLU (with retraining) reached 68.03% (FP 70.77).
   Finkelstein et al. (2019) drop dead channels as a preprocessing step. Nagel (2019) notes that
   channels with all weights near zero remain after CLE. Anneal's "starved channel" living
   entirely in SiLU's negative lobe (−0.27 to −0.10) is the activation-side cousin: nearly
   constant, and given less than one quantization level.
5. **Biased error.** Quantization error in depthwise layers is not zero-mean. Finkelstein (2019)
   traced MobileNet degradation to a mean activation shift, and bias correction takes MobileNetV2
   from −16.4pp to −1.2pp. Nagel (2019) reports 0.12→52.02% from bias correction alone on
   MobileNetV2.
6. **Few sensitive layers.** In Wu 2020 (Fig.3/Tab.6), EfficientNet-B0's most sensitive layers
   are the block-0/1/2/3/4 depthwise convs and the block-12/14/15 projections. Keeping 10 of 82
   layers in float recovers 72.06→76.35. This matches Anneal's float-stem and depthwise findings.

---

## 3. Calibration

| calibrator | what it does | evidence on efficient nets |
|---|---|---|
| Min-max / max | no clipping | Wu: EfficientNet-B0 **22.3%** (sym). MQBench: fine for ResNets, worst-but-one for Lite0 on SNPE. HPTQ no-clip: EfficientNet-B0 activations-only 13.56% |
| Percentile | clip a tail fraction | Wu: 99.99% best for MobileNetV2/EffNet-B3. EffNet-B0 is **non-monotone**: 99.9→70.87, 99.99→68.33, 99.999→51.88, 99.9999→42.49. MQBench quantile 0.9999 on Lite0: 67.0 (SNPE) / 71.4 (FBGEMM) |
| Entropy / KL (TensorRT; Migacz 2017) | minimise KL between histograms | Wu: best for EffNet-B0 (72.06). MQBench: KLD fails on Lite0/SNPE (48.1). HPTQ: KL poor on VGG/ResNet activations |
| MSE (OMSE, Choukroun 2019; Banner ACIQ analytic) | minimise ‖x−Q(x)‖² | Nagel 2021 Tab.2 (MobileNetV2 A8: MSE 71.35 vs min-max 70.96). MQBench: MSE best for MobileNetV2 and Lite0. HPTQ: MSE best overall, and MSE thresholds take EffNet-B0 activations from 13.7 to 74.1 |
| MSE + cross-entropy on logits | Nagel 2021 | small gains at A8, larger at A4 |

Consensus: "no single calibration is best for all networks" (Wu 2020). MQBench and Nagel 2021
agree. For efficient nets, clipping (MSE, percentile) beats min-max, but the best setting is
model-specific.

**Symmetric vs asymmetric for post-SiLU activations:**
- SiLU's minimum is ≈ −0.278, and h-swish's is −0.375. A *symmetric* signed range therefore
  spends about half its codes on a negative side that barely exists. HPTQ is built around
  symmetric power-of-two thresholds, and its fix, SNC (shift the tensor by |min| and use an
  unsigned quantizer), "effectively doubles the quantization grid resolution". SNC is worth
  +10pp on a swish MobileNetV1.
- Wu/NVIDIA 2020 use symmetric (scale) quantization for activations too ([−127,127]). That is one
  plausible reason their EfficientNet-B0 numbers (22.3 max, 72.06 best) are so much worse than
  their MobileNetV2 numbers, whose ReLU6 outputs are non-negative. The paper does not isolate
  this. **Their max-calibrated 22.3% is within about 2–4pp of Anneal's MinMax 24.2/26.5%**,
  which is independent corroboration of the collapse, although on a different dataset and a
  different EfficientNet port (lukemelas).
- Nagel 2019 (Tab.7) finds asymmetric vs symmetric almost negligible *after DFQ* on ReLU nets
  (MobileNetV2 71.15 vs 71.19). The asymmetry question only matters when the activation is
  signed and lopsided.
- **onnxruntime detail worth documenting in Anneal:** in `calibrate.py`,
  `CalibrationMethod.Percentile` defaults to `symmetric=True` (the histogram is built on |x|,
  99.999th percentile), whereas MinMax and Entropy default to `symmetric=False`. The stock ORT
  percentile path is therefore *symmetric* unless the user overrides it. Anneal's "asymmetric
  percentile" is a deliberate non-default and should be described that way.

---

## 4. Per-channel activation quantization and hardware

- A conv or GEMM computes Σ_c w·x over input channels c. A per-output-channel weight scale
  factors out of that sum. A per-*input*-channel activation scale does not: the accumulator
  would need rescaling for every input channel. Nagel 2021: "Per-channel quantization of
  activations is much harder to implement because we cannot factor the scale factor out of the
  summation". Krishnamoorthi 2018: "We do not consider per-channel quantization for activations
  as this would complicate the inner product computations". Wu 2020: "For activations, only
  per-tensor quantization is practical for performance reasons". SmoothQuant (Xiao 2023, Fig.3)
  says INT8 GEMM kernels can scale only along the outer dimensions (tokens, output channels),
  "but not inner dimension (i.e., in channel dimension Ci)".
- Specifications: ONNX `QLinearConv` fixes `x_scale` as "a scalar, which means a per-tensor/layer
  quantization", while `w_scale` may be 1-D per output channel. TFLite/LiteRT's int8 spec allows
  per-axis weights but per-tensor activations. MQBench Table 2 lists TensorRT, ACL, TVM, SNPE and
  FBGEMM, and all of them use per-tensor activations. TVM and SNPE even use per-tensor weights.
- Depthwise convs are the one place where per-channel activation scales could be folded, because
  input channel c feeds only output channel c. Standard kernels (MLAS, XNNPACK, TensorRT) still
  take a scalar input scale. Anneal's "+42pp from per-channel activation scales" is a diagnostic
  that cannot be deployed. Equalisation is the deployable equivalent, and that is exactly the
  argument CLE, HPTQ and SmoothQuant each make for their own settings.
- ACIQ (Banner 2019) used per-channel activation bit allocation. ZeroQ criticises it as "difficult
  for efficient hardware implementation in practice".

---

## 5. Positioning statement (proposed text)

> On Imagenette (3,925 val images, 10 ImageNet classes scored 1000-way), Anneal's closed-form
> recipe brings static W8A8 EfficientNet-B0 to within about 1pp of FP32 (76.6 → 75.6 emulated;
> 75.9 on non-VNNI x86 with reduce_range), and MobileNetV3-L to within 1.7pp emulated. It uses
> no gradient steps: an exact equalisation through SiLU/Hardswish gates, asymmetric percentile
> calibration, and a float stem. Per-channel weights and per-tensor activations are deployable
> on stock onnxruntime CPU kernels. For context, the published ImageNet-1k PTQ numbers for
> **EfficientNet-B0 with swish** are −54.6pp (max) and −4.8pp (best calibrator) in NVIDIA's
> study (Wu et al. 2020), and −3.0pp with HPTQ (Habi et al. 2021). Recovering to −0.5pp needed
> 10 layers left in float, and to +0.1pp needed QAT. Google avoided the problem by redesigning
> the network (EfficientNet-Lite: swish→ReLU6, no SE), after measuring 75→46% PTQ with swish.
> The reconstruction-based PTQ family (AdaRound, BRECQ, QDrop, PD-Quant, Genie) reports no
> W8A8 EfficientNet-B0 result at all. **These drops are not directly comparable**: a 10-class
> subset is easier than ImageNet-1k, and subset evaluation is known to inflate top-1 and even
> flip the sign of quantization deltas (Shin 2026: a 1,000-image ImageNet subset overstated
> top-1 by ~9.8pp on average and flipped 3 quantization deltas). The models, preprocessing
> and runtime also differ. What we can claim is qualitative: the MinMax collapse we see (24–26%)
> matches the one NVIDIA reported (22.3%), our mechanism is consistent with HPTQ's diagnosis
> (activation equalisation unavailable for swish), and our fix is closed-form. We do not claim
> that Anneal beats any published method.

**Why the zoo result strengthens the story.** EfficientNet-B1 (−76pp) and EfficientNetV2-S
(−25pp) collapse emulated, while ReLU-family, ReLU6 and group-conv nets lose ~0 emulated. That
is the same split the literature shows. ReLU/ReLU6 MobileNetV2, MnasNet, RegNet and ShuffleNet
are "W8A8-solved" with per-channel weights (MQBench: MobileNetV2 −1.4 to −1.7, RegNetX-600MF
~−0.2; torchvision ships ShuffleNet PTQ). The swish/SE EfficientNet family is the outlier
(Wu, HPTQ, TFLite blog). Anneal's MobileNetV2 at ~0pp emulated is *better* than MQBench's
−1.7pp MinMax/FBGEMM. That is probably the easier 10-class task, not the recipe, and it is a
warning against reading Imagenette drops as ImageNet drops. The 8–18pp losses on x86 without
VNNI are a separate, documented onnxruntime U8S8 saturation effect, which reduce_range or U8U8
fixes. They should not be mixed into the comparison with papers that simulate quantization in
float.

**What a fair comparison needs:**
1. **Full ImageNet-1k val (50,000 images), 1000-way,** with torchvision/timm reference weights
   and documented preprocessing (resize 256 / center-crop 224, or timm's per-model config).
   Shin 2026 found that preprocessing alone moved top-1 by ~1pp, which is 9x the quantization
   cost they measured. Report the float stem as partial quantization, as Wu does.
2. **Same pipeline baselines** in the same onnxruntime QDQ graph, same calibration set
   (e.g. 1,024 train images), same per-channel-W and per-tensor-A granularity: (a) ORT
   MinMax / Entropy / Percentile (sym and asym) / Distribution; (b) CLE+bias correction (DFQ,
   AIMET), which should fail through SiLU and so makes a useful control; (c) HPTQ-style SNC and
   MSE thresholds (Sony MCT is open source); (d) AdaRound and BRECQ/QDrop at W8A8, exported to
   QDQ (MQBench and AIMET both export ONNX). The AdaRound/BRECQ rows answer whether gradient
   reconstruction of weights fixes an *activation*-scale problem. My expectation is that it
   mostly does not, because Anneal's weights are already at ≥39 dB SQNR, but it has to be
   measured. (e) Anneal equalisation **combined** with AdaRound/BRECQ, since the two are
   orthogonal.
3. **Models:** EfficientNet-B0 (swish+SE) and EfficientNet-Lite0 (the literature's stand-in),
   MobileNetV2, MobileNetV3-L and MnasNet, so that Nagel 2021 (Lite 75.42→75.25/75.39), MQBench
   (Lite0 75.3→73.7) and Wu (B0 76.85→72.06) can be reproduced as sanity checks before claiming
   anything.
4. **Statistics:** Imagenette's 3,925 images give a binomial SE of ≈0.7pp at 75% top-1, so a
   1pp drop is at the edge of resolution. Use paired tests (McNemar on the same images) or
   several calibration seeds, as MQBench, Nagel and PD-Quant do (3–10 runs). ImageNet's 50k
   images give an SE of ≈0.19pp.
5. **Report emulated (float QDQ) and real-kernel numbers separately.** Papers report simulated
   quantization. Anneal's laptop numbers include saturation that the papers never see.

---

## Annotated bibliography

- **Krishnamoorthi 2018**, "Quantizing deep convolutional networks for efficient inference: A
  whitepaper", [arXiv:1806.08342](https://arxiv.org/abs/1806.08342). Google/TF. Per-layer PTQ
  takes MobileNetV1/V2 to 0.1% and per-channel weights fix it. Activations were easy thanks to
  ReLU6/BN, and per-channel activations are ruled out for kernel reasons.
- **Sheng et al. 2018**, "A Quantization-Friendly Separable Convolution for MobileNets",
  [arXiv:1803.08607](https://arxiv.org/abs/1803.08607). Qualcomm. TF8 MobileNetV1 70.5→1.8%.
  Identifies zero-variance (dead) channels and BN+ReLU6 in depthwise layers as the root causes.
  Its fix changes the architecture and retrains (68.03%).
- **Nagel et al. 2019**, "Data-Free Quantization Through Weight Equalization and Bias
  Correction" (DFQ), [arXiv:1906.04721](https://arxiv.org/abs/1906.04721). CLE requires
  positive homogeneity (ReLU/PReLU; ReLU6 is swapped for ReLU). MobileNetV2 per-tensor
  0.12→71.19%, data-free. This is the primary equalisation reference for Anneal.
- **Finkelstein, Almog, Grobman 2019**, "Fighting Quantization Bias With Bias",
  [arXiv:1906.03193](https://arxiv.org/abs/1906.03193). Mean-shift error in MobileNets. Bias
  fine-tuning gets MobileNetV2 from −16.4 to −1.2pp. Dead channels are dropped first.
- **Nagel et al. 2021**, "A White Paper on Neural Network Quantization",
  [arXiv:2106.08295](https://arxiv.org/abs/2106.08295). Standard PTQ pipeline (CLE → MSE ranges →
  AdaRound/BC). Table 6: MobileNetV2 −0.73/−0.56 and EfficientNet-lite −0.17/−0.03 at W8A8.
  Explains why per-channel activations are hard in hardware.
- **Wu, Judd, Zhang, Isaev, Micikevicius 2020**, "Integer Quantization for Deep Learning
  Inference: Principles and Empirical Evaluation", [arXiv:2004.09602](https://arxiv.org/abs/2004.09602).
  NVIDIA. Symmetric per-channel W and per-tensor A. **EfficientNet-B0: max 22.3, entropy 72.06,
  partial 76.35, QAT 76.95.** Per-tensor weights after BN folding: 12.93. The closest published
  analogue of Anneal's collapse.
- **Nagel et al. 2020**, "Up or Down? Adaptive Rounding for Post-Training Quantization"
  (AdaRound), [arXiv:2004.10568](https://arxiv.org/abs/2004.10568). Only W4 results (MobileNetV2
  W4A8 69.25 with CLE). Gradient-based rounding with about 1024 images.
- **Li et al. 2021**, "BRECQ: Pushing the Limit of Post-Training Quantization by Block
  Reconstruction", [arXiv:2102.05426](https://arxiv.org/abs/2102.05426). W4A4 MobileNetV2 66.57
  and MnasNet-2.0 73.56. No W8A8 for these nets.
- **Wei et al. 2022**, "QDrop: Randomly Dropping Quantization for Extremely Low-bit PTQ",
  [arXiv:2203.05740](https://arxiv.org/abs/2203.05740). W4A4 MobileNetV2 68.84 and MnasNet
  73.71. Its only W8A8 row is ResNet-18.
- **Liu et al. 2023**, "PD-Quant: Post-Training Quantization based on Prediction Difference
  Metric", [arXiv:2212.07048](https://arxiv.org/abs/2212.07048). W4A4 MobileNetV2 68.19 and
  MnasNet 73.26.
- **Jeon et al. 2023**, "Genie: Show Me the Data for Quantization",
  [arXiv:2212.04780](https://arxiv.org/abs/2212.04780). Zero-shot (synthetic data) plus
  QDrop-style distillation. W4A4 MobileNetV2 68.4–68.7 data-free. PC asym W, PT sym A.
- **Cai et al. 2020**, "ZeroQ: A Novel Zero Shot Quantization Framework",
  [arXiv:2001.00281](https://arxiv.org/abs/2001.00281). MobileNetV2 W8A8 73.03→72.91 with no
  data. Criticises per-channel activations (ACIQ) as not implementable efficiently.
- **Li et al. 2021**, "MQBench: Towards Reproducible and Deployable Model Quantization
  Benchmark", [arXiv:2111.03759](https://arxiv.org/abs/2111.03759). Hardware-faithful PTQ table
  (App. C, Tab.10): MobileNetV2 −1.4 and EfficientNet-Lite0 −1.6 at best on FBGEMM; KLD on SNPE
  collapses Lite0 to 48.1. Uses Lite because swish/SE are not integer-friendly.
- **Habi et al. 2021**, "HPTQ: Hardware-Friendly Post Training Quantization",
  [arXiv:2109.09113](https://arxiv.org/abs/2109.09113). Sony. Symmetric power-of-two thresholds
  plus SNC, MSE thresholds, and max channel equalisation for ReLU/ReLU8/PReLU only.
  EfficientNet-B0 77.2→74.2, versus ReLU6 variant −0.56. Names missing swish equalisation as the
  likely cause.
- **Liu (Google) 2020**, "Higher accuracy on vision models with EfficientNet-Lite", TensorFlow
  Blog,
  [link](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html).
  Swish EfficientNet-Lite0 PTQ 75%→46%, with an output range of −168 to 204. ReLU6 gives 74.4%
  (FP 75.1). SE was removed for accelerator support, and swish→ReLU6 "significantly improved
  the quality of post-training quantization".
- **Howard et al. 2019**, "Searching for MobileNetV3",
  [arXiv:1905.02244](https://arxiv.org/abs/1905.02244). Calls h-swish "more quantization-friendly".
  Quantized MobileNetV3-L 73.8 vs 75.2 float (the procedure is not specified).
- **Yun & Wong 2021**, "Do All MobileNets Quantize Poorly?…", CVPRW,
  [arXiv:2104.11849](https://arxiv.org/abs/2104.11849). A channel-vs-tensor range mismatch at
  depthwise layers drives error accumulation. ImageNet MobileNetV1 UINT8 3.0%.
- **Dinh et al. 2020**, "Subtensor Quantization for Mobilenets",
  [arXiv:2011.08009](https://arxiv.org/abs/2011.08009). Short paper. MobileNetV2 8-bit PTQ within
  0.7% without per-channel or QAT. Treat as weak evidence.
- **Xiao et al. 2023**, "SmoothQuant", [arXiv:2211.10438](https://arxiv.org/abs/2211.10438).
  Mathematically equivalent migration of per-channel scale from activations to weights. States
  that INT8 GEMMs cannot scale the inner (input-channel) dimension. This is the LLM analogue of
  Anneal's equalisation.
- **Choukroun et al. 2019**, "Low-bit Quantization of Neural Networks for Efficient Inference"
  (OMSE), [arXiv:1902.06822](https://arxiv.org/abs/1902.06822). MSE-optimal ranges.
- **Banner et al. 2019**, "Post-training 4-bit quantization…" (ACIQ),
  [arXiv:1810.05723](https://arxiv.org/abs/1810.05723) (mem.). Analytic clipping, plus
  per-channel bit allocation that includes activations.
- **Migacz 2017**, "8-bit Inference with TensorRT", GTC talk (mem.). The origin of KL/entropy
  calibration.
- **McKinstry et al. 2018**, "Discovering Low-Precision Networks Close to Full-Precision
  Networks", [arXiv:1809.04191](https://arxiv.org/abs/1809.04191) (mem.). Percentile calibration,
  as cited by Wu.
- **Shin 2026**, "Is INT8 Portable? A Cross-Platform Measurement Study of Quantized Inference on
  Embedded and Automotive Accelerators", [arXiv:2609.16085](https://arxiv.org/abs/2609.16085).
  Preprint. The sign of the INT8 speedup depends on dot-product ISA support (VNNI/SDOT). A
  1,000-image subset inflates top-1 by ~9.8pp. Per-layer SQNR has no rank correlation with top-1
  delta (ρ=−0.03), which matters because Anneal diagnoses with SQNR. Per-tensor MinMax
  activation scales drive the DETR collapse in ORT.
- **ONNX operator spec**, `QLinearConv`:
  [Operators.md](https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv). Scalar
  `x_scale`, per-output-channel `w_scale`.
- **onnxruntime quantization docs**:
  [quantization.md](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
  U8S8 `VPMADDUBSW` saturation on AVX2/AVX512 without VNNI; reduce_range and U8U8 as fixes.
  Also **`calibrate.py`**:
  [source](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py).
  Percentile defaults to symmetric=True at 99.999; MinMax and Entropy default to asymmetric.
- **LiteRT/TFLite 8-bit quantization spec**:
  [link](https://ai.google.dev/edge/litert/models/quantization_spec). Per-axis weights,
  per-tensor activations.
- **torchvision quantized models**:
  [references/classification](https://github.com/pytorch/vision/tree/main/references/classification#quantized).
  MobileNetV2/V3 shipped via QAT (71.658 / 73.004). PTQ is used only for ResNet, Inception,
  GoogLeNet and ShuffleNet.

---

# Part 3 — Hardware-level INT8 and hardware-aware tools


Scope: Anneal's hardware claims (u8s8 pair saturation on non-VNNI x86, the per-layer saturation
analyser and guard, and recipe speed differing by architecture), checked against vendor docs,
runtime source code and research. Compiled 2026-09-24. The source-code claims are pinned to
onnxruntime **v1.30.0**, the version in Anneal's venv, and were checked again on `main`.

Legend: **[doc]** vendor documentation · **[src]** source code · **[paper]** peer-reviewed or arXiv · **[issue]** GitHub issue or PR.

---

## TL;DR

1. **The mechanism is well documented.** oneDNN, onnxruntime, OpenVINO/NNCF, PyTorch and a 2018
   Intel white paper all describe VPMADDUBSW's saturating int16 pair sum, the 7-bit fix and the
   fact that VNNI removes the problem. Anneal should not claim the mechanism as new.
2. **The magnitude is not documented.** No vendor gives an accuracy number. oneDNN says its 0.5x
   weight scaling "might insignificantly affect" accuracy. Intel's 2018 paper says "statistical
   performance does not suffer" with the mitigation. ORT says only "large accuracy drop".
   OpenVINO treats a difference above 1% as the sign of saturation. PyTorch's warning and ORT PR
   #24220 give no number either. I found **no** published per-model figure like Anneal's 8–18pp
   over 9 CNNs, or its "same QDQ graph in float loses about 0pp" control.
3. **No per-layer predictive tool was found.** OpenVINO's documentation says the opposite: *"it is
   impossible to predict if the issue occurs in a given setup"*, and *"the only way to detect the
   saturation issue is to run inference on a CPU that allows it and then on one that does not."*
   Existing tools are either **static heuristics** (NNCF/POT `overflow_fix=FIRST_LAYER`, the
   default, and ORT/PyTorch `reduce_range` everywhere) or a **runtime debug check**: ORT's
   experimental `onnxruntime_ENABLE_CONVSYMKERNELAVX2_SAT_CHECKER`, a build flag that is OFF by
   default. It prints one warning per session, with no per-layer attribution, and covers only the
   ConvSym kernel. Anneal's analyser (per-layer emulation on real weights and activations, from
   any host, followed by FP32 fallback of only the flagged layers) looks new.
4. **S8S8 is slow on x86 because of kernel dispatch.** ORT's MLAS only has a fast S8S8 GEMM on x86
   with AVX-VNNI-INT8. Otherwise S8S8 uses `MlasGemmQuantDispatchDefault` (portable C++), and the
   x86 ConvSym path is U8S8-only. On ARM, S8S8 has first-class SDOT, SMMLA (i8mm) and SVE-i8mm
   kernels, and QDQ int8 is kept native (`QDQIsInt8Allowed()` is true only on ARM).

---

## 1. What vendors document about u8×s8 saturation

### Intel white paper (2018): the origin of the "reduce by 1 bit" advice
- Rodriguez et al., *Lower Numerical Precision Deep Learning Inference and Training*, Intel, Jan 2018.
  [PDF](https://www.intel.com/content/dam/develop/external/us/en/documents/lower-numerical-precision-deep-learning-jan2018-754765.pdf)
- Pre-VNNI int8 uses 3 instructions (VPMADDUBSW u8×s8→s16, VPMADDWD, VPADDD). Figure 1 notes the
  cost is "3x more instructions", or **only 33% more compute than fp32**. This is the theoretical
  basis for Anneal's observation that INT8 barely beats FP32 on non-VNNI x86. VNNI fuses the three
  into VPDPBUSD, giving 4x.
- On overflow, the paper says the problem arises "when both u8 and s8 values are near their maximum
  values", and that it is "mitigated by reducing the precision of the inputs by 1-bit". Footnote 2
  argues that, in practice, u8 values sit near their *minimum* when a ReLU precedes the layer.
  **This footnote explains why the problem is under-estimated.** With asymmetric u8 activations,
  real zero maps to the zero point. For non-ReLU inputs (the normalised image fed to the stem,
  hardswish or SiLU outputs, linear bottlenecks) the zero point is near 128, so typical u8 codes
  are large whatever the real magnitude. Anneal's stem and expand-conv findings fit this.
- The paper claims "Preliminary results show that statistical performance does not suffer" with
  these mitigations. No numbers are given for the unmitigated case.

### oneDNN: *Nuances of int8 Computations*
- [uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html](https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html) · [source md](https://github.com/uxlfoundation/oneDNN/blob/main/doc/advanced/int8_computations.md)
- Describes VPMADDUBSW, VPMADDWD, VPADDD on AVX2/AVX-512, with "potential saturation", and the
  worked example (255,255)·(127,127) = 64,770, which saturates to 32,767.
- The **user** is told to choose quantisation parameters so that saturation cannot occur, e.g. u7
  activations or s7 weights.
- **Compensation.** For s8 activations, the reorder adds 128 at run time so that the input becomes
  u8, then subtracts 128·ΣW (the "compensation"). For **s8/s8 convolution** on AVX2/AVX-512, the
  weight reorder also **scales weights by 0.5** and rescales the result, which "might
  insignificantly affect the inference accuracy". s8/s8 GEMM applies no automatic protection. VNNI
  (VPDPBUSD) is said to avoid intermediate saturation.
- It does **not** quantify accuracy loss and offers **no detection tool**.

### FBGEMM and PyTorch: why `reduce_range` exists
- FBGEMM paper: Khudia et al. 2021, [arXiv:2101.05615](https://arxiv.org/abs/2101.05615).
  - The int16-accumulation path (vpmaddubsw plus vpaddsw) "usually leads to frequent
    overflow/saturation". They avoid it with **outlier-aware quantisation**, splitting B into a
    dense small-magnitude part plus sparse outliers (Park et al.).
  - On the int32 path (vpmaddubsw, vpmaddwd, vpaddd), "the theoretical compute peak for INT8 is not
    better than FP32" on AVX2 (Broadwell). This is a second citation for "INT8 ≈ FP32 on non-VNNI".
  - FBGEMM requires A to be u8 and B to be s8 *because* of vpmaddubsw ([issue #199](https://github.com/pytorch/FBGEMM/issues/199)).
- PyTorch `torch/ao/quantization/qconfig.py` [src]
  ([link](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/qconfig.py)):
  - the `fbgemm` and `x86` default qconfigs use `HistogramObserver(reduce_range=True)` on
    **activations** (7-bit u8, 0..127), not weights.
  - the `onednn` qconfig warns *"Default qconfig of oneDNN backend with reduce_range of false may
    have accuracy issues on CPU without Vector Neural Network Instruction support"*, gated on
    `torch.cpu._is_vnni_supported()`. It is a platform-level warning, not a per-layer one.
  - A PyTorch maintainer confirms the rationale on the
    [forum](https://discuss.pytorch.org/t/understanding-differences-in-the-default-qconfig-for-fbgemm-and-qnnpack/175952).
- Note the design split. PyTorch/FBGEMM make **activations** 7-bit. ORT and NNCF make **weights**
  7-bit (ORT `reduce_range`) or half-range (NNCF). Both keep 2·|a|max·|w|max ≤ 32,767.

### onnxruntime: documentation
- [Quantize ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) [doc]
  - On x86-64 with AVX2/AVX-512, U8S8 uses VPMADDUBSW, which "might suffer from saturation
    issues". Use `reduce_range` (7-bit weights) or U8U8. There is "no such issue" on x64 with
    VNNI or on Arm.
  - "On AVX2 and AVX512 machines, you will generally need to enable reduce-range as well if
    per-channel is enabled." This is the per-channel interaction Anneal found. It is documented,
    but without a number.
  - S8S8 with QDQ is called the "first choice" default. The docs do not say that it performs badly
    on non-VNNI x86 (see §2).
  - The debugging API (`qdq_loss_debug`: `create_weight_matching`, `collect_activations`,
    `create_activation_matching`) compares float and quantized activations per tensor. This
    **could** localise saturation damage, but only when run on the affected hardware, and it is
    not framed as a saturation tool.

### onnxruntime: source (v1.30.0)
- `core/mlas/lib/amd64/QgemmU8S8KernelAvx2.asm`, `x86_64/QgemmU8S8KernelAvx2.S` and
  `x86_64/ConvSymKernelAvx2.S` use `vpmaddubsw` then `vpmaddwd`. The AvxVnni variant uses
  `vpdpbusds`. [src](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/x86_64/ConvSymKernelAvx2.S)
- **The closest prior art to Anneal's analyser** is
  [`saturation_check_avx2.cpp`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/intrinsics/avx2/saturation_check_avx2.cpp)
  plus the CMake option `onnxruntime_ENABLE_CONVSYMKERNELAVX2_SAT_CHECKER` ("Experimental", OFF),
  added in [PR #24220](https://github.com/microsoft/onnxruntime/pull/24220) (Apr 2025). Before each
  VPMADDUBSW in the **ConvSym AVX2 kernel only**, it recomputes the 16 pairs in int32 and logs
  `"Warning: saturation detected in VPMADDUBSW instruction."` once per session. The PR's motivation
  is *"On some models running with AVX2 (instead of AVX-VNNI), we've observed accuracy degradation
  due to saturation"*, with no numbers. How it differs from Anneal:
  (a) it is a custom build that must run **on** AVX2 hardware, so it cannot predict;
  (b) it gives no per-layer attribution, counts or error magnitude;
  (c) it does **not** cover the QGEMM path, so it would miss ResNet's stem conv (below);
  (d) it applies no remedy.
- **Which path the stem uses.** MLAS ConvSym requires `InputChannels % 4 == 0` on AVX2
  ([convsym.cpp](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/convsym.cpp)),
  so a 3-channel stem conv falls to the NHWC im2col + QGEMM path. K is then ordered (kh, kw, cin)
  and packed in groups of 4, which is exactly the pairing Anneal's `saturation.py` assumes. For
  cin % 4 == 0 layers on the ConvSym path, weights are packed per kernel position in blocks of 4
  input channels, so pairs are again adjacent input channels. **Anneal's pairing assumption is
  consistent with ORT 1.30 for both paths.** Worth stating in the README with these links.
- **Platform-level detection exists; per-layer detection does not.**
  - `MlasPlatformU8S8Overflow()` returns `GemmU8U8Dispatch != GemmU8S8Dispatch`. This is a coarse
    test: the dispatch *tables* also differ on AVX-512-VNNI machines, where only the kernel
    pointer changes.
  - The session option `session.x64quantprecision` (`kOrtSessionOptionsAvx2PrecisionMode`) is
    documented in the header as: *"x64 SSE4.1/AVX2/AVX512(with no VNNI) has overflow problem with
    quantized matrix multiplication with U8S8. To avoid this we need to use slower U8U8…"*. When
    set, the `Avx2WeightS8ToU8Transformer` rewrites the s8 weights of QLinearConv, MatMulInteger,
    QGemm, DynamicQuantizeMatMul and others to u8 for **all** layers.
    [config keys](https://github.com/microsoft/onnxruntime/blob/v1.30.0/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h) ·
    [transformer](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/qdq_transformer/avx2_weight_s8_to_u8.cc)
  - **This is a whole-model alternative guard that Anneal should benchmark against its
    per-layer guard.** It keeps full-range weights, costs some speed on every layer, and fixes
    saturation everywhere.

### OpenVINO POT and NNCF
- POT *Saturation (overflow) Issue Workaround* (2022.3; deprecated in 2023.3)
  [doc](https://docs.openvino.ai/2022.3/pot_saturation_issue.html) ·
  [source md](https://github.com/openvinotoolkit/openvino/blob/releases/2022/3/tools/pot/docs/SaturationIssue.md):
  - covers SSE, AVX2 and AVX-512 without VNNI. The issue is *"a common problem for models with
    non-ReLU activation functions and low level of redundancy (for example, optimized or efficient
    models)"*. That matches Anneal's MobileNetV3 result (−10.5pp) and the zero-point argument above.
  - it says the issue is **impossible to predict**; the only detection is a cross-CPU accuracy
    comparison, with a difference above 1% as the indicator.
  - `saturation_fix`: **"first_layer" is the default**, with "all" and "no" as alternatives.
- NNCF `nncf.OverflowFix` [src](https://github.com/openvinotoolkit/nncf/blob/develop/src/nncf/quantization/advanced_parameters.py):
  ENABLE, FIRST_LAYER ("weights of the first Convolutions of each model inputs are quantized using
  a half of the 8-bit quantization range") and DISABLE. The INT8 mode default is
  `OverflowFix.FIRST_LAYER` (`min_max/algorithm.py`); FP8 modes disable it.
  [FAQ](https://github.com/openvinotoolkit/nncf/blob/develop/docs/FAQ.md).
- **Relevance to Anneal.** Intel's shipped heuristic ("fix only the first conv") matches Anneal's
  ResNet-18 result (stem only). It would *miss* the two MobileNetV3 expand convs that Anneal's
  analyser flagged. The heuristic also halves the weight range, whereas Anneal keeps the layer in
  FP32. **A direct comparison would be informative:** NNCF-style FIRST_LAYER half-range, Anneal's
  guard, full reduce_range and ORT x64quantprecision, on the 9-CNN set.

### TensorFlow Lite / XNNPACK
- XNNPACK is optimised for **signed symmetric (QS8)** quantisation. QU8 is legacy and "may perform
  suboptimally on mobile processors with NEON DOT product instructions"
  ([TFLite XNNPACK delegate README](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/master/tensorflow/lite/delegates/xnnpack/);
  [TF blog 2021](https://blog.tensorflow.org/2021/09/faster-quantized-inference-with-xnnpack.html)).
- XNNPACK restricts symmetric weights to [−127, 127], so that its NEON MLAL fallback's two-product
  int16 partial sum cannot saturate (Cruz Romero & Maldonado Guerra 2026, below). TFLite's
  int8 spec makes weights symmetric in [−127, 127]. This prevents the −128·−128 corner, not u8×s8
  pair saturation.
- The x86 TFLite path: the paper below identifies PMADDUBSW saturating int16 intermediates as the
  x86 mechanism that breaks bit-reproducibility.

### Research that touches the same mechanism
- **Cruz Romero & Maldonado Guerra, *INT8 Quantization Makes ARM Edge Inference
  Dispatch-Invariant*, arXiv [2607.23227](https://arxiv.org/abs/2607.23227), Jul 2026.** This is
  the most closely related recent paper.
  - It proves that SDOT and NEON paths give byte-identical int32 results on ARM (Pi 4/5,
    Cortex-A53/A72/A76, XNNPACK).
  - It names **PMADDUBSW saturating int16 intermediates** as the x86 mechanism, "which has no ARM
    analogue", with a worked example (200·90 + 210·90 > 32,767), and cites the oneDNN and ORT docs.
  - It says vendor docs "treat the INT16 saturation step as a single-platform accuracy concern".
  - Its focus is **reproducibility / equivalence classes**, not accuracy magnitude. It has no
    per-layer predictor and no fix. **Anneal should cite it**, both as independent confirmation
    and to separate its own contribution (accuracy magnitude, per-layer prediction, targeted guard).
- Schlögl, Hofer & Böhme, *Causes and effects of unanticipated numerical deviations in neural
  network inference frameworks*, NeurIPS 2023
  ([paper](https://papers.neurips.cc/paper_files/paper/2023/hash/af076c3bdbf935b81d808e37c5ede463-Abstract-Conference.html)).
  In FP32, output equivalence classes across 75 x86 platforms follow SSE/AVX/AVX-512 dispatch.
- **Accumulator-overflow-aware quantisation** (training-time and bound-based, not ISA emulation):
  - A2Q (Colbert et al., ICCV 2023; [arXiv:2308.13504](https://arxiv.org/abs/2308.13504)) gives an
    L1-norm bound that guarantees no overflow for a P-bit accumulator. It is a relative of Anneal's
    "impossible" arithmetic bound, but for the whole dot product, not a pair.
  - A2Q+ (Colbert et al. 2024, [arXiv:2401.10432](https://arxiv.org/abs/2401.10432)).
  - Overflow-Aware Quantization (Xie et al. 2020, [arXiv:2005.13297](https://arxiv.org/abs/2005.13297))
    trains for 16-bit accumulation on ARM.
  - WrapNet (Ni et al., ICLR 2021).
  - None of these models VPMADDUBSW's *pairwise* int16 saturation on a given model.

### Verdict on Q1
- **Mechanism:** fully documented (Intel 2018, oneDNN, FBGEMM, ORT, OpenVINO/NNCF, PyTorch; ORT
  source).
- **Magnitude:** not quantified anywhere I found. The best in print is OpenVINO's ">1% means you
  have it" and "common for efficient / non-ReLU models". (ORT
  [issue #11415](https://github.com/microsoft/onnxruntime/issues/11415), MobileNetV3 per-channel,
  is a *bias int32 overflow* bug and should not be cited as saturation.) Anneal's controlled
  numbers look **unreported**:
  - 9 CNNs, −8 to −18pp on non-VNNI x86, against about 0pp for the same QDQ graph run in float;
  - ResNet-50 −13.1 vs +0.3; MobileNetV2 −18.2 vs −1.0;
  - five-CPU cross-check.
- **Per-layer prediction:** none in vendor tools. OpenVINO explicitly says it is impossible, and
  ORT's checker is runtime, whole-session and ConvSym-only. Anneal's emulator, and the guard that
  keeps only the flagged layers in FP32, appear **new**. The "impossible" verdict under
  reduce_range is a pairwise analogue of A2Q-style bounds.
- **Caveat on "nondeterministic".** POT claims saturation is nondeterministic because of
  parallelism. Given the fixed packing order in MLAS (K groups of 4, pairs adjacent), it is
  deterministic per kernel path. Anneal's analyser implicitly contradicts POT here, and could
  validate the point bit-exactly (see experiment E1).

---

## 2. ARM INT8: SDOT/UDOT, i8mm, and why S8S8 is fast on ARM but slow on x86

### Instructions
- **SDOT/UDOT** (FEAT_DotProd, Armv8.2+, mandatory from 8.4) do four 8-bit products summed into
  each int32 lane, with no saturation. The ARM ARM pseudocode has no SignedSat, and overflow wraps
  mod 2³². There is also **USDOT** (mixed sign) with i8mm.
- **i8mm** (FEAT_I8MM, Armv8.6; Neoverse N2/V1/V2, Apple M2+, *not* M1) adds **SMMLA, UMMLA and
  USMMLA**: a 2×8 by 8×2 int8 matrix multiply into a 2×2 int32 tile per instruction, i.e. 32 MACs
  against 16 for SDOT.
  - On **Neoverse N2**, SDOT/UDOT and SMMLA/UMMLA/USMMLA all have latency 3 and throughput 2/cycle
    on the V pipes ([N2 Software Optimization Guide](https://developer.arm.com/documentation/109914/latest/)),
    so i8mm doubles peak int8 MACs.
  - On Apple M-series, SDOT issues on 4 pipes and SMMLA on 2, so both reach about the same peak.
    rten's docs note that i8mm is "a significant improvement over SDOT" on Neoverse but equivalent
    on Apple M ([rten quantization.md](https://github.com/robertknight/rten/blob/main/docs/quantization.md)).
- **SME/SME2** (Apple M4, recent Android SoCs) are outer-product engines targeted by KleidiAI.

### How ORT (MLAS) dispatches (v1.30.0 `platform.cpp`, [src](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/platform.cpp))

| path | x86 without VNNI (AVX2 / AVX-512) | x86 AVX-512-VNNI or AVX-VNNI | x86 AVX-VNNI-INT8 | x86 AMX | ARM64 dotprod | ARM64 i8mm (Linux) |
|---|---|---|---|---|---|---|
| U8S8 GEMM | `…U8S8DispatchAvx2` (vpmaddubsw, **saturates**) | same dispatch, VNNI kernel | — | `MlasGemmU8S8DispatchAmx` | `U8X8DispatchUdot` | `U8X8DispatchUmmla` |
| **S8S8 GEMM** | **`MlasGemmQuantDispatchDefault` (portable C++)** | **Default** | `S8S8DispatchAvx2Vnni` | Default | `S8S8DispatchSdot` | `S8S8DispatchSmmla` (+SVE `svmmla`) |
| ConvSym U8S8 | Avx2 / Avx512Core | Avx512Vnni / AvxVnni | | | `ConvSymU8DispatchDot` | |
| ConvSym S8S8 | **none** | **none** | | | `ConvSymS8DispatchDot` | |

- `QDQIsInt8Allowed()` is `true` only on ARM
  ([qdq_selector_action_transformer.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h)).
  On x86, `QDQS8ToU8Transformer` rewrites int8 activation Q/DQ pairs to uint8 (it requires matching
  Q and DQ zero points and has a TODO for per-row). Where that rewrite does not apply, S8
  activations reach the Default S8S8 GEMM.
- **Explanation of Anneal's speed table.**
  - On ARM, S8S8 is the *native* format: SDOT/SMMLA are signed×signed, and ConvSym has an S8
    dotprod kernel.
  - U8S8 on ARM routes to U8X8 UDOT/UMMLA kernels, which need B flipped to unsigned (xor 0x80)
    plus zero-point fixups, so there is no penalty for S8S8 and a small tax on U8.
  - On x86, U8S8 is native (vpmaddubsw/vpdpbusd is unsigned×signed). S8S8 either costs an extra
    conversion pass or, for layers the rewrite misses, lands on the portable C++ QGEMM.
  - Anneal's Xeon 6973P-C (AVX-512-VNNI) still shows S8S8 per-tensor at 0.97x and per-channel at
    1.04x, while U8S8 gets 2.8–2.9x. That is consistent with this dispatch table. **To confirm,
    run ORT's profiler (`enable_profiling`) and read the kernel names per recipe.**
- **Hidden hardware effect in Anneal's lab.** The Xeon 6973P-C (Family 6, Model 173 = Granite
  Rapids) has **AMX-INT8**. ORT switches U8S8 GEMM to `MlasGemmU8S8DispatchAmx` when AMX-TILE and
  AMX-INT8 are present and the OS enables XTILE. Anneal's recorded `int8_features` list only
  avx2/avx512f/avx512vnni, so the "VNNI" row may partly be an AMX row. Record `amx_int8`,
  `amx_tile`, `avx_vnni`, `avx_vnni_int8`, `i8mm`, `sve`, `sve2` and `bf16`.

### XNNPACK / KleidiAI / TFLite
- XNNPACK picks microkernels per ISA via cpuinfo, with NEON, NEONDOT, NEONI8MM, AVX2, AVX-VNNI and
  AVX512-VNNI variants. QS8 is the primary scheme.
- Arm's **KleidiAI** microkernels (integrated into XNNPACK, and used by ORT for some
  MatMulNBits/SME2 paths) target DotProd, I8MM and SME2. The Arm blog reports int8
  dynamic-quantisation speedups for CNNs and GenAI from SDOT/i8mm kernels
  ([One year of KleidiAI in XNNPack](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/arm-kleidiai-in-xnnpack);
  [KleidiAI explainer](https://learn.arm.com/learning-paths/cross-platform/kleidiai-explainer/page1/)).
- On x86, XNNPACK QS8 kernels avoid the pair-saturation problem by sign-extending to int16 and using
  VPMADDWD, or by using VNNI. So TFLite+XNNPACK on non-VNNI x86 is a useful **control runtime**:
  same model, no pair saturation.

---

## 3. Hardware-aware optimisation tools and research

| tool / paper | automated search? | what it measures latency with | accuracy measured where? | cross-hardware validation? |
|---|---|---|---|---|
| **Microsoft Olive** ([docs](https://microsoft.github.io/Olive/why-olive.html), [design](https://microsoft.github.io/Olive/0.2.0/overview/design.html)) | Yes: search strategy over pass parameters (exhaustive / random / **TPE**) | **Real** runs of ORT on a configured *target system* (local, Docker, AzureML, python env) | On the target system if configured, otherwise on the host | No built-in cross-hardware study. Default recipes (S8S8 QDQ) are hardware-agnostic |
| **Intel NNCF / OpenVINO** ([NNCF](https://github.com/openvinotoolkit/nncf); [accuracy control](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/quantizing-with-accuracy-control.html)) | Partly: `quantize_with_accuracy_control` greedily reverts layers to FP by ranked sensitivity until drop ≤ max_drop | Not in the loop (`benchmark_app` is separate). A `target_device` preset (CPU/GPU/NPU) picks the quantisation scheme | Accuracy with the OpenVINO runtime on the host CPU, so saturation *would* show if the host lacks VNNI, but without being attributed to it | No. The hardware knowledge is encoded as presets (e.g. `OverflowFix.FIRST_LAYER`) |
| **Qualcomm AIMET** ([docs](https://quic.github.io/aimet-pages/)) | Yes: AutoQuant (BN fold, CLE, AdaRound against a target), AMP mixed-precision search, QuantAnalyzer per-layer sensitivity | Mixed-precision uses a **cost model** (BOPs/MACs). Real device profiling is a separate service (Qualcomm AI Hub) | **Simulated** (QuantSim fake-quant) on the host, with a HW config JSON for HTP/DSP | Not within AIMET. AI Hub measures on real phones but does not feed back automatically |
| **NVIDIA TensorRT** ([docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/)) | Yes, at kernel level: the builder times candidate **tactics** per layer on the actual GPU | **Real measurement** on the build GPU (timing cache). Engines are GPU- and version-specific unless hardware-compatibility mode is set | Not searched. INT8 calibration only | Implicitly: an engine is rebuilt per GPU. No cross-GPU accuracy check |
| **NVIDIA ModelOpt `auto_quantize`** ([API](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html)) | Yes: per-layer format search; gradient/Fisher sensitivity plus an LP under an `effective_bits` constraint | **None**: the constraint is effective bits, not latency | Simulated fake-quant | No |
| **Apache TVM AutoTVM / Ansor** ([Chen et al. NeurIPS'18](https://arxiv.org/abs/1805.08166); [Ansor, OSDI'20](https://arxiv.org/abs/2006.06762)) | Yes: schedule search (evolutionary plus a learned XGBoost cost model) | **Real measurement** on device via RPC; the cost model only prunes candidates | N/A (the tensor program is semantically fixed; quantisation is not searched) | Yes for speed: Ansor evaluates on Intel CPU, ARM CPU and NVIDIA GPU and finds hardware-specific schedules |
| **HAQ** (Wang et al., CVPR'19, [arXiv:1811.08886](https://arxiv.org/abs/1811.08886)) | Yes: DDPG RL chooses per-layer weight/activation bit widths | **Hardware simulators** (BISMO edge/cloud FPGA, BitFusion) give latency and energy directly, instead of FLOPs | Fake-quant plus short fine-tune on GPU | **Yes**: optimal policies differ between edge and cloud accelerators and between BISMO and BitFusion. Motivates "result belongs to the hardware" (simulated) |
| **HAWQ-V3** (Yao et al., ICML'21, [arXiv:2011.10680](https://arxiv.org/abs/2011.10680)) | Yes: ILP over per-layer bit widths with a Hessian sensitivity objective | **Measured** per-layer latency of TVM-generated INT4/INT8 kernels on an **NVIDIA T4** (lookup) | Integer-only (dyadic) inference, verified to match | No: T4 only |
| **APQ** (Wang et al., CVPR'20, [arXiv:2006.08509](https://arxiv.org/abs/2006.08509)) | Yes: evolutionary joint search over architecture, pruning and mixed-precision quantisation | **Lookup tables** of latency/energy for the BitFusion accelerator | Quantisation-aware **accuracy predictor** (transferred from an FP predictor) | Limited (one accelerator family) |
| **Once-for-All** (Cai et al., ICLR'20, [arXiv:1908.09791](https://arxiv.org/abs/1908.09791)) | Yes: evolutionary search over subnets of a supernet | **Latency lookup tables / predictors built from measurements** on each target (several phones, GPUs, Xeon CPU, Jetson, FPGAs) | Accuracy predictor, then real evaluation | **Yes for latency**: different optimal subnets per device |
| nn-Meter (Zhang et al., MobiSys'21, [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/05/nn-Meter-Mobisys21.pdf)) | (predictor, not optimiser) | Kernel-level predictor that detects fusion via test cases; 99% ±10% accuracy on mobile CPU and GPU, 83.4% on VPU | — | Yes (3 device types); shows FLOPs are a poor proxy |
| HW-NAS-Bench (Li et al., ICLR'21, [arXiv:2103.10584](https://arxiv.org/abs/2103.10584)) | benchmark | Measured or estimated latency and energy on 6 devices (edge GPU, Raspberry Pi 4, Edge TPU, Pixel 3, ASIC, FPGA) | — | Yes: latency rankings correlate poorly across devices |

**What this means for Anneal.**
- Every hardware-aware tool above treats **latency** as hardware-dependent. Almost all treat
  **accuracy** as a model property, evaluated by fake-quant simulation on the host (AIMET, ModelOpt,
  HAQ, HAWQ-V3, APQ; NNCF evaluates on the host CPU).
- Anneal's claim that **accuracy (and per-layer sensitivity) is a property of the hardware's integer
  arithmetic** is the gap. Simulation-based tools cannot see VPMADDUBSW saturation: fake-quant runs
  in float, which is exactly Anneal's "QDQ in float ≈ 0pp" control.
- Olive can see it only if its evaluator runs on the affected CPU. TVM, TensorRT and OFA use real
  measurement but for speed only.
- Anneal's measured-only ledger, paired accuracy tests and cross-CPU lab have no direct equivalent.
  The closest are HAQ's cross-accelerator policies (simulated) and OFA's per-device specialisation
  (latency only).

---

## 4. Measurement methodology

- **MLPerf Inference** (Reddi et al., ISCA 2020, [arXiv:1911.02549](https://arxiv.org/abs/1911.02549);
  [rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)):
  - LoadGen-driven runs of at least **600 s**.
  - Minimum query counts tied to the tail percentile: 24,576 for p90, 270,336 for p99, with an
    **early-stopping** rule based on the binomial CDF.
  - Accuracy targets of 99% (or 99.9%) of the FP32 reference, from a separate accuracy run over the
    whole validation set.
  - Results must be replicable (audits). Power runs use SPEC PTDaemon.
  - Anneal's p50/p90/p99 at a few hundred iterations are far below MLPerf's p99 sample size.
    Either report p99 with a CI, or drop it.
- **MLPerf Mobile** (Reddi et al., MLSys 2022, [arXiv:2012.02328](https://arxiv.org/abs/2012.02328))
  addresses thermal throttling on phones (fixed run lengths, cooldown between benchmarks).
- **Scientific benchmarking:**
  - Hoefler & Belli, *Scientific Benchmarking of Parallel Computing Systems*, SC'15
    ([PDF](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)): 12
    rules, including reporting nonparametric CIs, stating the summary statistic and testing
    normality.
  - Chen & Revels, *Robust benchmarking in noisy environments*, 2016
    ([arXiv:1608.04295](https://arxiv.org/abs/1608.04295)): the minimum is a robust estimator for
    deterministic code.
  - Mytkowicz et al., *Producing wrong data without doing anything obviously wrong!*, ASPLOS'09:
    measurement bias from environment size and link order.
- **Cloud and CI noise (relevant to GitHub Actions):**
  - Laaber, Scheuner & Leitner, *Software microbenchmarking in the cloud. How bad is it really?*,
    EMSE 2019 ([arXiv:1903.07393](https://arxiv.org/abs/1903.07393)): large variability across
    instances. Recommends repeated trials across instances and randomised interleaving.
  - **Duet benchmarking** (Bulej et al., ICPE 2020, [arXiv:2001.05811](https://arxiv.org/abs/2001.05811)):
    running two variants *concurrently* on the same VM and comparing ratios cuts noise 2–82x.
  - Anneal's drift re-timing is a sequential cousin of this. Randomised A/B interleaving (ABBA
    ordering) is the cheap upgrade.
- **Mobile and edge variance:**
  - Wu et al., *Machine Learning at Facebook: Understanding Inference at the Edge*, HPCA 2019
    ([paper](https://research.facebook.com/publications/machine-learning-at-facebook-understanding-inference-at-the-edge/)):
    large performance variability across devices and from thermal state.
  - Ignatov et al., *AI Benchmark* (ECCVW 2018 / ICCVW 2019).
- **Accuracy comparison:** Dietterich, *Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms*, Neural Computation 1998 recommends **McNemar's test** for
  paired single-test-set comparisons. This is exactly Anneal's paired accuracy test and is the
  canonical citation.
- **Reproducibility of numerics:** Schlögl et al. (NeurIPS'23) and Cruz Romero & Maldonado Guerra
  (2026), above.
- **Assessment.** Anneal's methodology is already above typical ML-systems practice:
  - warm-up discard, percentiles, battery/AC detection, drift re-timing and paired McNemar.
  - Gaps: CIs on latency ratios (bootstrap over repeats), interleaved A/B ordering, fixed thread
    pinning and affinity, and recording the CPU frequency governor and turbo state.
  - On GitHub runners the CPU model is *assigned*. Record it (already done) and treat runner
    identity as a random effect.

---

## 5. Hardware effects Anneal may have missed, and what to add to the lab

The GitHub-hosted inventory, as observed in Anneal's lab:
- ubuntu/windows x64 give **AMD EPYC 7763** (Zen 3, AVX2, no VNNI) or, sometimes, **Intel Xeon
  6973P-C** (Granite Rapids: AVX-512-VNNI **and AMX**).
- `ubuntu-24.04-arm` gives **Neoverse N2** (Azure Cobalt 100: dotprod, i8mm, SVE2, bf16).
- `macos-latest` gives **Apple M1** (dotprod, fp16, **no i8mm**, no bf16).
- `windows-11-arm` also exists and runs on Cobalt 100 (Neoverse N2).

| effect | why it matters | ORT hook | worth adding? |
|---|---|---|---|
| **i8mm (SMMLA) vs SDOT** | 2x MACs/instr on N2. ORT uses SMMLA only on Linux (the GAS-only kernels); off Linux the SVE `svmmla` path is used if compiled in | Automatic dispatch; no user switch | **Yes (E2).** Same N2 silicon, Linux vs Windows-ARM, *if* the Windows build lacks SMMLA (check with the profiler or a disassembly of kernel symbols). Also an i8mm-free ARM (M1) vs N2 at equal clock |
| **AMX on Sapphire/Granite Rapids** | U8S8 GEMM goes to AMX tiles (`MlasGemmU8S8DispatchAmx`). Also changes accuracy? No: int32 accumulation. Speed can jump for large GEMMs; convs on ConvSym still use AVX-512-VNNI | Automatic when XTILE is enabled | **Record it now** (cpuinfo flags). Hosted runners don't guarantee Intel, so use the random draw, or Intel **SDE** `-spr`/`-gnr` for correctness only (not timing) |
| **AVX-VNNI-INT8** (Arrow/Lunar Lake, Sierra Forest) | The only x86 ISA where ORT has a fast S8S8 GEMM | Automatic | Unlikely on hosted runners. Note it as a prediction: S8S8 should stop being slowest there |
| **fp16 on ARM** | M1 and N2 have FEAT_FP16. ORT CPU EP has NEON fp16 kernels for Conv and others, a "no-calibration" recipe with ~0 accuracy loss | FP16 model + CPU EP (ARM64) | **Yes (E3)**, as a recipe axis alongside INT8 on the ARM runners |
| **bf16 fast-math on N2** | FP32 GEMM via BFMMLA | `mlas.enable_gemm_fastmath_arm64_bfloat16=1` | Yes, cheap: one session flag on `ubuntu-24.04-arm` (M1 lacks bf16, a nice negative control) |
| **Denormals** | x86 denormal micro-code assists can slow FP32 badly. ORT doc: FTZ/DAZ "may hurt model accuracy" | `session.set_denormal_as_zero=1` | Low priority for CNN inference (few denormals after BN folding). A one-flag check of the FP32 baseline and calibration statistics |
| **U8U8 / `x64quantprecision`** | Documented whole-model saturation fix on x86 | `session.x64quantprecision=1` | **Yes**: the natural baseline against Anneal's per-layer guard |
| **Thread count / SMT / frequency** | Hosted runners are 2–4 vCPU with shared SMT; turbo varies | intra_op threads, affinity | Already via cpu-1t/cpu-4t. Record frequency if possible |

### Top 3 hardware experiments to add
1. **E1: bit-exact ground truth for the analyser, on any runner, via Intel SDE.**
   - Run single-layer QLinearConv/QGEMM models under Intel SDE emulating Haswell (`-hsw`, AVX2 with
     no VNNI) against Cascade Lake (`-clx`, VNNI). ORT will then dispatch the saturating kernels.
   - Compare the int32 outputs with Anneal's emulator layer by layer. This turns the "pairing order
     is an assumption" caveat into a verified fact, for both the ConvSym path (cin % 4 == 0) and the
     QGEMM path (stem).
   - Optional cross-check: an ORT build with `onnxruntime_ENABLE_CONVSYMKERNELAVX2_SAT_CHECKER=ON`,
     extended to count per call.
   - SDE is a free Intel download that needs licence acceptance. It is fine for correctness, not for
     timing.
   - It also removes the dependence on drawing an AMD runner.
2. **E2: i8mm vs SDOT on identical silicon.**
   - Compare `ubuntu-24.04-arm` against `windows-11-arm` (both Cobalt 100 / N2), after confirming
     with ORT profiling which S8S8/U8S8 kernels each picks.
   - Alternatively, compare ORT against TFLite/XNNPACK with and without i8mm kernels on the same box.
   - This separates "ARM is fast because of dot-product" from "because of i8mm", and tests whether
     the S8S8-vs-U8S8 gap on ARM comes from the U8→X8 fixup.
3. **E3: the guard vs documented whole-model fixes, plus AMX and fp16/bf16 recipe axes.**
   - On non-VNNI x86, over the 9-CNN set, compare:
     - Anneal's per-layer guard;
     - NNCF-style FIRST_LAYER half-range;
     - full `reduce_range`;
     - ORT `session.x64quantprecision=1` (U8U8);
     - TFLite/XNNPACK QS8 (no pair saturation).
   - Report accuracy **and** latency. This is the head-to-head the literature lacks.
   - On the ARM runners, add FP16 (M1, N2) and bf16 fast-math (N2) as recipes.
   - Record AMX, AVX-VNNI-INT8 and i8mm flags in every lab JSON, so rows like the Xeon's are
     labelled by the path that actually ran.

---

## Sources (primary)
- oneDNN int8 nuances: https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html
- Intel 2018 white paper: https://www.intel.com/content/dam/develop/external/us/en/documents/lower-numerical-precision-deep-learning-jan2018-754765.pdf
- FBGEMM paper: https://arxiv.org/abs/2101.05615 · FBGEMM issue #199: https://github.com/pytorch/FBGEMM/issues/199
- PyTorch qconfig: https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/qconfig.py
- ORT quantisation docs: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ORT MLAS platform dispatch (v1.30.0): https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/platform.cpp
- ORT saturation checker PR #24220: https://github.com/microsoft/onnxruntime/pull/24220
- ORT session config keys: https://github.com/microsoft/onnxruntime/blob/v1.30.0/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
- OpenVINO POT saturation doc: https://github.com/openvinotoolkit/openvino/blob/releases/2022/3/tools/pot/docs/SaturationIssue.md
- NNCF OverflowFix: https://github.com/openvinotoolkit/nncf/blob/develop/src/nncf/quantization/advanced_parameters.py
- Cruz Romero & Maldonado Guerra 2026: https://arxiv.org/abs/2607.23227
- Schlögl et al. NeurIPS 2023: https://papers.neurips.cc/paper_files/paper/2023/hash/af076c3bdbf935b81d808e37c5ede463-Abstract-Conference.html
- rten quantisation notes: https://github.com/robertknight/rten/blob/main/docs/quantization.md
- Neoverse N2 SWOG: https://developer.arm.com/documentation/109914/latest/
- KleidiAI in XNNPack: https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/arm-kleidiai-in-xnnpack
- XNNPACK quantised inference: https://blog.tensorflow.org/2021/09/faster-quantized-inference-with-xnnpack.html
- MLPerf Inference rules: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
- Duet benchmarking: https://arxiv.org/abs/2001.05811 · Laaber et al.: https://arxiv.org/abs/1903.07393
- nn-Meter: https://www.microsoft.com/en-us/research/wp-content/uploads/2021/05/nn-Meter-Mobisys21.pdf
- HAQ / HAWQ-V3 / APQ / OFA / Ansor: arXiv 1811.08886 / 2011.10680 / 2006.08509 / 1908.09791 / 2006.06762

*Not verified directly (from memory or secondary sources; check before citing):* the MLPerf Mobile
cooldown detail, HW-NAS-Bench's exact device list, and Wu et al. HPCA'19 specifics.
