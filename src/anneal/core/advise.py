"""Recommend a static INT8 recipe from a model's graph and the target CPU's INT8 arithmetic.

Every rule here is a finding measured in this repository, not a best practice copied from
elsewhere. Each recommendation carries the evidence it rests on and how confident that
evidence is. The two inputs that decide the recipe are:

* **the architecture family**, read from the graph (no quantization, no data):

  ``transformer``       LayerNorm + mostly MatMul/Gemm (ViT, Swin). onnxruntime's default
                        also quantizes LayerNorm inputs, i.e. the residual stream, whose
                        outlier channels starve under one 8-bit scale: ViT-B/16 lost 8pp.
                        Quantizing only the matrix products recovers it (+0.6pp).
  ``convnext``          LayerNorm + depthwise convs. No recipe tested here gets within 2pp;
                        the advice says so.
  ``gated-depthwise``   Conv -> SiLU/Hardswish -> depthwise chains (EfficientNet,
                        MobileNetV3, and MobileViT, whose softmax attention sits between
                        such blocks). Per-tensor activation scales collapse these on every
                        CPU; equalisation + asymmetric percentile + float stem fixes them.
  ``gated-linear-attention``  the same chains plus linear attention (MatMul, no Softmax;
                        EfficientViT). Every per-tensor recipe loses ~72pp; attention MatMuls
                        in float + gated and dense equalisation + four 16-bit tensors: -7pp.
  ``cnn``               everything else with convolutions (ResNet, RegNet, MobileNetV2,
                        ShuffleNet, MnasNet). Fine on 32-bit hardware; on x86 without VNNI
                        the stem saturates, which keeping it in float removes. Symmetric
                        99.999 percentile; reduce_range and a 99.99 percentile both cost
                        ~0.5pp on ImageNet.

* **the INT8 path** (:func:`anneal.core.environment.cpu_features`): ``x86-avx2-16bit`` is the
  one whose 16-bit pair sums saturate; ``x86-vnni`` and ``arm-dotprod`` accumulate in 32 bits.

* optionally **an accelerator target** (:data:`ACCELERATOR_TARGETS`): ``tidl`` (TI TDA4VM) and
  ``amd-xint8`` (AMD NPUs) quantize weights per tensor and feature maps symmetrically with
  power-of-two scales. Every candidate then carries those emulation flags, so :func:`verify`
  scores something close to the hardware, and the recipe is the per-tensor one measured there.

The advice is a starting point with stated confidence. :func:`verify` measures it — the
recommended recipe, its alternatives and onnxruntime's default side by side — because a rule
learned on eleven models can be wrong on the twelfth.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

INT8_PATHS = ("x86-avx2-16bit", "x86-vnni", "arm-dotprod", "unknown")
BASE = {"per_channel": True, "activation_type": "uint8", "calib_samples": 64}
P_STEM = {"calibrate_method": "percentile_asym", "float_stem": True}

ACCELERATOR_TARGETS = ("tidl", "amd-xint8")
#: How TI TIDL and AMD XINT8 quantize: per-tensor weights, symmetric int8 feature maps with
#: power-of-two scales. Added to every candidate for such a target so verify emulates it.
ACCELERATOR_EMULATION = {
    "per_channel": False,
    "activation_type": "int8",
    "activation_symmetric": True,
    "pow2_activation_scales": True,
}
TARGET_NAMES = {"tidl": "TI TIDL (TDA4VM)", "amd-xint8": "AMD XINT8 (Ryzen AI / Vitis AI NPU)"}


@dataclass
class ModelProfile:
    convs: int
    depthwise: int
    matmuls: int
    layernorms: int
    gated_sites: int
    relu_sites: int
    activations: dict[str, int]
    family: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Candidate:
    label: str
    params: dict[str, Any]
    why: str


@dataclass
class Advice:
    profile: ModelProfile
    int8_path: str
    recommended: Candidate
    alternatives: list[Candidate]
    confidence: str  # "high", "medium", "low"
    evidence: list[str]
    caveats: list[str] = field(default_factory=list)
    #: An accelerator from :data:`ACCELERATOR_TARGETS`, or None for a CPU.
    target: str | None = None

    def candidates(self) -> list[Candidate]:
        return [self.recommended, *self.alternatives]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "int8_path": self.int8_path,
            "target": self.target,
            "recommended": self.recommended.__dict__,
            "alternatives": [c.__dict__ for c in self.alternatives],
            "confidence": self.confidence,
            "evidence": self.evidence,
            "caveats": self.caveats,
        }


# ---------------------------------------------------------------------------
# Reading the graph
# ---------------------------------------------------------------------------


def profile(model_path: Path) -> ModelProfile:
    """What kind of network this is, from its ONNX graph alone."""
    import onnx

    from anneal.core.equalize import find_sites

    model = onnx.load(str(model_path))
    ops = Counter(n.op_type for n in model.graph.node)
    inits = {i.name: i for i in model.graph.initializer}
    depthwise = 0
    for n in model.graph.node:
        if n.op_type == "Conv" and len(n.input) > 1 and n.input[1] in inits:
            group = next((a.i for a in n.attribute if a.name == "group"), 1)
            if group > 1 and inits[n.input[1]].dims[1] == 1:
                depthwise += 1
    sites = find_sites(model)
    gated = sum(s.kind == "gated" for s in sites)
    relu = sum(s.kind == "relu" for s in sites)
    activations = {
        k: v for k, v in {
            "SiLU (Sigmoid*x)": _count_silu(model),
            "Hardswish": ops["HardSwish"] + ops["HardSigmoid"],
            "ReLU": ops["Relu"],
            "ReLU6/Clip": ops["Clip"],
            "GELU": ops["Gelu"] + ops["Erf"],
        }.items() if v
    }
    convs, matmuls, lns = ops["Conv"], ops["MatMul"] + ops["Gemm"], ops["LayerNormalization"]
    lns += _count_decomposed_layernorm(model)

    # Gated chains decide first: MobileViT and EfficientViT also have LayerNorm and depthwise
    # convs, and were misfiled as convnext (docs/zoo_gated_predictions.md, prediction 4).
    if gated and ops["MatMul"] and not ops["Softmax"]:
        family = "gated-linear-attention"
    elif gated:
        family = "gated-depthwise"
    elif lns and depthwise:
        family = "convnext"
    elif lns and matmuls > convs:
        family = "transformer"
    elif convs:
        family = "cnn"
    else:
        family = "other"
    return ModelProfile(convs, depthwise, matmuls, lns, gated, relu, activations, family)


def _count_silu(model) -> int:
    producer = {o: n for n in model.graph.node for o in n.output}
    count = 0
    for n in model.graph.node:
        if n.op_type == "Mul" and len(n.input) == 2:
            for a, b in ((n.input[0], n.input[1]), (n.input[1], n.input[0])):
                g = producer.get(b)
                if g is not None and g.op_type == "Sigmoid" and g.input[0] == a:
                    count += 1
                    break
    return count


def _count_decomposed_layernorm(model) -> int:
    """LayerNorm exported as ReduceMean -> Sub -> Pow -> ReduceMean -> ... (older opsets)."""
    producer = {o: n for n in model.graph.node for o in n.output}
    count = 0
    for n in model.graph.node:
        if n.op_type == "Sub":
            p = producer.get(n.input[1]) if len(n.input) > 1 else None
            if p is not None and p.op_type == "ReduceMean" and p.input[0] == n.input[0]:
                count += 1
    return count


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


def advise(model_path: Path, int8_path: str = "unknown", target: str | None = None) -> Advice:
    """The recipe for this model on a CPU with ``int8_path``, or on an accelerator ``target``.

    With a ``target`` the CPU's INT8 path does not matter (the accelerator does the arithmetic)
    and the per-tensor-weight recipe measured on that target replaces the CPU one.
    """
    if int8_path not in INT8_PATHS:
        raise ValueError(f"int8_path must be one of {INT8_PATHS}, got {int8_path!r}")
    if target is not None and target not in ACCELERATOR_TARGETS:
        raise ValueError(f"target must be one of {ACCELERATOR_TARGETS} or None, got {target!r}")
    prof = profile(model_path)
    if target is not None:
        return _advise_target(prof, int8_path, target)
    return _advise_cpu(prof, int8_path)


def _advise_cpu(prof: ModelProfile, int8_path: str) -> Advice:
    saturating = int8_path == "x86-avx2-16bit"
    caveats: list[str] = []
    if int8_path == "unknown":
        caveats.append(
            "The target's INT8 arithmetic is unknown; advice assumes 32-bit accumulation. On an "
            "x86 CPU without VNNI, keep the stem in float or run `anneal saturation`."
        )

    if prof.family == "transformer":
        rec = Candidate(
            "compute ops only",
            {**BASE, "calibrate_method": "minmax", "quantize_ops": "compute"},
            "Quantize only the matrix products; leave LayerNorm, residual adds and softmax in "
            "float, so the residual stream's outlier channels never meet an 8-bit scale.",
        )
        alts = [Candidate("onnxruntime default", {**BASE, "calibrate_method": "minmax"},
                          "Control: quantizes LayerNorm inputs too.")]
        evidence = [
            "ViT-B/16, 512 images: default -8.0pp fused / -8.4pp emulated; compute-only +0.6 / +0.2.",
            "Floating only the 25 LayerNorm inputs recovered ~5pp (causal ablation).",
            "Swin-T: compute-only -2.5 / +0.2, within noise at 512 images.",
            "No x86 saturation seen on transformer MatMuls (fused and emulated agree within noise).",
        ]
        confidence = "medium"
        caveats.append("Evidence is two models on 512 images each.")
    elif prof.family == "convnext":
        rec = Candidate(
            "percentile + float stem + reduce_range",
            {**BASE, **P_STEM, "reduce_range": True},
            "The least-bad recipe measured; none gets within 2pp.",
        )
        alts = [
            Candidate("onnxruntime default", {**BASE, "calibrate_method": "minmax"},
                      "Control. On ConvNeXt-Tiny its accuracy looked fine, but it agreed with FP32 on only 85% of images."),
            Candidate("compute ops only", {**BASE, "calibrate_method": "minmax", "quantize_ops": "compute"},
                      "The transformer fix; did not carry over to ConvNeXt-Tiny."),
        ]
        evidence = [
            "ConvNeXt-Tiny, 1,024 images: every recipe lost 2-4pp; floating any single op group did not recover it.",
            "A prototype of gate-side equalisation into the dense fc2 layer reached -1.0pp, at the noise limit.",
        ]
        confidence = "low"
        caveats.append("No validated recipe for this family: verify, and consider keeping the model FP32.")
    elif prof.family == "gated-linear-attention":
        rec = Candidate(
            "attention in float + gated, residual and dense equalisation + 4 tensors at 16 bits",
            {**BASE, "calibrate_method": "percentile_asym", "calib_percentile": 99.99, "quantize_ops": "conv",
             "equalize": True, "equalize_residual": True, "equalize_dense": True, "int16_top_k": 4},
            "Quantize only Conv/Gemm (the linear attention's MatMuls and normaliser stay float), "
            "equalise the gated chains, the stem's residual one and the dense consumers, and keep "
            "the four most damaged tensors at 16 bits.",
        )
        alts = [
            Candidate("attention in float + equalisation",
                      {**BASE, "calibrate_method": "percentile_asym", "calib_percentile": 99.99,
                       "quantize_ops": "conv", "equalize": True, "equalize_dense": True},
                      "Without the 16-bit tensors and the residual site: no mixed precision needed."),
            Candidate("onnxruntime default", {**BASE, "calibrate_method": "minmax"},
                      "Control: every per-tensor recipe tested lost ~72pp on EfficientViT-B0."),
        ]
        evidence = [
            "EfficientViT-B0, Imagenette 2,048 images: -72pp for every standard recipe; this recipe -7.0pp "
            "(residual site +5.7pp, p=9e-10).",
            "EfficientViT-B1, same: -80pp -> -7.0pp; the residual site is neutral there (-0.2pp, p=0.73).",
        ]
        confidence = "medium"
        caveats.append("Measured on Imagenette only, and ~7pp remain: verify; FP16 may be the better target.")
    elif prof.family == "gated-depthwise":
        eq = {**BASE, **P_STEM, "equalize": True}
        # reduce_range helped the SiLU net on non-VNNI x86 (EfficientNet-B0: -3.1 -> -0.6pp)
        # and hurt the Hardswish one (MobileNetV3: -2.6 -> -4.8pp), so it follows the gate.
        silu = prof.activations.get("SiLU (Sigmoid*x)", 0) >= prof.activations.get("Hardswish", 0)
        if saturating and silu:
            rec = Candidate(
                "equalise + percentile + float stem + reduce_range", {**eq, "reduce_range": True},
                "Equalise the SiLU/Hardswish -> depthwise chains exactly, calibrate by asymmetric "
                "percentile, keep the stem in float; 7-bit weights because equalised channels "
                "fill their range and saturate the 16-bit pair sums of this CPU.",
            )
            alts = [
                Candidate("equalise + percentile + float stem", eq,
                          "Without reduce_range: best for MobileNetV3 even on non-VNNI x86 in the lab."),
                Candidate("equalise + percentile + float stem + guard", {**eq, "guard_saturation": True},
                          "Keep only the saturating layers in FP32 instead of shrinking every weight."),
            ]
        else:
            rec = Candidate(
                "equalise + percentile + float stem", eq,
                "Equalise the SiLU/Hardswish -> depthwise chains exactly, calibrate by asymmetric "
                "percentile, keep the stem in float.",
            )
            alts = [Candidate("percentile + float stem", {**BASE, **P_STEM},
                              "Without equalisation: faster (no gate multiplies), a few pp less accurate on EfficientNet.")]
            if saturating:
                alts.insert(0, Candidate(
                    "equalise + percentile + float stem + reduce_range", {**eq, "reduce_range": True},
                    "7-bit weights: removed EfficientNet's saturation, but cost MobileNetV3 2pp more."))
        alts.append(Candidate("onnxruntime default", {**BASE, "calibrate_method": "minmax"},
                              "Control: per-tensor activation scales collapse this family."))
        evidence = [
            "EfficientNet-B0 on six CPUs, 2,048 images: default -48 to -51pp everywhere; "
            "this recipe -0.5 to -0.7pp (not significant); with reduce_range -0.6pp on non-VNNI x86.",
            "MobileNetV3-Large, same six CPUs: -11 to -25pp -> -1.0 to -2.6pp.",
            "Cost: about 10% of the INT8 speedup for the gate multiplies (2.33x -> 2.11x on ARM).",
        ]
        confidence = "high"
        caveats.append(f"{prof.gated_sites} gated chains will be equalised. EfficientNet-B1 kept ~7pp "
                       "after this recipe; verify.")
    elif prof.family == "cnn":
        # ImageNet ablation (examples/advise/resnet50_ablation.json, 10,000 images): the earlier
        # advice, asymmetric 99.99 percentile + float stem (+ reduce_range on this CPU), lost
        # 0.44pp (0.65pp fused) to plain symmetric 99.999 percentile. 99.99 clips too much, and
        # reduce_range costs ~0.5pp once the float stem has removed the saturation.
        rec = Candidate(
            "percentile 99.999 + float stem",
            {**BASE, "calibrate_method": "percentile", "calib_percentile": 99.999, "float_stem": True},
            "Symmetric 99.999 percentile calibration, stem kept in float"
            + ("; the float stem is what removes this CPU's 16-bit pair-sum saturation." if saturating
               else "; the stem costs nothing measurable here and protects non-VNNI x86."),
        )
        alts = [Candidate("minmax + float stem", {**BASE, "calibrate_method": "minmax", "float_stem": True},
                          "Within noise of the recommendation on ResNet-50 (ImageNet).")]
        alts.append(Candidate("onnxruntime default", {**BASE, "calibrate_method": "minmax"},
                              "Control: fine on 32-bit hardware, 8-18pp worse on non-VNNI x86."))
        evidence = [
            "ResNet-50, ImageNet (10,000 images): -0.14pp emulated, -0.04pp on non-VNNI x86 (real kernels); "
            "adding reduce_range: -0.63pp; asymmetric 99.99 percentile: -0.54pp.",
            "ResNet-50, MobileNetV2, RegNetY, ShuffleNetV2, MnasNet (1,024 images): default -8 to -18pp "
            "on non-VNNI x86, ~0 emulated.",
            "ResNet-18 on five CPUs: the loss appears only on x86 without VNNI; the stem is the saturating layer.",
        ]
        confidence = "high"
    else:
        rec = Candidate("onnxruntime default", {**BASE, "calibrate_method": "minmax"},
                        "No convolutions or recognised structure; nothing measured here applies.")
        alts = []
        evidence = []
        confidence = "low"
        caveats.append("Outside the families this advice was measured on.")

    if saturating and prof.family in ("cnn", "gated-depthwise"):
        # Measured on AC power, cpu-1t, interleaved rounds (examples/advise/*-timing.json): the
        # recipes that keep accuracy are no faster than FP32 here; the fast default is fast only
        # because it quantizes the stem, which is what breaks it.
        caveats.append(
            "On x86 without VNNI the accurate recipes ran at about FP32 speed (EfficientNet-B0 "
            "1.04x, ResNet-50 0.99x); the default's 1.05-1.12x comes from the quantized stem that "
            "costs 13-51pp. INT8 may not be worth deploying on this CPU; the same recipes gave "
            "2-4x on ARM."
        )
    return Advice(prof, int8_path, rec, alts, confidence, evidence, caveats)


def _emulated(c: Candidate) -> Candidate:
    """The same recipe with the accelerator's quantization flags (which override its own)."""
    params = {**c.params, **ACCELERATOR_EMULATION}
    if params.pop("equalize_dense", False) and not params.get("equalize"):
        # equalize_dense needs per-channel weights, which these accelerators do not have
        params.pop("float_gates", None)
    return Candidate(c.label, params, c.why)


def _advise_target(prof: ModelProfile, int8_path: str, target: str) -> Advice:
    """Per-tensor weights, symmetric power-of-two feature maps: TIDL and AMD XINT8."""
    name = TARGET_NAMES[target]
    base = {"calib_samples": BASE["calib_samples"], "calibrate_method": "minmax", **ACCELERATOR_EMULATION}
    if target == "amd-xint8":
        # AMD's default power-of-two MinMSE calibration cost EfficientNet-B0 ~25pp on XINT8 (with
        # per-tensor equalisation, exact sigmoid): -28.6pp, MinMax -24.2, Percentile -3.7.
        base = {**base, "calibrate_method": "percentile", "calib_percentile": 99.999}
    control = Candidate(
        "8-bit control", dict(base),
        f"Control: {name}'s plain 8-bit quantization, per-tensor weights, no equalisation.",
    )
    caveats = [
        f"Scored by emulating {name} in onnxruntime (per-tensor weights, symmetric int8 feature "
        "maps, power-of-two scales); the accelerator's own calibration and layer fusion differ, "
        "so confirm on the device before trusting a difference of a point or two.",
        "16-bit feature maps: TIDL's tensor_bits 16 lost ~0pp on these models. It is not a "
        "parameter of this transform; set it in the target's toolchain if 8 bits are not enough.",
    ]
    if target == "tidl":
        caveats.append(
            "On TIDL, combine the recipe with TIDL's own mixed precision (advanced_options:"
            "mixed_precision_factor 1.2): MobileNetV3-Large -4.1pp -> -0.4pp. Do not expect 16-bit "
            "activations alone to fix a per-tensor weight collapse: 16-bit on Anneal's 8 most "
            "damaged activations left MobileNetV3-Small at -65.5pp (equalisation: -9.7pp).")
    silu = prof.activations.get("SiLU (Sigmoid*x)", 0) > 0
    if prof.family == "gated-depthwise":
        params = {**base, "equalize": True, "equalize_se": True, "equalize_mix": 0.5,
                  "cle": True, "cle_max_scale": 4}
        alts = [
            Candidate("per-tensor equalisation (old)", {**base, "equalize": True, "equalize_mix": 0.5},
                      "Gated sites only: no squeeze-excite sites, no CLE. The earlier recipe."),
            control,
        ]
        evidence = [
            "TIDL (TDA4VM emulator, Imagenette 1,500, paired): MobileNetV3-Small 8-bit -65.8pp, "
            "old equalisation -62.9pp, this recipe -9.7pp (+53.3pp vs old, p=8e-235).",
            "TIDL: MobileNetV3-Large -15.5pp -> -4.1pp.",
            "AMD A8W8 (per-tensor weights): EfficientNet-B0 -15.1pp (old equalisation) -> -2.8pp.",
        ]
        confidence = "medium"
        why = ("Equalise the gated and squeeze-excite chains with a 0.5 blend of cross-layer weight "
               "equalisation (weights share one scale per tensor), plus CLE capped at 4x across "
               "the ReLU pairs.")
        label = "per-tensor equalisation + SE sites + CLE cap 4"
        if target == "amd-xint8" and silu:
            alts.insert(0, Candidate(
                label + ", no surrogate", dict(params),
                "Without the HardSigmoid surrogate: what the float Sigmoid would give if the NPU kept it."))
            params["sigmoid_surrogate"] = 3
            label += " + sigmoid surrogate"
            why += (" Each Sigmoid becomes a fitted sum of 3 HardSigmoids, because AMD's NPU "
                    "replaces Sigmoid by HardSigmoid.")
            evidence.append("AMD's Sigmoid -> HardSigmoid swap in float: EfficientNet-B0 -48.8pp, "
                            "B1 -75.7pp; with sigmoid_surrogate 3: +0.2pp / -0.5pp.")
            evidence.append("AMD Quark XINT8 (Imagenette 1,000): EfficientNet-B0 -75.1pp -> -3.5pp with "
                            "percentile calibration + this recipe + sigmoid_surrogate 3; each part alone "
                            "stays near -75pp. EfficientNet-B1 -75.7 -> -50.1pp; with the gate ops at "
                            "16 bits -33.9pp (B0 -2.2pp); gate ops in float -28.1pp.")
            caveats.append(
                "AMD's NPU replaces Sigmoid by HardSigmoid, hence sigmoid_surrogate 3. In Quark, "
                "calibrate activations with CalibMethod.Percentile: its default MinMSE cost "
                "EfficientNet-B0 ~25pp.")
            caveats.append(
                "Deeper SiLU nets (EfficientNet-B1) stay far from FP32 at 8 bits: the gate branches "
                "cost most of it. Keep the gate ops at 16 bits in Quark (specific_layer_config with "
                "Int16Spec input/output tensors on the nodes named anneal_sur_* and "
                "anneal_eq_gate_mul_*): B1 -50.1pp -> -33.9pp. Do not add a formula-based bias "
                "correction on XINT8: its assumed weight rounding is not Quark's, and it cost 10pp.")
            confidence = "medium"
        rec = Candidate(label, params, why)
    elif prof.family == "cnn":
        rec = Candidate(
            "CLE capped at 4x", {**base, "cle": True, "cle_max_scale": 4},
            "Cross-layer weight equalisation so each per-tensor weight scale serves every channel, "
            "each pair's scale capped at 4x: through ReLU6 the exact rewrite adds unfused "
            "per-channel Min ceilings whose inputs are quantized before clipping, and large "
            "scales widen them.",
        )
        alts = [
            Candidate("CLE uncapped", {**base, "cle": True},
                      "Default cap (1000x): lost more than no CLE at all on MobileNetV2 (ReLU6)."),
            control,
        ]
        evidence = [
            "TIDL (TDA4VM emulator, Imagenette 1,500, paired): MobileNetV2 8-bit -10.7pp, "
            "this recipe -1.2pp (+9.5pp, p=5e-21).",
            "Same: uncapped CLE -16.7pp, cap 16 -12.9pp.",
        ]
        confidence = "medium" if target == "tidl" else "low"
        caveats.append("Measured on one CNN (MobileNetV2), on TIDL"
                       + ("." if target == "tidl" else "; unmeasured on AMD XINT8."))
    else:
        cpu = _advise_cpu(prof, int8_path)
        rec = _emulated(cpu.recommended)
        alts = [_emulated(c) for c in cpu.alternatives] + [control]
        evidence = list(cpu.evidence)
        confidence = "low"
        caveats.append(f"The {target} recipe is unvalidated for the {prof.family} family: this is the "
                       f"CPU recipe with {name}'s quantization flags added. Verify.")
        caveats.extend(c for c in cpu.caveats if "x86" not in c)
        if target == "amd-xint8" and silu:
            caveats.append("AMD's NPU replaces Sigmoid by HardSigmoid; consider sigmoid_surrogate 3 "
                           "(EfficientNet-B0 in float: -48.8pp swapped, +0.2pp with the surrogate).")
    return Advice(prof, int8_path, rec, alts, confidence, evidence, caveats, target=target)


# ---------------------------------------------------------------------------
# Measuring it
# ---------------------------------------------------------------------------


@dataclass
class VerifiedCandidate:
    candidate: Candidate
    accuracy: float | None
    delta_pp: float | None
    ci95_pp: tuple[float, float] | None
    mcnemar_p: float | None
    agreement: float | None
    p50_ms: float | None = None
    speedup: float | None = None
    path: str | None = None
    error: str | None = None


@dataclass
class Verification:
    mode: str  # "fused" (this CPU's kernels) or "emulated"
    n: int
    fp32_accuracy: float
    fp32_p50_ms: float | None
    rows: list[VerifiedCandidate]
    best: str | None
    notes: list[str]
    #: McNemar p-value between the best-measured recipe and the recommended one on the same
    #: images; the advice is contradicted only when this is below 0.05.
    best_vs_recommended_p: float | None = None

    @property
    def advice_contradicted(self) -> bool:
        return self.best_vs_recommended_p is not None and self.best_vs_recommended_p < 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "n": self.n, "fp32_accuracy": self.fp32_accuracy,
            "fp32_p50_ms": self.fp32_p50_ms, "best": self.best, "notes": self.notes,
            "best_vs_recommended_p": self.best_vs_recommended_p,
            "advice_contradicted": self.advice_contradicted,
            "rows": [{**{k: v for k, v in r.__dict__.items() if k != "candidate"},
                      "label": r.candidate.label, "params": r.candidate.params} for r in self.rows],
        }


def verification_mode(target_path: str, local_path: str) -> tuple[str, list[str]]:
    """Whether this machine's kernels can stand in for the target's.

    Accuracy under static INT8 depends on the CPU only through its accumulation: a
    32-bit-accumulating target is reproduced exactly by running the QDQ graph in float
    ("emulated"); a saturating target can only be reproduced on a saturating CPU.
    """
    if target_path == local_path and target_path != "unknown":
        return "fused", []
    if target_path in ("x86-vnni", "arm-dotprod", "unknown"):
        note = [] if local_path != "unknown" else ["this machine's INT8 path is unknown"]
        return "emulated", [f"Target {target_path} accumulates in 32 bits; scored by running the "
                            f"QDQ graph in float on this {local_path} machine."] + note
    return "emulated", [
        f"Target {target_path} saturates 16-bit pair sums but this machine ({local_path}) does "
        f"not: accuracy is scored emulated, which cannot show saturation. Run `anneal "
        f"saturation` on the chosen model, or verify on the target."
    ]


def verify(
    advice: Advice,
    model_path: Path,
    evalset,
    calibset,
    workdir: Path,
    *,
    local_path: str,
    time_target=None,
) -> Verification:
    """Build every candidate plus onnxruntime's default and score each against FP32."""
    import onnxruntime as ort

    from anneal.core.artifact import ModelArtifact
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.transforms import TransformContext, apply_transform

    if advice.target is not None:
        mode, notes = "emulated", [f"Target {advice.target}: scored by emulating its arithmetic "
                                   "(QDQ graph run in float) on this machine."]
    else:
        mode, notes = verification_mode(advice.int8_path, local_path)

    def predict(path: Path) -> np.ndarray:
        opts = ort.SessionOptions()
        if mode == "emulated":
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        s = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        return np.concatenate([np.asarray(evalset.decode(s.run(None, {name: x})[0]))
                               for x, _ in evalset.batches()])

    labels = np.concatenate([np.asarray(y) for _, y in evalset.batches()])
    fp = predict(model_path)
    fp_right = fp == labels
    n = len(labels)
    bench = None
    fp32_ms = None
    if time_target is not None and mode == "fused":
        from anneal.core.measure import Benchmarker

        bench = Benchmarker(time_target, warmup=10, runs=50)
        fp32_ms = bench.measure(ModelArtifact(path=model_path)).latency_ms_p50

    ctx = TransformContext(workdir=workdir, calibset=calibset)
    rows: list[VerifiedCandidate] = []
    preds: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    for cand in advice.candidates():
        key = repr(sorted(cand.params.items()))
        if key in seen:
            continue
        seen.add(key)
        try:
            art = apply_transform("quantize_static_int8", dict(cand.params), ModelArtifact(path=model_path), ctx)
            pred = predict(art.path)
        except Exception as exc:  # a recipe that fails to build is a result, not a crash
            rows.append(VerifiedCandidate(cand, None, None, None, None, None, error=f"{type(exc).__name__}: {exc}"[:300]))
            continue
        preds[cand.label] = pred
        right = pred == labels
        b = int(np.sum(fp_right & ~right))
        c = int(np.sum(~fp_right & right))
        delta, lo, hi = paired_delta_ci(b, c, n)
        row = VerifiedCandidate(cand, float(right.mean()), delta, (lo, hi), mcnemar_exact(b, c),
                                float((pred == fp).mean()), path=str(art.path))
        if bench is not None:
            row.p50_ms = bench.measure(art).latency_ms_p50
            row.speedup = fp32_ms / row.p50_ms if row.p50_ms else None
        rows.append(row)

    scored = [r for r in rows if r.accuracy is not None]
    best = max(scored, key=lambda r: (r.accuracy, r.agreement or 0.0)).candidate.label if scored else None
    p_best = None
    rec = advice.recommended.label
    if best is not None and best != rec and rec in preds:
        # Paired on the same images: which one gets right what the other gets wrong.
        rec_right, best_right = preds[rec] == labels, preds[best] == labels
        p_best = mcnemar_exact(int(np.sum(rec_right & ~best_right)), int(np.sum(~rec_right & best_right)))
    elif best == rec:
        p_best = 1.0
    return Verification(mode, n, float(fp_right.mean()), fp32_ms, rows, best, notes, p_best)
