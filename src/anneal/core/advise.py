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
                        MobileNetV3). Per-tensor activation scales collapse these on every
                        CPU; equalisation + asymmetric percentile + float stem fixes them.
  ``cnn``               everything else with convolutions (ResNet, RegNet, MobileNetV2,
                        ShuffleNet, MnasNet). Fine on 32-bit hardware; on x86 without VNNI
                        the stem saturates, which keeping it in float removes. Symmetric
                        99.999 percentile; reduce_range and a 99.99 percentile both cost
                        ~0.5pp on ImageNet.

* **the INT8 path** (:func:`anneal.core.environment.cpu_features`): ``x86-avx2-16bit`` is the
  one whose 16-bit pair sums saturate; ``x86-vnni`` and ``arm-dotprod`` accumulate in 32 bits.

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

    def candidates(self) -> list[Candidate]:
        return [self.recommended, *self.alternatives]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "int8_path": self.int8_path,
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

    if lns and depthwise:
        family = "convnext"
    elif lns and matmuls > convs:
        family = "transformer"
    elif gated:
        family = "gated-depthwise"
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


def advise(model_path: Path, int8_path: str) -> Advice:
    if int8_path not in INT8_PATHS:
        raise ValueError(f"int8_path must be one of {INT8_PATHS}, got {int8_path!r}")
    prof = profile(model_path)
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
        # Measured on AC power, cpu-1t, interleaved rounds: the recipes that keep accuracy
        # are ~5% faster than FP32 here; the fast default is fast only because it quantizes
        # the stem, which is what breaks it.
        caveats.append(
            "On x86 without VNNI the accurate recipes were only ~1.05x faster than FP32 "
            "(EfficientNet-B0 1.04x, ResNet-50 1.05x); the default's 1.12x comes from the "
            "quantized stem that costs 13-51pp. INT8 may not be worth deploying on this CPU; "
            "the same recipes gave 2-4x on ARM."
        )
    return Advice(prof, int8_path, rec, alts, confidence, evidence, caveats)


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
