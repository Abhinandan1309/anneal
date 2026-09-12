"""The action space: transforms the agent can apply to a model.

Each transform is a pure function from (artifact, params) to a new artifact on disk. They
compose, and the composition is recorded in the artifact's lineage, so any point on the
final Pareto frontier can be reproduced from its recipe alone.

The interesting member of this family is :func:`quantize_dynamic_sensitive`. Blanket INT8
quantization is a blunt instrument — a handful of layers are typically responsible for
most of the accuracy loss, and excluding just those recovers most of the accuracy while
keeping most of the speed. Anneal ranks layers by weight-quantization error and lets the
agent choose how many to spare, turning "quantize or don't" into a tunable dial.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from anneal.core.artifact import ModelArtifact, TransformRecord
from anneal.core.measure import EvalSet

#: ONNX op types whose weights dominate compute in a CNN/transformer.
QUANTIZABLE_OPS = ("Conv", "Gemm", "MatMul", "ConvTranspose")


@dataclass
class TransformContext:
    """Everything a transform may need that is not the model itself."""

    workdir: Path
    evalset: EvalSet | None = None
    calib_samples: int = 64
    extra: dict[str, Any] = field(default_factory=dict)

    def path_for(self, artifact: ModelArtifact, record: TransformRecord) -> Path:
        """Deterministic output path derived from the full recipe."""
        recipe = f"{artifact.lineage_key}|{record}"
        digest = hashlib.sha1(recipe.encode()).hexdigest()[:10]
        stem = _slug(str(record))
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.workdir / f"{stem}-{digest}.onnx"


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() else "-" for c in text)
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-")[:60] or "model"


class TransformError(RuntimeError):
    """A transform could not be applied (bad params, unsupported graph, missing dep)."""


@dataclass(frozen=True)
class TransformSpec:
    """Declarative description of a transform — also the source of the agent's tool schema."""

    name: str
    summary: str
    params: dict[str, dict[str, Any]]
    fn: Callable[[ModelArtifact, dict[str, Any], TransformContext], ModelArtifact]
    #: Applying this more than once in a chain is meaningless or harmful.
    idempotent: bool = True
    requires: tuple[str, ...] = ()

    def available(self) -> tuple[bool, str]:
        for module in self.requires:
            try:
                __import__(module)
            except ImportError:
                return False, f"requires the {module!r} package, which is not installed"
        return True, ""

    def json_schema(self) -> dict[str, Any]:
        props = {}
        for pname, meta in self.params.items():
            entry = {"type": meta["type"], "description": meta["description"]}
            if "enum" in meta:
                entry["enum"] = meta["enum"]
            props[pname] = entry
        return {"type": "object", "properties": props, "required": []}


# ---------------------------------------------------------------------------
# Weight sensitivity — the analysis that makes selective quantization possible
# ---------------------------------------------------------------------------


def weight_quantization_error(w: np.ndarray, per_channel: bool = True) -> float:
    """Relative L2 error introduced by symmetric INT8 quantization of this tensor.

    This is a *proxy* for how much a layer will suffer under INT8, not a measurement of
    end-to-end accuracy loss — it ignores activation range and error propagation. It is
    cheap (no inference required) and empirically ranks layers well enough to be a useful
    prior for the agent's search. Anneal never reports it as an accuracy number.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.size == 0:
        return 0.0

    if per_channel and w.ndim > 1:
        flat = w.reshape(w.shape[0], -1)
        scale = np.abs(flat).max(axis=1, keepdims=True) / 127.0
    else:
        flat = w.reshape(1, -1)
        scale = np.abs(flat).max() / 127.0
        scale = np.array([[scale]], dtype=np.float32)

    scale = np.where(scale == 0, 1.0, scale)
    dequant = np.clip(np.round(flat / scale), -127, 127) * scale
    num = np.linalg.norm(flat - dequant)
    den = np.linalg.norm(flat)
    return float(num / den) if den > 0 else 0.0


def rank_layer_sensitivity(
    model_path: Path, per_channel: bool = True
) -> list[tuple[str, float, str]]:
    """Rank quantizable nodes by weight-quantization error, most sensitive first.

    Returns ``(node_name, relative_error, op_type)``.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(model_path))
    inits = {i.name: i for i in model.graph.initializer}

    ranked: list[tuple[str, float, str]] = []
    for node in model.graph.node:
        if node.op_type not in QUANTIZABLE_OPS:
            continue
        weight_name = next((inp for inp in node.input[1:] if inp in inits), None)
        if weight_name is None:
            continue
        w = numpy_helper.to_array(inits[weight_name])
        err = weight_quantization_error(w, per_channel=per_channel)
        name = node.name or f"{node.op_type}_{weight_name}"
        ranked.append((name, err, node.op_type))

    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class _EvalSetCalibrationReader:
    """Feeds real images to onnxruntime's static-quantization calibrator.

    Calibrating on random noise is the classic way to produce a quantized model with
    plausible-looking activation ranges and terrible accuracy. This uses the same
    preprocessing as evaluation.
    """

    def __init__(self, evalset: EvalSet, input_name: str, limit: int) -> None:
        self._batches: list[dict[str, np.ndarray]] = [
            {input_name: np.ascontiguousarray(x, dtype=np.float32)}
            for x in evalset.calibration_batches(limit)
        ]
        if not self._batches:
            raise TransformError("calibration requested but the eval set yielded no batches")
        self._iter: Iterator[dict[str, np.ndarray]] = iter(self._batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = iter(self._batches)


def _input_name(model_path: Path) -> str:
    from anneal.core.artifact import describe_io

    io = describe_io(model_path)
    if not io["inputs"]:
        raise TransformError(f"{model_path.name} has no graph inputs")
    return io["inputs"][0]["name"]


def _preprocessed(artifact: ModelArtifact, ctx: TransformContext) -> Path:
    """Run ORT's quantization pre-processing (shape inference + symbolic shapes).

    Skipping this is the most common cause of 'quantization produced a model that is
    somehow slower', because unresolved shapes block kernel fusion downstream.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = ctx.workdir / f"{artifact.path.stem}-preproc.onnx"
    if out.exists():
        return out
    ctx.workdir.mkdir(parents=True, exist_ok=True)
    try:
        quant_pre_process(str(artifact.path), str(out), skip_symbolic_shape=False)
    except Exception:
        # Pre-processing is best-effort; a graph it cannot handle can still be quantized.
        shutil.copyfile(artifact.path, out)
    return out


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def graph_optimize(
    artifact: ModelArtifact, params: dict[str, Any], ctx: TransformContext
) -> ModelArtifact:
    """Bake onnxruntime's graph optimisations (fusion, constant folding) into the file.

    At ``level='all'`` onnxruntime runs the NCHWc transformer, which rewrites convolution
    layouts for *the CPU doing the optimising* — including its specific SIMD width. The
    resulting file is not portable: shipping it to a different machine can be slower than
    the original, or fail outright. Anneal records that on the artifact so the report can
    warn rather than letting a fast local number turn into a production surprise.
    """
    import onnxruntime as ort

    level = params.get("level", "all")
    levels = {
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if level not in levels:
        raise TransformError(f"level must be one of {sorted(levels)}, got {level!r}")

    record = TransformRecord("graph_optimize", {"level": level})
    out = ctx.path_for(artifact, record)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = levels[level]
    opts.optimized_model_filepath = str(out)
    ort.InferenceSession(str(artifact.path), sess_options=opts, providers=["CPUExecutionProvider"])

    if not out.exists():
        raise TransformError("onnxruntime did not emit an optimised model")

    extra: dict[str, Any] = {}
    if level == "all":
        extra["portability_warning"] = (
            "level='all' bakes in NCHWc layout transforms specific to the CPU that ran "
            "the optimisation; this artifact is only valid on matching hardware. Use "
            "level='extended' for a portable file."
        )
    return artifact.derive(record, out, **extra)


def quantize_dynamic_int8(
    artifact: ModelArtifact, params: dict[str, Any], ctx: TransformContext
) -> ModelArtifact:
    """INT8 weights, activations quantized on the fly. No calibration data needed."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    per_channel = bool(params.get("per_channel", True))
    reduce_range = bool(params.get("reduce_range", False))
    weight_type = params.get("weight_type", "int8")

    record = TransformRecord(
        "quantize_dynamic_int8",
        {"per_channel": per_channel, "reduce_range": reduce_range, "weight_type": weight_type},
    )
    out = ctx.path_for(artifact, record)
    src = _preprocessed(artifact, ctx)

    quantize_dynamic(
        model_input=str(src),
        model_output=str(out),
        weight_type=QuantType.QInt8 if weight_type == "int8" else QuantType.QUInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
    )
    return artifact.derive(record, out)


def quantize_dynamic_sensitive(
    artifact: ModelArtifact, params: dict[str, Any], ctx: TransformContext
) -> ModelArtifact:
    """INT8 dynamic quantization that *spares* the k most quantization-sensitive layers.

    ``skip_top_k`` layers with the highest weight-quantization error stay in FP32. This
    is the accuracy/latency dial: k=0 is blanket quantization, larger k trades speed back
    for fidelity. ``skip_first_last`` additionally spares the stem convolution and the
    classifier, which are sensitive in almost every vision backbone.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    skip_top_k = int(params.get("skip_top_k", 2))
    skip_first_last = bool(params.get("skip_first_last", False))
    per_channel = bool(params.get("per_channel", True))

    if skip_top_k < 0:
        raise TransformError("skip_top_k must be >= 0")

    src = _preprocessed(artifact, ctx)
    ranked = rank_layer_sensitivity(src, per_channel=per_channel)
    if not ranked:
        raise TransformError("no quantizable Conv/Gemm/MatMul nodes found in this graph")

    exclude = [name for name, _, _ in ranked[:skip_top_k]]

    if skip_first_last:
        import onnx

        model = onnx.load(str(src))
        quant_nodes = [n for n in model.graph.node if n.op_type in QUANTIZABLE_OPS]
        for node in (quant_nodes[0], quant_nodes[-1]) if quant_nodes else ():
            name = node.name or ""
            if name and name not in exclude:
                exclude.append(name)

    record = TransformRecord(
        "quantize_dynamic_sensitive",
        {
            "skip_top_k": skip_top_k,
            "skip_first_last": skip_first_last,
            "per_channel": per_channel,
        },
    )
    out = ctx.path_for(artifact, record)

    quantize_dynamic(
        model_input=str(src),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        nodes_to_exclude=exclude,
    )

    return artifact.derive(
        record,
        out,
        excluded_nodes=exclude,
        sensitivity_top5=[{"node": n, "rel_err": round(e, 5), "op": o} for n, e, o in ranked[:5]],
    )


def quantize_static_int8(
    artifact: ModelArtifact, params: dict[str, Any], ctx: TransformContext
) -> ModelArtifact:
    """Full INT8 (weights *and* activations) using real calibration data.

    Defaults to U8S8 — unsigned activations, signed weights — which is the combination
    x86 VNNI kernels are built for. ``reduce_range`` trades a bit of precision for
    overflow safety on pre-VNNI AVX2 machines.
    """
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    if ctx.evalset is None:
        raise TransformError("static quantization needs calibration data; no eval set supplied")

    per_channel = bool(params.get("per_channel", True))
    reduce_range = bool(params.get("reduce_range", False))
    calib_method = params.get("calibrate_method", "minmax")
    n_calib = int(params.get("calib_samples", ctx.calib_samples))

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    if calib_method not in methods:
        raise TransformError(f"calibrate_method must be one of {sorted(methods)}")

    record = TransformRecord(
        "quantize_static_int8",
        {
            "per_channel": per_channel,
            "reduce_range": reduce_range,
            "calibrate_method": calib_method,
            "calib_samples": n_calib,
        },
    )
    out = ctx.path_for(artifact, record)
    src = _preprocessed(artifact, ctx)

    reader = _EvalSetCalibrationReader(ctx.evalset, _input_name(src), n_calib)
    quantize_static(
        model_input=str(src),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
        calibrate_method=methods[calib_method],
    )
    return artifact.derive(record, out, calib_samples=n_calib)


def cast_fp16(
    artifact: ModelArtifact, params: dict[str, Any], ctx: TransformContext
) -> ModelArtifact:
    """Cast the graph to FP16. Halves size; a win on GPU, usually a loss on CPU."""
    from onnxconverter_common import float16
    import onnx

    keep_io_fp32 = bool(params.get("keep_io_fp32", True))
    record = TransformRecord("cast_fp16", {"keep_io_fp32": keep_io_fp32})
    out = ctx.path_for(artifact, record)

    model = onnx.load(str(artifact.path))
    converted = float16.convert_float_to_float16(
        model, keep_io_types=keep_io_fp32, disable_shape_infer=True
    )
    onnx.save(converted, str(out))
    return artifact.derive(record, out)


REGISTRY: dict[str, TransformSpec] = {
    "graph_optimize": TransformSpec(
        name="graph_optimize",
        summary=(
            "Bake onnxruntime's fusions and constant folding into the model file. Cheap, "
            "lossless, and often a prerequisite for quantization paying off. Note that "
            "level='all' produces a file tied to the optimising CPU's layout and SIMD "
            "width; level='extended' stays portable."
        ),
        params={
            "level": {
                "type": "string",
                "enum": ["basic", "extended", "all"],
                "description": "Optimisation aggressiveness. 'all' includes layout opts.",
            }
        },
        fn=graph_optimize,
    ),
    "quantize_dynamic_int8": TransformSpec(
        name="quantize_dynamic_int8",
        summary=(
            "INT8 weights with activations quantized at runtime. No calibration data "
            "required. Strong size win; latency win depends on whether the kernels for "
            "this op mix are actually INT8-accelerated on the target."
        ),
        params={
            "per_channel": {
                "type": "boolean",
                "description": "Per-output-channel weight scales. Better accuracy, slight overhead.",
            },
            "reduce_range": {
                "type": "boolean",
                "description": "Use 7-bit weight range to avoid overflow on pre-VNNI AVX2 CPUs.",
            },
            "weight_type": {
                "type": "string",
                "enum": ["int8", "uint8"],
                "description": "Signed or unsigned INT8 weights.",
            },
        },
        fn=quantize_dynamic_int8,
    ),
    "quantize_dynamic_sensitive": TransformSpec(
        name="quantize_dynamic_sensitive",
        summary=(
            "INT8 dynamic quantization that leaves the k most quantization-sensitive "
            "layers in FP32, ranked by weight-quantization error. The main dial for "
            "trading a little speed back for accuracy."
        ),
        params={
            "skip_top_k": {
                "type": "integer",
                "description": "How many of the most sensitive layers to leave in FP32 (0 = none).",
            },
            "skip_first_last": {
                "type": "boolean",
                "description": "Also spare the stem conv and the classifier layer.",
            },
            "per_channel": {
                "type": "boolean",
                "description": "Per-output-channel weight scales.",
            },
        },
        fn=quantize_dynamic_sensitive,
    ),
    "quantize_static_int8": TransformSpec(
        name="quantize_static_int8",
        summary=(
            "Full INT8 (weights and activations) calibrated on real data, emitted as QDQ. "
            "The biggest latency win when the target has INT8 kernels, and the most "
            "accuracy risk. Needs an eval set for calibration."
        ),
        params={
            "per_channel": {"type": "boolean", "description": "Per-output-channel weight scales."},
            "reduce_range": {"type": "boolean", "description": "7-bit weights for AVX2 safety."},
            "calibrate_method": {
                "type": "string",
                "enum": ["minmax", "entropy", "percentile"],
                "description": "How activation ranges are estimated. Entropy is slower, often kinder to outliers.",
            },
            "calib_samples": {
                "type": "integer",
                "description": "Number of calibration images to use.",
            },
        },
        fn=quantize_static_int8,
    ),
    "cast_fp16": TransformSpec(
        name="cast_fp16",
        summary=(
            "Cast the graph to FP16. Halves model size. Typically a win on GPU/NPU and a "
            "regression on CPUs without native FP16 arithmetic."
        ),
        params={
            "keep_io_fp32": {
                "type": "boolean",
                "description": "Keep graph inputs/outputs FP32 so callers need no changes.",
            }
        },
        fn=cast_fp16,
        requires=("onnxconverter_common",),
    ),
}


def available_transforms() -> dict[str, TransformSpec]:
    """Transforms whose dependencies are actually installed here."""
    return {name: spec for name, spec in REGISTRY.items() if spec.available()[0]}


def apply_transform(
    name: str, params: dict[str, Any], artifact: ModelArtifact, ctx: TransformContext
) -> ModelArtifact:
    """Apply one transform by name, with validation."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise TransformError(f"unknown transform {name!r}; known: {sorted(REGISTRY)}")
    ok, why = spec.available()
    if not ok:
        raise TransformError(f"transform {name!r} unavailable: {why}")

    unknown = set(params) - set(spec.params)
    if unknown:
        raise TransformError(
            f"transform {name!r} got unknown parameter(s) {sorted(unknown)}; "
            f"accepts {sorted(spec.params)}"
        )

    if spec.idempotent and any(t.name == name for t in artifact.lineage):
        raise TransformError(f"{name!r} is already in this model's lineage; applying it twice is a no-op")

    return spec.fn(artifact, params, ctx)
