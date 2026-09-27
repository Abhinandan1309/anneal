"""The action space: transforms Anneal can apply to a model.

Each transform is a pure function from (artifact, params) to a new artifact on disk. They
compose, and the composition is recorded in the artifact's lineage, so any point on the
final Pareto frontier can be reproduced from its recipe alone.

The interesting member of this family is :func:`quantize_dynamic_sensitive`. Blanket INT8
quantization is a blunt instrument — a handful of layers are typically responsible for
most of the accuracy loss, and excluding just those recovers most of the accuracy while
keeping most of the speed. Anneal ranks layers by weight-quantization error and lets the
search choose how many to spare, turning "quantize or don't" into a tunable dial.
"""

from __future__ import annotations

import contextlib

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from anneal.core.artifact import ModelArtifact, TransformRecord
from anneal.core.measure import EvalSet

#: Fraction of a layer's accumulators that may be corrupted by 16-bit saturation before the
#: saturation guard keeps that layer in FP32. On ResNet-18 the stem saturated 18-22% under
#: full-range per-channel weights (and broke accuracy) but ~1% under per-tensor (harmless).
SATURATION_TOLERANCE = 0.02

#: How far (as a fraction of its range) equalisation may extend the bottom of an activation
#: tensor's quantization range. On EfficientNet-B0 anything from 0.1 to 1.0 performed alike.
EQUALIZE_SLACK = 0.1

#: Default ``equalize_mix`` when weights are quantized per tensor: the share t of cross-layer
#: weight equalisation blended into the activation equalisation scale. MobileNetV3-Small with
#: per-tensor weights on onnxruntime: -63.5pp without equalisation, -8.6pp at t=0, -7.2pp at
#: t=0.5, -11.1pp at t=1 (plain CLE).
EQUALIZE_PER_TENSOR_MIX = 0.5

#: int16_top_k ranks tensors by the predictions they flip on this many calibration images
#: (the probe is taken from the calibration set, never the eval set, so nothing leaks).
INT16_PROBE_IMAGES = 64

#: ONNX op types whose weights dominate compute in a CNN/transformer.
QUANTIZABLE_OPS = ("Conv", "Gemm", "MatMul", "ConvTranspose")


@dataclass
class TransformContext:
    """Everything a transform may need that is not the model itself."""

    workdir: Path
    evalset: EvalSet | None = None
    #: Where static quantization draws calibration inputs. Should be disjoint from
    #: ``evalset``: calibrating on the images you then score on fits activation ranges to
    #: the test data and flatters the accuracy that gets reported.
    calibset: EvalSet | None = None
    calib_samples: int = 64
    extra: dict[str, Any] = field(default_factory=dict)

    def path_for(self, artifact: ModelArtifact, record: TransformRecord) -> Path:
        """Deterministic output path derived from the source model and the full recipe.

        The source path is part of the key: two different models given the same recipe in one
        workdir must not share an output file (a derived artifact's path already encodes its
        own lineage).
        """
        recipe = f"{artifact.path.resolve()}|{artifact.lineage_key}|{record}"
        digest = hashlib.sha1(recipe.encode()).hexdigest()[:10]
        stem = _slug(str(record))
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.workdir / f"{stem}-{digest}.onnx"


#: Concatenated inputs whose ranges differ by this factor cannot share one 8-bit scale: the
#: smallest keeps fewer than ~13 of 256 levels.
MIXED_OUTPUT_RATIO = 20.0
MIXED_CHECK_IMAGES = 16
_SHAPE_OPS = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity", "Flatten", "Cast"}
_COMPUTE_OPS = {"Conv", "Gemm", "MatMul", "ConvTranspose"}


def _output_concat(model, output: str):
    """The Concat an output is made of, looking through shape-only ops; None if there is none."""
    producer = {o: n for n in model.graph.node for o in n.output}
    node = producer.get(output)
    while node is not None and node.op_type in _SHAPE_OPS:
        node = producer.get(node.input[0])
    return node if node is not None and node.op_type == "Concat" else None


def outputs_fed_by_concat(model_path: Path) -> bool:
    import onnx

    m = onnx.load(str(model_path))
    return any(_output_concat(m, o.name) is not None for o in m.graph.output)


def mixed_range_outputs(model_path: Path, batches, ratio: float = MIXED_OUTPUT_RATIO) -> list[dict[str, Any]]:
    """Outputs concatenating tensors whose ranges differ by ``ratio`` or more, with their tails.

    The tail is every node on a path from the output back to the nearest compute op
    (Conv/Gemm/MatMul), which it does not include: keeping it float leaves every compute op INT8.
    """
    import onnx
    import onnxruntime as ort
    from onnx import helper

    model = onnx.load(str(model_path))
    inits = {i.name for i in model.graph.initializer}
    producer = {o: n for n in model.graph.node for o in n.output}
    found = []
    for out in model.graph.output:
        cat = _output_concat(model, out.name)
        if cat is not None:
            found.append((out.name, cat))
    if not found:
        return []
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    known = {o.name for o in probe.graph.output}
    wanted = sorted({i for _, cat in found for i in cat.input if i not in inits})
    probe.graph.output.extend([helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
                               for t in wanted if t not in known])
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.enable_cpu_mem_arena = False
    session = ort.InferenceSession(probe.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    lo = dict.fromkeys(wanted, 0.0)
    hi = dict.fromkeys(wanted, 0.0)
    for batch in batches:
        for t, v in zip(wanted, session.run(wanted, {input_name: batch}), strict=True):
            lo[t], hi[t] = min(lo[t], float(v.min())), max(hi[t], float(v.max()))
    flagged = []
    for name, cat in found:
        spans = [hi[i] - lo[i] for i in cat.input if i in hi and hi[i] - lo[i] > 0]
        if len(spans) < 2 or max(spans) / min(spans) < ratio:
            continue
        tail, stack = [], [name]
        while stack:
            node = producer.get(stack.pop())
            if node is None or node.op_type in _COMPUTE_OPS or node.name in tail:
                continue
            tail.append(node.name)
            stack += [i for i in node.input if i not in inits]
        flagged.append({"output": name, "concat": cat.name, "range_ratio": round(max(spans) / min(spans), 1),
                        "tail_nodes": tail})
    return flagged


def round_activation_scales_pow2(quantized: Path) -> int:
    """Round every activation scale of a QDQ model up to a power of two, in place; the count.

    Activation Q/DQ pairs are those whose QuantizeLinear quantizes a computed tensor (weights
    are stored pre-quantized with a DequantizeLinear only). Rounding up keeps the range covered.
    """
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(quantized))
    inits = {i.name: i for i in m.graph.initializer}
    scales = {n.input[1] for n in m.graph.node
              if n.op_type == "QuantizeLinear" and n.input[0] not in inits and n.input[1] in inits}
    for name in scales:
        v = numpy_helper.to_array(inits[name]).astype(np.float64)
        v = np.power(2.0, np.ceil(np.log2(np.maximum(v, np.finfo(np.float32).tiny))))
        inits[name].CopyFrom(numpy_helper.from_array(v.astype(np.float32), name))
    onnx.save(m, str(quantized))
    return len(scales)


def concat_groups(quantized: Path) -> list[tuple[list[str], tuple[float, float]]]:
    """Each Concat of a QDQ model: its float input and output tensor names and their union range.

    Ranges come from the QuantizeLinear nodes' scale and zero point (uint8 or int8 grids)."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(quantized))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    producer = {o: n for n in m.graph.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in m.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def q_range(q) -> tuple[float, float] | None:
        if len(q.input) < 3 or q.input[1] not in inits or q.input[2] not in inits:
            return None
        scale, zp = inits[q.input[1]], inits[q.input[2]]
        if np.size(scale) != 1:
            return None
        qmin, qmax = (0, 255) if zp.dtype == np.uint8 else (-128, 127) if zp.dtype == np.int8 else (0, 65535)
        return float((qmin - int(zp)) * float(scale)), float((qmax - int(zp)) * float(scale))

    groups = []
    for n in m.graph.node:
        if n.op_type != "Concat":
            continue
        names, ranges = [], []
        for i in n.input:  # Concat <- DequantizeLinear <- QuantizeLinear(float tensor)
            dq = producer.get(i)
            q = producer.get(dq.input[0]) if dq is not None and dq.op_type == "DequantizeLinear" else None
            if q is None or q.op_type != "QuantizeLinear" or q_range(q) is None:
                continue
            names.append(q.input[0])
            ranges.append(q_range(q))
        outs = [c for c in consumers.get(n.output[0], []) if c.op_type == "QuantizeLinear"]
        if outs and q_range(outs[0]) is not None:
            names.append(n.output[0])
            ranges.append(q_range(outs[0]))
        if len(names) >= 2:
            groups.append((names, (min(r[0] for r in ranges), max(r[1] for r in ranges))))
    return groups


def mean_minmax_ranges(model_path: Path, batches) -> dict[str, tuple[float, float]]:
    """Per activation tensor: the mean over images of each image's min and max (0 included).

    Every float tensor a node produces is measured (initializers and constants are weights, not
    activations). This is NNCF's MEAN_MINMAX estimator with one image per sample.
    """
    import onnx
    import onnxruntime as ort
    from onnx import helper

    model = onnx.load(str(model_path))
    consts = {i.name for i in model.graph.initializer}
    consts |= {o for n in model.graph.node if n.op_type == "Constant" for o in n.output}
    inferred = onnx.shape_inference.infer_shapes(model)
    float_t = {v.name for v in list(inferred.graph.value_info) + list(inferred.graph.output)
               if v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT}
    float_t |= {v.name for v in model.graph.input if v.name not in consts}
    tensors = [o for n in model.graph.node for o in n.output if o in float_t and o not in consts]
    tensors += [v.name for v in model.graph.input if v.name not in consts]
    tensors = list(dict.fromkeys(tensors))
    known_out = {o.name for o in model.graph.output}
    model.graph.output.extend([helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
                               for t in tensors if t not in known_out])
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.enable_cpu_mem_arena = False
    session = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    lo_sum = dict.fromkeys(tensors, 0.0)
    hi_sum = dict.fromkeys(tensors, 0.0)
    count = 0
    for batch in batches:
        for x in batch:  # one image at a time: per-image extremes
            values = session.run(tensors, {input_name: x[None]})
            for t, v in zip(tensors, values, strict=True):
                if v.size:
                    lo_sum[t] += min(float(v.min()), 0.0)
                    hi_sum[t] += max(float(v.max()), 0.0)
            count += 1
    if not count:
        raise TransformError("mean_minmax calibration needs calibration images; none were given")
    return {t: (lo_sum[t] / count, hi_sum[t] / count) for t in tensors}


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() else "-" for c in text)
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-")[:60] or "model"


class TransformError(RuntimeError):
    """A transform could not be applied (bad params, unsupported graph, missing dep)."""


@dataclass(frozen=True)
class TransformSpec:
    """Declarative description of a transform — also the source of the search's parameter schema."""

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
    prior for the search. Anneal never reports it as an accuracy number.
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

#: onnxruntime's quantize_static cannot pass histogram sizes to its entropy calibrator, which
#: then defaults to 128 bins *and* 128 quantized bins: the KL search has nothing to search and
#: returns the min/max range, so "entropy" silently equals "minmax". TensorRT's KL calibration
#: uses 2048 bins folded to 128; we inject the same.
ENTROPY_NUM_BINS, ENTROPY_NUM_QUANTIZED_BINS = 2048, 128


@contextlib.contextmanager
def _entropy_bins(active: bool):
    if not active:
        yield
        return
    import importlib

    # The package re-exports a function named `quantize`, so import the module by path.
    ort_quantize = importlib.import_module("onnxruntime.quantization.quantize")

    original = ort_quantize.create_calibrator

    def create_calibrator(*args, **kwargs):
        extra = dict(kwargs.get("extra_options") or {})
        extra.setdefault("num_bins", ENTROPY_NUM_BINS)
        extra.setdefault("num_quantized_bins", ENTROPY_NUM_QUANTIZED_BINS)
        kwargs["extra_options"] = extra
        return original(*args, **kwargs)

    ort_quantize.create_calibrator = create_calibrator
    try:
        yield
    finally:
        ort_quantize.create_calibrator = original


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

    # onnxruntime's strided calibration (CalibStridedMinMax) feeds one range of batches at a time.
    def __len__(self) -> int:
        return len(self._batches)

    def set_range(self, start_index: int, end_index: int) -> None:
        self._iter = iter(self._batches[start_index:end_index])


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

    # Keyed on the source's bytes, not its name: two models called model.onnx in different
    # folders, or a source rewritten in place, must never share a cached pre-processed file.
    out = ctx.workdir / f"{artifact.path.stem}-{artifact.content_hash()[:12]}-preproc.onnx"
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

    ``skip_top_k`` layers stay in FP32. This is the accuracy/latency dial: k=0 is blanket
    quantization, larger k trades speed back for fidelity. ``skip_first_last`` additionally
    spares the stem convolution and the classifier.

    ``ranking`` decides what "most sensitive" means:

    ``measured`` (the default whenever calibration data is available)
        Quantize each layer alone, run the eval set, and rank by how many predictions
        changed. Costs one eval pass per layer, once per base model per run — then cached.
    ``proxy``
        Rank by weight-quantization error. Free, and on ResNet-18 only weakly predictive
        (Spearman +0.33 against the measured sweep): it ranks the stem convolution, the
        most damaging layer to quantize, last of 21, because stem sensitivity lives in the
        raw-pixel activations and the proxy only looks at weights. Kept for when there is
        no eval set, and so the two can be compared.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    skip_top_k = int(params.get("skip_top_k", 2))
    skip_first_last = bool(params.get("skip_first_last", False))
    per_channel = bool(params.get("per_channel", True))
    ranking = params.get("ranking") or (
        "measured" if (ctx.evalset is not None or ctx.extra.get("measured_ranking")) else "proxy"
    )

    if skip_top_k < 0:
        raise TransformError("skip_top_k must be >= 0")
    if ranking not in ("measured", "proxy"):
        raise TransformError(f"ranking must be 'measured' or 'proxy', got {ranking!r}")

    src = _preprocessed(artifact, ctx)
    ranked = rank_layer_sensitivity(src, per_channel=per_channel)
    if not ranked:
        raise TransformError("no quantizable Conv/Gemm/MatMul nodes found in this graph")

    if ranking == "measured":
        order = _measured_order(src, ctx, per_channel, {name for name, _, _ in ranked})
    else:
        order = [name for name, _, _ in ranked]

    exclude = order[:skip_top_k]

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
            # Recorded resolved, not as passed: two runs that defaulted differently must
            # not share a lineage key.
            "ranking": ranking,
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
        sensitivity_ranking=ranking,
        sensitivity_top5=[{"node": n, "rel_err": round(e, 5), "op": o} for n, e, o in ranked[:5]],
    )


def _measured_order(
    src: Path, ctx: TransformContext, per_channel: bool, graph_nodes: set[str]
) -> list[str]:
    """Layers ordered most-damaging-first by the measured one-layer-at-a-time sweep.

    Resolution order: a ranking seeded into ``ctx.extra['measured_ranking']`` (e.g. from
    a saved ``anneal sensitivity --measured`` run), then one cached earlier in this run
    for the same graph, then a fresh sweep. The sweep is expensive — one eval pass per
    layer — so it runs at most once per (graph, per_channel) per run.
    """
    seeded = ctx.extra.get("measured_ranking")
    if seeded:
        order = [n for n in seeded if n in graph_nodes]
        if not order:
            raise TransformError(
                "the supplied measured sensitivity ranking shares no node names with this "
                "graph; it was probably produced for a different model"
            )
        # Layers the saved sweep never scored go last, least-known-first is not a ranking.
        return order + sorted(graph_nodes - set(order))

    cache: dict[tuple[str, bool], list[str]] = ctx.extra.setdefault("_measured_cache", {})
    key = (str(src), per_channel)
    if key in cache:
        return cache[key]

    if ctx.evalset is None:
        raise TransformError(
            "ranking='measured' needs an eval set to measure against; pass one, supply a "
            "saved sweep with --sensitivity, or use ranking='proxy'"
        )

    from anneal.core.sensitivity import measured_sensitivity
    from anneal.core.targets import default_target

    # Accuracy is a property of the graph, not the runtime, so any working CPU target
    # scores it correctly; the fastest available one is used.
    results = measured_sensitivity(
        ModelArtifact(path=src),
        ctx.extra.get("scoring_target") or default_target(),
        ctx.evalset,
        TransformContext(workdir=ctx.workdir / "sweep", evalset=ctx.evalset),
        per_channel=per_channel,
    )
    order = order_by_measured_damage(results)
    cache[key] = order
    return order


def order_by_measured_damage(results: Any) -> list[str]:
    """Most prediction changes first; ties broken by the proxy; failures last."""
    measured = [r for r in results if r.changed_fraction is not None]
    unmeasured = [r for r in results if r.changed_fraction is None]
    measured.sort(key=lambda r: (r.changed_fraction, r.proxy_error), reverse=True)
    return [r.node for r in measured] + [r.node for r in unmeasured]


def load_measured_ranking(path: Path) -> list[str]:
    """Read the node order out of an ``anneal sensitivity --measured`` result file."""
    import json

    from anneal.core.sensitivity import LayerSensitivity

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = [
        LayerSensitivity(
            node=layer["node"],
            op_type=layer.get("op_type", "?"),
            proxy_error=float(layer.get("proxy_error", 0.0)),
            changed_fraction=layer.get("changed_fraction"),
            accuracy_drop_pp=layer.get("accuracy_drop_pp"),
            error=layer.get("error"),
        )
        for layer in data.get("layers", [])
    ]
    if not layers:
        raise ValueError(f"{path} contains no layer measurements")
    return order_by_measured_damage(layers)


#: Which op types static quantization touches. "default" is onnxruntime's own list, which
#: also quantizes LayerNormalization, Add, Softmax and friends. "compute" quantizes only the
#: inputs and weights of the matrix products. On ViT-B/16 the default put an 8-bit scale on
#: every LayerNorm input -- the residual stream, whose outlier channels span ~9x the median --
#: and lost 7.2pp; "compute" lost none (emulated, 512 images) while keeping every matmul INT8.
COMPUTE_OP_SETS: dict[str, list[str] | None] = {
    "default": None,
    "compute": ["Conv", "MatMul", "Gemm"],
    # Convolutions and the classifier only: activation-activation MatMuls (attention) stay float.
    "conv": ["Conv", "Gemm"],
}

#: Element-wise ops that belong to a convolution's activation (the stem's SiLU, Hardswish...).
_ACTIVATION_OPS = ("Sigmoid", "HardSigmoid", "Mul", "Relu", "Clip", "HardSwish", "Add", "Div")


def _with_named_nodes(path: Path) -> Path:
    """``path``, or a copy of it in which every node has a name (exclusions work by name)."""
    import onnx

    model = onnx.load(str(path))
    if all(n.name for n in model.graph.node):
        return path
    from anneal.core.equalize import _name_unnamed_nodes

    _name_unnamed_nodes(model)
    named = path.with_name(path.stem + "-named.onnx")
    onnx.save(model, str(named))
    return named


def stem_nodes(model_path: Path) -> list[str]:
    """The first convolution and the activation ops that follow it, up to the next Conv.

    The stem sees raw pixels and, in every net studied here, is the most quantization-
    sensitive block (the saturating layer on ResNet-18; the starved SiLU on EfficientNet-B0).
    Keeping it in float is cheap: it is one layer of dozens.
    """
    import onnx

    nodes = list(onnx.load(str(model_path)).graph.node)
    first = next((i for i, n in enumerate(nodes) if n.op_type == "Conv"), None)
    if first is None:
        return []
    names = [nodes[first].name]
    for n in nodes[first + 1:]:
        if n.op_type not in _ACTIVATION_OPS:
            break
        names.append(n.name)
    return [n for n in names if n]


def stem_output(model_path: Path) -> str | None:
    """The tensor the stem block hands to the rest of the network (its last activation's output)."""
    import onnx

    names = set(stem_nodes(model_path))
    last = [n for n in onnx.load(str(model_path)).graph.node if n.name in names]
    return last[-1].output[0] if last else None


def quantize_static_int8(
    artifact: ModelArtifact, params: dict[str, Any], ctx: TransformContext
) -> ModelArtifact:
    """Full INT8 (weights *and* activations) using real calibration data.

    Defaults to U8S8 — unsigned activations, signed weights — which is the combination
    x86 VNNI kernels are built for. ``reduce_range`` trades a bit of precision for
    overflow safety on pre-VNNI AVX2 machines.

    ``equalize`` first rescales channels across Conv -> SiLU/Hardswish/ReLU -> depthwise Conv
    (see :mod:`anneal.core.equalize`) so each shared activation scale serves every channel.
    The float model is unchanged. With ``per_channel`` weights the per-channel weight scales
    absorb the rescale exactly. With per-tensor weights the rescale also moves precision between
    the channels of the two weight tensors, so the scale is blended with cross-layer weight
    equalisation: ``equalize_mix`` = t uses ``mix=(1 - t, t)`` of
    :func:`anneal.core.equalize.equalise` (0 = activation equalisation only, 1 = plain CLE),
    defaulting to 0.5 when ``per_channel`` is off (MobileNetV3-Small, per-tensor weights,
    onnxruntime: -63.5pp plain, -8.6pp at t=0, -7.2pp at t=0.5, -11.1pp at t=1).
    ``float_gates`` additionally keeps each gate branch (the
    inserted Mul and the Sigmoid/HardSigmoid) out of quantization. ``equalize_top_k`` limits the
    rewrite to the k sites :func:`anneal.core.equalize.rank_sites` predicts gain most.

    ``cle`` first applies data-free cross-layer weight equalisation
    (:mod:`anneal.core.cle`) across Conv/Gemm -> ReLU/ReLU6 -> Conv/Gemm pairs, for targets whose
    weights are quantized per tensor; it runs before ``equalize`` when both are set.
    ``cle_max_scale`` caps each pair's cumulative scale (default 1000); 4 kept MobileNetV2 on
    TI's TDA4VM at -1.2pp where uncapped CLE lost 16.7pp through its unfused ReLU6 ceilings.
    ``sigmoid_surrogate`` (K > 0) then replaces every Sigmoid by a per-gate fitted sum of K
    fixed HardSigmoids (see :mod:`anneal.core.surrogate`), the only gate AMD's NPUs run.

    ``int16_top_k`` keeps the k activation tensors that are most damaging at 8 bits in 16 bits,
    ranked by :func:`anneal.core.activation_sensitivity.rank_activation_tensors` on the model
    being quantized (after equalisation), with a probe of the first :data:`INT16_PROBE_IMAGES`
    calibration images. ``int16_tensors`` names them by hand instead.
    """
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    if ctx.calibset is None and ctx.evalset is None:
        raise TransformError("static quantization needs calibration data; none supplied")
    calib_source = ctx.calibset if ctx.calibset is not None else ctx.evalset
    leaks = ctx.calibset is None

    per_channel = bool(params.get("per_channel", True))
    reduce_range = bool(params.get("reduce_range", False))
    calib_method = params.get("calibrate_method", "minmax")
    activation_type = params.get("activation_type", "uint8")
    if activation_type not in ("uint8", "int8"):
        raise TransformError(f"activation_type must be 'uint8' or 'int8', got {activation_type!r}")
    n_calib = int(params.get("calib_samples", ctx.calib_samples))
    guard = bool(params.get("guard_saturation", False))
    tolerance = float(params.get("saturation_tolerance", SATURATION_TOLERANCE))
    cle = bool(params.get("cle", False))
    cle_max_scale = params.get("cle_max_scale")
    if cle_max_scale is not None:
        if not cle:
            raise TransformError("cle_max_scale only applies together with cle")
        if (isinstance(cle_max_scale, bool) or not isinstance(cle_max_scale, (int, float))
                or not np.isfinite(cle_max_scale) or cle_max_scale < 1):
            raise TransformError(f"cle_max_scale must be a number >= 1, got {cle_max_scale!r}")
        cle_max_scale = float(cle_max_scale)
    equalize = bool(params.get("equalize", False))
    equalize_dense = bool(params.get("equalize_dense", False))
    equalize_residual = bool(params.get("equalize_residual", False))
    equalize_se = bool(params.get("equalize_se", False))
    equalize_gate_conv = bool(params.get("equalize_gate_conv", False))
    grid_inverse = bool(params.get("equalize_grid_inverse", False))
    derived = bool(params.get("equalize_derived", False))
    slack = float(params.get("equalize_slack", EQUALIZE_SLACK))
    top_k = params.get("equalize_top_k")
    min_gain = params.get("equalize_min_gain")
    min_damage = params.get("equalize_min_damage")
    eq_mix = params.get("equalize_mix")
    calib_stride = params.get("calib_stride")
    stem_int16 = bool(params.get("stem_int16", False))
    int16_tensors = params.get("int16_tensors")
    if int16_tensors is not None and (not isinstance(int16_tensors, list)
                                      or not all(isinstance(t, str) for t in int16_tensors)):
        raise TransformError("int16_tensors must be a list of tensor names")
    int16_top_k = params.get("int16_top_k")
    if int16_top_k is not None:
        if isinstance(int16_top_k, bool) or not isinstance(int16_top_k, int) or int16_top_k < 0:
            raise TransformError(f"int16_top_k must be a non-negative integer, got {int16_top_k!r}")
        if int16_tensors is not None:
            raise TransformError("give int16_tensors or int16_top_k, not both")
    if calib_stride is not None and (isinstance(calib_stride, bool) or not isinstance(calib_stride, int) or calib_stride < 1):
        raise TransformError(f"calib_stride must be a positive integer, got {calib_stride!r}")
    float_gates = bool(params.get("float_gates", False))
    float_stem = bool(params.get("float_stem", False))
    concat_shared = bool(params.get("concat_shared_scale", False))
    float_mixed = bool(params.get("float_mixed_outputs", True))
    act_symmetric = bool(params.get("activation_symmetric", False))
    pow2 = bool(params.get("pow2_activation_scales", False))
    quantize_ops = params.get("quantize_ops", "default")
    if quantize_ops not in COMPUTE_OP_SETS:
        raise TransformError(f"quantize_ops must be one of {sorted(COMPUTE_OP_SETS)}, got {quantize_ops!r}")
    percentile = float(params.get("calib_percentile", 99.99))
    surrogate_k = params.get("sigmoid_surrogate", 0)
    if isinstance(surrogate_k, bool) or not isinstance(surrogate_k, int) or not 0 <= surrogate_k <= 8:
        raise TransformError(f"sigmoid_surrogate must be an integer in [0, 8], got {surrogate_k!r}")
    if eq_mix is not None:
        if not equalize:
            raise TransformError("equalize_mix only applies together with equalize")
        if isinstance(eq_mix, bool) or not isinstance(eq_mix, (int, float)) or not 0 <= eq_mix <= 1:
            raise TransformError(f"equalize_mix must be a number in [0, 1], got {eq_mix!r}")
        eq_mix = float(eq_mix)
    elif equalize and not per_channel:
        # One weight scale per tensor: pure activation equalisation would move rounding error
        # into the weights, so blend in cross-layer weight equalisation (measured best at 0.5).
        eq_mix = EQUALIZE_PER_TENSOR_MIX
    if equalize_residual and not equalize:
        raise TransformError("equalize_residual only applies together with equalize")
    if equalize_se and not equalize:
        raise TransformError("equalize_se only applies together with equalize")
    if equalize_gate_conv and not equalize:
        raise TransformError("equalize_gate_conv only applies together with equalize")
    if float_gates and not (equalize or equalize_dense):
        raise TransformError("float_gates only applies together with equalize or equalize_dense")
    if equalize_dense and not per_channel:
        raise TransformError("equalize_dense needs per_channel weights")
    if not 0.0 <= slack <= 4.0:
        raise TransformError(f"equalize_slack must be in [0, 4], got {slack}")
    if top_k is not None:
        if not equalize:
            raise TransformError("equalize_top_k only applies together with equalize")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise TransformError(f"equalize_top_k must be a non-negative integer, got {top_k!r}")
    if min_damage is not None:
        if not equalize:
            raise TransformError("equalize_min_damage only applies together with equalize")
        if top_k is not None or min_gain is not None:
            raise TransformError("give one of equalize_top_k, equalize_min_gain, equalize_min_damage")
        if isinstance(min_damage, bool) or not isinstance(min_damage, (int, float)) or not 0 <= min_damage <= 1:
            raise TransformError(f"equalize_min_damage must be in [0, 1], got {min_damage!r}")
    if min_gain is not None:
        if not equalize:
            raise TransformError("equalize_min_gain only applies together with equalize")
        if top_k is not None:
            raise TransformError("give equalize_top_k or equalize_min_gain, not both")
        if isinstance(min_gain, bool) or not isinstance(min_gain, (int, float)) or min_gain < 0:
            raise TransformError(f"equalize_min_gain must be a non-negative number, got {min_gain!r}")

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        # onnxruntime's percentile calibration makes each range symmetric around zero by
        # default, which wastes half the uint8 grid on post-SiLU tensors. The asymmetric form
        # clips outliers without that: -12.9pp vs -19.1pp on EfficientNet-B0.
        "percentile_asym": CalibrationMethod.Percentile,
        # OpenVINO/NNCF's default estimator: the mean over calibration images of each image's
        # min and max. onnxruntime has no such calibrator (its moving average is exponential),
        # so the ranges are measured here and passed as rmin/rmax overrides.
        "mean_minmax": CalibrationMethod.MinMax,
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
            "activation_type": activation_type,
            # Recorded only when on, so recipes without the guard keep their lineage keys.
            **({"guard_saturation": True, "saturation_tolerance": tolerance} if guard else {}),
            **({"cle": True} if cle else {}),
            # Recorded only when set, so uncapped (default 1000x) CLE recipes keep their keys.
            **({"cle_max_scale": cle_max_scale} if cle_max_scale is not None else {}),
            **(
                {"equalize": True, "equalize_slack": slack, "float_gates": float_gates}
                if equalize
                else {}
            ),
            **({"equalize_top_k": top_k} if top_k is not None else {}),
            **({"equalize_residual": True} if equalize_residual else {}),
            **({"equalize_se": True} if equalize_se else {}),
            **({"equalize_gate_conv": True} if equalize_gate_conv else {}),
            **({"equalize_grid_inverse": True} if grid_inverse else {}),
            **({"equalize_derived": True} if derived else {}),
            **({"equalize_min_gain": min_gain} if min_gain is not None else {}),
            **({"equalize_min_damage": min_damage} if min_damage is not None else {}),
            # Recorded only when set (always for per-tensor equalisation, whose default is 0.5),
            # so per-channel recipes keep their lineage keys.
            **({"equalize_mix": eq_mix} if eq_mix is not None else {}),
            **({"calib_stride": calib_stride} if calib_stride is not None else {}),
            **({"stem_int16": True} if stem_int16 else {}),
            **({"int16_tensors": list(int16_tensors)} if int16_tensors else {}),
            **({"int16_top_k": int16_top_k} if int16_top_k is not None else {}),
            **({"float_stem": True} if float_stem else {}),
            **({"concat_shared_scale": True} if concat_shared else {}),
            # On by default and recorded only when switched off, so earlier lineages are unchanged.
            **({"float_mixed_outputs": False} if not float_mixed else {}),
            **({"activation_symmetric": True} if act_symmetric else {}),
            **({"pow2_activation_scales": True} if pow2 else {}),
            **({"equalize_dense": True, "equalize_slack": slack, "float_gates": float_gates}
               if equalize_dense else {}),
            **({"quantize_ops": quantize_ops} if quantize_ops != "default" else {}),
            **({"sigmoid_surrogate": surrogate_k} if surrogate_k else {}),
            **({"calib_percentile": percentile} if calib_method.startswith("percentile") and "calib_percentile" in params else {}),
            **({"float_nodes": sorted(str(n) for n in params["float_nodes"])} if params.get("float_nodes") else {}),
        },
    )
    out = ctx.path_for(artifact, record)
    src = _preprocessed(artifact, ctx)

    base_exclude: list[str] = []
    eq_meta: dict[str, Any] = {}
    if cle:
        # Cross-layer weight equalisation first: it changes weights only, so the activation
        # equalisation below (and its joint-damage switch) sees the model being quantized.
        from anneal.core.cle import cross_layer_equalise

        cle_path = out.with_name(out.stem + "-cle-fp32.onnx")
        cle_result = cross_layer_equalise(
            src, cle_path, check_batch=next(iter(calib_source.calibration_batches(1)), None),
            **({"max_scale": cle_max_scale} if cle_max_scale is not None else {}),
        )
        src = cle_path
        eq_meta["cle"] = cle_result.summary()
        eq_meta["cle_pairs"] = [p.to_dict() for p in cle_result.pairs]
    if equalize and min_damage is not None:
        # The model-level switch: equalise only if rounding every site tensor to 8 bits at once
        # flips enough of the calibration images' top-1 predictions (no labels needed). The
        # summed predicted gain failed this job (docs/zoo_gated_predictions.md); joint damage
        # ordered the measured benefit on the same eight models.
        import onnx

        from anneal.core.activation_sensitivity import joint_damage
        from anneal.core.equalize import _name_unnamed_nodes, find_sites

        probe_model = onnx.load(str(src))
        _name_unnamed_nodes(probe_model)
        # The same sites the rewrite below would take, with every tensor each one rescales.
        probe_sites = find_sites(probe_model, residual=equalize_residual, se=equalize_se)
        site_tensors = sorted({t for site in probe_sites for t in site.tensors})
        batches = list(calib_source.calibration_batches(n_calib))
        damage = joint_damage(src, site_tensors, batches, batches)
        eq_meta["joint_damage"] = round(damage, 4)
        if damage < float(min_damage):
            equalize = False
            eq_meta["equalisation_skipped"] = f"joint damage {damage:.3f} < {min_damage}"
    if equalize:
        from anneal.core.equalize import equalise

        eq_path = out.with_name(out.stem + "-equalised-fp32.onnx")
        probe = next(iter(calib_source.calibration_batches(1)), None)
        result = equalise(
            src,
            eq_path,
            calib_source.calibration_batches(n_calib),
            slack=slack,
            check_batch=probe,
            top_k=top_k,
            min_gain=None if min_gain is None else float(min_gain),
            residual=equalize_residual,
            se=equalize_se,
            mix=None if eq_mix is None else (1.0 - eq_mix, eq_mix),
            gate_conv=equalize_gate_conv,
            grid_inverse=grid_inverse,
            derived=derived,
        )
        src = eq_path
        if float_gates:
            base_exclude = list(result.gate_nodes)
        eq_meta = {
            **eq_meta,
            "equalisation": result.summary(),
            "equalised_sites": [s.to_dict() for s in result.sites],
            "equalised_site_ids": [s.producer for s in result.sites],
            # Every candidate, best first, so a top-k run shows what it left out.
            "equalisation_ranking": [
                {"site": r.site, "predicted_gain": round(r.gain, 4)} for r in result.ranking
            ],
        }
    if equalize_dense:
        from anneal.core.equalize_dense import equalise_dense

        dense_path = out.with_name(out.stem + "-equalised-dense-fp32.onnx")
        batches = list(calib_source.calibration_batches(n_calib))
        dense_sites, dense_gates, change = equalise_dense(src, dense_path, batches, slack=slack)
        src = dense_path
        if float_gates:
            base_exclude = base_exclude + dense_gates
        eq_meta["dense_equalisation"] = {
            "sites_rewritten": len(dense_sites),
            "max_abs_logit_change": change,
            "sites": [d.to_dict() for d in dense_sites],
        }
    if surrogate_k:
        # AMD's NPU toolchain swaps every Sigmoid for HardSigmoid(1/6, 1/2); a per-gate sum of
        # such HardSigmoids is a sigmoid it can run. After equalisation, which looks for the
        # Sigmoid gates, and on its output (the gate then sees x'/s = x, the same distribution).
        from anneal.core.surrogate import replace_sigmoids

        sur_path = out.with_name(out.stem + f"-surrogate{surrogate_k}-fp32.onnx")
        report = replace_sigmoids(
            src, sur_path, calib_source.calibration_batches(n_calib), k_terms=surrogate_k
        )
        src = sur_path
        if base_exclude:
            # float_gates named the Sigmoid nodes it keeps in float; keep their replacements.
            replaced = {g["node"]: g["nodes"] for g in report["per_gate"] if g["node"]}
            # A gate conv folded into the surrogate's terms is gone; its copies are among "nodes".
            replaced.update({g["folded"]: [] for g in report["per_gate"] if g.get("folded")})
            base_exclude = [n for old in base_exclude for n in replaced.get(old, [old])]
        eq_meta["sigmoid_surrogate"] = {k: v for k, v in report.items() if k != "per_gate"}
        eq_meta["sigmoid_surrogate_gates"] = [
            {k: v for k, v in g.items() if k != "nodes"} for g in report["per_gate"]
        ]

    def run_quantizer(exclude: list[str]) -> None:
        with _entropy_bins(calib_method == "entropy"):
            _quantize(exclude)

    def _quantize(exclude: list[str]) -> None:
        quantize_static(
            model_input=str(src),
            model_output=str(out),
            calibration_data_reader=_EvalSetCalibrationReader(calib_source, _input_name(src), n_calib),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8 if activation_type == "uint8" else QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=reduce_range,
            calibrate_method=methods[calib_method],
            nodes_to_exclude=exclude,
            op_types_to_quantize=COMPUTE_OP_SETS[quantize_ops],
            extra_options=extra,
        )

    extra: dict[str, Any] = {}
    if calib_method.startswith("percentile"):
        extra["CalibPercentile"] = percentile
    if calib_method == "percentile_asym":
        extra["CalibTensorRangeSymmetric"] = False
    if calib_stride is not None:
        extra["CalibStridedMinMax"] = calib_stride
    if stem_int16:
        # EfficientNet-B1: after equalisation the stem's output alone still flipped 13.7% of
        # top-1 predictions at 8 bits (tensor_sensitivity.py); one 16-bit tensor, not the model.
        src = _with_named_nodes(src)
        stem_out = stem_output(src)
        if stem_out is None:
            raise TransformError("stem_int16: no stem convolution found")
        extra["TensorQuantOverrides"] = {stem_out: [{"quant_type": QuantType.QUInt16}]}
    if int16_tensors:
        # Tensors ranked most sensitive by noise injection (examples/advise/tensor_sensitivity.py).
        import onnx

        known = {o for n in onnx.load(str(src)).graph.node for o in n.output}
        missing = [t for t in int16_tensors if t not in known]
        if missing:
            raise TransformError(f"int16_tensors not in the model: {missing[:3]}")
        extra.setdefault("TensorQuantOverrides", {}).update(
            {t: [{"quant_type": QuantType.QUInt16}] for t in int16_tensors})
    int16_meta: dict[str, Any] = {}
    if int16_top_k:
        # Rank on the model actually being quantized (after equalisation). The probe is the
        # first INT16_PROBE_IMAGES calibration images, never the eval set: no leakage.
        from anneal.core.activation_sensitivity import rank_activation_tensors

        probe_x: list[np.ndarray] = []
        seen = 0
        for x in calib_source.calibration_batches(INT16_PROBE_IMAGES):
            x = x[: INT16_PROBE_IMAGES - seen]
            probe_x.append(x)
            seen += x.shape[0]
            if seen >= INT16_PROBE_IMAGES:
                break
        ranking = rank_activation_tensors(src, calib_source.calibration_batches(n_calib), probe_x)
        del probe_x
        chosen = ranking[:int16_top_k]
        extra.setdefault("TensorQuantOverrides", {}).update(
            {r.tensor: [{"quant_type": QuantType.QUInt16}] for r in chosen})
        int16_meta = {
            "int16_top_k_tensors": [r.tensor for r in chosen],
            "int16_top_k_damage": {r.tensor: round(r.damage, 5) for r in chosen},
            "int16_probe_images": seen,
            "int16_probe_source": "calibration set (not the eval set)",
            "activation_sensitivity": [
                {"tensor": r.tensor, "damage": round(r.damage, 5)} for r in ranking
            ],
        }
    if act_symmetric:
        extra["ActivationSymmetric"] = True
    if calib_method == "mean_minmax":
        overrides = extra.setdefault("TensorQuantOverrides", {})
        for t, (lo, hi) in mean_minmax_ranges(src, calib_source.calibration_batches(n_calib)).items():
            entry = overrides.setdefault(t, [{}])[0]
            entry.update({"rmin": np.float32(lo), "rmax": np.float32(hi)})
    if float_stem:
        src = _with_named_nodes(src)
        base_exclude = base_exclude + [n for n in stem_nodes(src) if n not in base_exclude]
    mixed_meta: dict[str, Any] = {}
    if float_mixed and outputs_fed_by_concat(src):
        # A model output that concatenates tensors of very different ranges (a detector's pixel
        # boxes with its 0-1 scores) cannot share one 8-bit scale: Ultralytics' YOLOv8n export
        # scored 0 mAP. Keep that output's tail (everything after its last Conv/Gemm/MatMul) float.
        src = _with_named_nodes(src)
        flagged = mixed_range_outputs(src, calib_source.calibration_batches(MIXED_CHECK_IMAGES))
        for f in flagged:
            base_exclude = base_exclude + [n for n in f["tail_nodes"] if n not in base_exclude]
        mixed_meta = {"mixed_range_outputs": [{k: v for k, v in f.items() if k != "tail_nodes"} | {
            "float_nodes": len(f["tail_nodes"])} for f in flagged]}
    float_nodes = [str(n) for n in (params.get("float_nodes") or [])]
    if float_nodes:  # diagnostic: named nodes stay float (bisecting where a model breaks)
        base_exclude = base_exclude + [n for n in float_nodes if n not in base_exclude]
    run_quantizer(base_exclude)
    concat_meta: dict[str, Any] = {}
    if concat_shared:
        # Accelerators (TIDL, Hexagon HTP, ...) give a Concat's inputs and output one scale;
        # onnxruntime gives each its own. Emulate the accelerator: read every range back from the
        # quantized model, take each Concat group's union, and quantize again with it pinned.
        groups = concat_groups(out)
        overrides = extra.setdefault("TensorQuantOverrides", {})
        for names, (lo, hi) in groups:
            for t in names:
                overrides.setdefault(t, [{}])[0].update({"rmin": np.float32(lo), "rmax": np.float32(hi)})
        concat_meta = {"concat_groups_shared": len(groups)}
        run_quantizer(base_exclude)
    if pow2:
        # TI's TIDL (and AMD's XINT8) scale feature maps by powers of two: each activation scale
        # is rounded up to the next one, which can cost up to 1 bit of resolution per tensor.
        concat_meta["pow2_activation_scales"] = round_activation_scales_pow2(out)
    guard_meta: dict[str, Any] = {}
    if guard:
        # Emulate the 16-bit pair arithmetic of x86 CPUs without VNNI on this very model and
        # keep only the layers that saturate in FP32. Everything else keeps full 8-bit weights.
        from anneal.core.saturation import analyse

        probe = [next(iter(calib_source.batches()))[0]]
        layers = analyse(out, probe, n_positions=128)
        saturating = {r.node: r.accumulator_rate for r in layers if r.accumulator_rate > tolerance}
        if saturating:
            run_quantizer(base_exclude + sorted(saturating))
        guard_meta = {
            "saturation_excluded": sorted(saturating),
            "saturation_rates": {k: round(v, 5) for k, v in saturating.items()},
        }

    return artifact.derive(
        record,
        out,
        **guard_meta,
        **eq_meta,
        **int16_meta,
        **concat_meta,
        **mixed_meta,
        calib_samples=n_calib,
        calibration_source=(
            "eval set (overlaps evaluation images; accuracy is optimistic)"
            if leaks
            else f"{getattr(calib_source, 'name', 'calibration set')} "
            f"{getattr(calib_source, 'split', '')}".strip()
        ),
    )


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
            "layers in FP32. The main dial for trading a little speed back for accuracy. "
            "By default layers are ranked by a measured one-layer-at-a-time sweep; the "
            "free weight-error proxy is available but weakly predictive."
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
            "ranking": {
                "type": "string",
                "enum": ["measured", "proxy"],
                "description": (
                    "How sensitivity is ranked. 'measured' quantizes each layer alone and "
                    "counts changed predictions (one eval pass per layer, cached per run). "
                    "'proxy' uses weight-quantization error: free, but on ResNet-18 it ranked "
                    "the most damaging layer last."
                ),
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
                "enum": ["minmax", "entropy", "percentile", "percentile_asym", "mean_minmax"],
                "description": (
                    "How activation ranges are estimated. percentile_asym clips outliers without "
                    "forcing a symmetric range; with equalize it was the best recipe on "
                    "EfficientNet-B0 and MobileNetV3."
                ),
            },
            "calib_samples": {
                "type": "integer",
                "description": "Number of calibration images to use.",
            },
            "guard_saturation": {
                "type": "boolean",
                "description": (
                    "Emulate the saturating 16-bit arithmetic of x86 CPUs without VNNI on the "
                    "quantized model and keep only the layers that saturate in FP32. Use it when "
                    "the target is x86 without VNNI; elsewhere it costs speed for nothing."
                ),
            },
            "saturation_tolerance": {
                "type": "number",
                "description": "Fraction of a layer's accumulators allowed to saturate (default 0.02).",
            },
            "activation_type": {
                "type": "string",
                "enum": ["uint8", "int8"],
                "description": (
                    "Activation precision. uint8 with int8 weights (U8S8) is the classic x86 "
                    "pairing; int8 activations (S8S8) avoid the intermediate saturation U8S8 "
                    "can hit on CPUs without VNNI."
                ),
            },
            "cle": {
                "type": "boolean",
                "description": (
                    "Before quantizing (and before equalize), apply data-free cross-layer weight "
                    "equalisation (Nagel et al. 2019) across Conv/Gemm -> ReLU/LeakyReLU "
                    "(+ MaxPool/zero Pad) -> Conv/depthwise Conv/Gemm, so each weight tensor's "
                    "channels share one range. Exact in float. For targets that quantize weights "
                    "per tensor (TI TIDL, AMD XINT8; per_channel false); ReLU6/Clip is left alone."
                ),
            },
            "cle_max_scale": {
                "type": "number",
                "minimum": 1,
                "description": (
                    "With cle: clamp each pair's cumulative scale to [1/m, m] (default 1000). "
                    "Small caps matter through ReLU6: the exact rewrite adds unfused per-channel "
                    "Min ceilings whose inputs are quantized before clipping. On TI TDA4VM, "
                    "MobileNetV2 with CLE lost 16.7pp uncapped, 12.9pp at 16 and only 1.2pp at 4 "
                    "(8-bit baseline -10.7pp)."
                ),
            },
            "equalize": {
                "type": "boolean",
                "description": (
                    "Before quantizing, rescale channels across Conv -> SiLU/Hardswish/ReLU -> "
                    "depthwise Conv so one shared activation scale serves every channel. Exact in "
                    "float. The fix for EfficientNet/MobileNetV3-style nets whose INT8 accuracy "
                    "collapses on every CPU. With per_channel false it is blended with cross-layer "
                    "weight equalisation (see equalize_mix, default 0.5 then)."
                ),
            },
            "equalize_slack": {
                "type": "number",
                "description": "How far equalisation may extend a tensor's range downward (default 0.1).",
            },
            "equalize_mix": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": (
                    "With equalize: blend t in [0, 1] of cross-layer weight equalisation into "
                    "the scale (0 = activation equalisation only, 1 = plain CLE), for weights "
                    "quantized per tensor. Default 0.5 with per_channel false (MobileNetV3-Small: "
                    "-63.5pp plain, -7.2pp at 0.5, -8.6pp at 0, -11.1pp at 1); unset otherwise."
                ),
            },
            "equalize_top_k": {
                "type": "integer",
                "description": (
                    "With equalize: rewrite only the k sites with the highest predicted gain "
                    "(rounding noise removed from starved channels) instead of all. Each gated "
                    "site adds one element-wise Mul, which is costly on NPUs; 0 = none."
                ),
            },
            "int16_tensors": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Activation tensors to quantize to 16 bits (the rest stay 8): mixed precision "
                    "for the few tensors ranked most sensitive by noise injection."
                ),
            },
            "int16_top_k": {
                "type": "integer",
                "description": (
                    "Quantize the k activation tensors that are most sensitive at 8 bits to 16 "
                    "bits, chosen automatically: after equalisation, each Conv/Gemm/MatMul data "
                    "input is fake-quantized alone (per-tensor uint8) in the float model and "
                    "ranked by the share of top-1 predictions it flips on a probe of the first "
                    "64 calibration images (never eval images). Cost: one float inference pass "
                    "over the probe per candidate tensor. Excludes int16_tensors; 0 = off."
                ),
            },
            "stem_int16": {
                "type": "boolean",
                "description": (
                    "Quantize the stem block's output tensor (the first depthwise conv's input) "
                    "to 16 bits, everything else to 8: mixed precision for the one tensor that "
                    "stays too coarse at 8 bits on EfficientNet-B1 even after equalisation."
                ),
            },
            "calib_stride": {
                "type": "integer",
                "description": (
                    "Calibrate in chunks of this many batches (onnxruntime CalibStridedMinMax): "
                    "bounds calibration memory, which otherwise holds every activation of every "
                    "calibration image. Histogram methods merge chunk by chunk, so ranges can "
                    "differ slightly from one-shot calibration."
                ),
            },
            "equalize_min_gain": {
                "type": "number",
                "description": (
                    "With equalize: rewrite all sites if the model's summed predicted gain "
                    "(channels' worth of quantization signal recovered) is at least this, none "
                    "otherwise. Superseded by equalize_min_damage: on eight gated classifiers the "
                    "summed gain did not predict collapse (5.2 collapsed, 153.5 did not)."
                ),
            },
            "quantize_ops": {
                "type": "string",
                "enum": ["default", "compute", "conv"],
                "description": (
                    "'compute' quantizes only Conv/MatMul/Gemm and leaves LayerNorm, residual "
                    "adds and softmax in float. The fix for transformers, whose residual stream "
                    "has outlier channels; on CNNs the default is usually better."
                ),
            },
            "float_stem": {
                "type": "boolean",
                "description": (
                    "Keep the first convolution and its activation in float. The stem sees raw "
                    "pixels and was the most sensitive block in every net studied."
                ),
            },
            "calib_percentile": {
                "type": "number",
                "description": "Percentile for percentile calibration (default 99.99).",
            },
            "equalize_residual": {
                "type": "boolean",
                "description": (
                    "With equalize: also rewrite gated sites whose output feeds a residual Add "
                    "as well as a depthwise Conv (EfficientViT's and MobileNetV3-Large's stems). "
                    "The branch joining the Add and the Add's consumers take the scale too."
                ),
            },
            "equalize_se": {
                "type": "boolean",
                "description": (
                    "With equalize: also rewrite gated and ReLU sites whose output feeds a "
                    "squeeze-excite block (EfficientNet's and MobileNetV3's depthwise conv -> "
                    "SiLU/Hardswish/ReLU -> SE -> projection). The depthwise conv's output channels take the scale; the "
                    "SE's first FC and the projection divide it out of their input channels."
                ),
            },
            "equalize_gate_conv": {
                "type": "boolean",
                "description": (
                    "With equalize: feed each gate x'/s through a depthwise 1x1 Conv (weight 1/s) "
                    "instead of an element-wise Mul. Same float function; on NPUs that fuse Conv + "
                    "activation (TI TIDL, AMD XINT8) the imbalanced x'/s is then never quantized per "
                    "tensor, which alone cost EfficientNet-B1 ~37pp there. With sigmoid_surrogate the "
                    "surrogate's per-term scale and shift fold into copies of that conv. Unproven: AMD "
                    "Quark's XINT8 output quantizes every such conv's output (195/195 HardSigmoids read "
                    "an 8-bit input), and B1 did not improve (-54.2 vs -50.1pp); whether a device "
                    "compiler fuses the pair is untested."
                ),
            },
            "float_mixed_outputs": {
                "type": "boolean",
                "description": (
                    "On by default. If a model output concatenates tensors whose ranges differ 20x "
                    "or more (a detector's pixel boxes and 0-1 scores), keep that output's tail "
                    "after the last Conv/Gemm/MatMul in float; one 8-bit scale on it rounds every "
                    "score to zero (Ultralytics' YOLOv8n export: 0 mAP)."
                ),
            },
            "activation_symmetric": {
                "type": "boolean",
                "description": "Symmetric activation ranges (zero point 0), as TI TIDL and AMD XINT8 use.",
            },
            "pow2_activation_scales": {
                "type": "boolean",
                "description": (
                    "Round every activation scale up to a power of two after quantization, as TI "
                    "TIDL and AMD XINT8 do for feature maps. With per_channel=False, "
                    "activation_type=int8, activation_symmetric and concat_shared_scale it "
                    "emulates TIDL's 8-bit arithmetic in onnxruntime."
                ),
            },
            "concat_shared_scale": {
                "type": "boolean",
                "description": (
                    "Emulate accelerators (TIDL, Hexagon HTP) that give a Concat's inputs and output "
                    "one shared scale: the union of their calibrated ranges. onnxruntime alone gives "
                    "each its own. Experimental: with asymmetric activations onnxruntime applies "
                    "the pinned ranges only to some Concat inputs, and a Carvana U-Net then "
                    "collapsed (IoU 0.08) while symmetric emulation held (0.95-0.98); trust it with "
                    "activation_symmetric only."
                ),
            },
            "equalize_min_damage": {
                "type": "number",
                "description": (
                    "With equalize: equalise only if rounding every equalisation site's tensors to "
                    "8 bits at once flips at least this share of the calibration images' top-1 "
                    "predictions (measured, no labels). 0.3 separated the models that collapse from "
                    "those that do not on eight gated classifiers."
                ),
            },
            "equalize_dense": {
                "type": "boolean",
                "description": (
                    "Also equalise gated activations into dense consumers (ConvNeXt's GELU -> "
                    "Linear, EfficientNetV2's Fused-MBConv, squeeze-excite MLPs), choosing the "
                    "strength per site by simulated INT8 error; sites where it would not help "
                    "are left alone."
                ),
            },
            "sigmoid_surrogate": {
                "type": "integer",
                "description": (
                    "For AMD NPUs/DPUs (Vitis AI, Ryzen AI via Quark), whose toolchain replaces "
                    "every Sigmoid with HardSigmoid(u/6 + 1/2): replace each Sigmoid first by a sum "
                    "of this many such HardSigmoids, sum_i w_i h(k_i x + b_i) with exact 0/1 "
                    "asymptotes, fitted per gate on its calibration inputs (x^2-weighted for SiLU). "
                    "AMD's plain swap costs EfficientNet-B0 48.8pp and B1 75.7pp top-1 in float; "
                    "3 terms: +0.2pp and -0.5pp vs FP32 (Imagenette 1000). Applied after equalisation; 0 = off."
                ),
            },
            "equalize_grid_inverse": {
                "type": "boolean",
                "description": (
                    "With equalize: choose the scales so every gate-side 1/s lies exactly on a "
                    "power-of-two int8 grid, so a target that quantizes that constant (Qualcomm, TIDL "
                    "emulation) loses nothing; otherwise channels with small 1/s round badly or to 0. "
                    "Exact; caps the per-site scale spread at 127x."
                ),
            },
            "equalize_derived": {
                "type": "boolean",
                "description": (
                    "With equalize: per-channel noise-optimal scales (minimise the summed output "
                    "noise of every tensor the site rescales, weighted by Hutchinson estimates of "
                    "the output sensitivity; needs onnx2torch, `pip install anneal[derived]`). "
                    "Use with equalize_grid_inverse. Measured on pure 8-bit targets: TIDL "
                    "EfficientNet-B1 -58.1 -> -31.4, B0 -11.2 -> -4.0 (1500 images); AMD XINT8 B0 "
                    "-5.4 -> -1.8 but B1 -8.0 -> -11.0; ties once 16-bit layers are mixed in."
                ),
            },
            "float_nodes": {
                "type": "array",
                "description": (
                    "Diagnostic: node names to keep in float, to bisect which part of a model "
                    "breaks under a target's quantization rules. Not a deployment recipe."
                ),
            },
            "float_gates": {
                "type": "boolean",
                "description": (
                    "With equalize: keep each gate branch (Sigmoid/HardSigmoid and its input "
                    "rescale) in float. More accuracy on SiLU nets; costs some speed."
                ),
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
