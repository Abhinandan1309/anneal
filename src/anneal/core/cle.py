"""Cross-layer weight equalisation (CLE) for accelerators that quantize weights per tensor.

Nagel et al. (2019), "Data-Free Quantization Through Weight Equalization and Bias Correction".

The failure this fixes. Accelerators such as TI's TIDL (TDA4VM) and AMD's XINT8 flow give each
weight tensor *one* scale. After batch-norm folding, the output channels of a convolution can
differ in range by 100x or more, so the small channels are rounded to a handful of levels (or
to zero). On TIDL's 8-bit emulation MobileNetV2 loses ~10pp top-1 and conv-BN-ReLU U-Nets
collapse. :mod:`anneal.core.equalize` balances *activations* and relies on per-channel weight
scales absorbing the rescale; with per-tensor weights the weights themselves need balancing.

The rewrite. For a pair

    x = P(...);   y = f(x);   z = C(y)       f: ReLU, LeakyReLU, ReLU6/Clip(0, M), MaxPool,
                                                zero Pad (any chain)

every f commutes with a positive per-channel scale: f(s * x) = s * f(x) for s > 0. Multiplying
producer output channel i (its weight row and bias) by s_i and dividing the consumer's input
channel i by s_i therefore leaves the float function unchanged. With r1_i the range of P's
row i and r2_i the range of C's input channel i,

    s_i = sqrt(r2_i / r1_i)    gives    r1_i' = r2_i' = sqrt(r1_i * r2_i),

which balances the two weight tensors' channels against each other. A convolution can be the
consumer of one pair and the producer of the next; the scales of a chain are coupled, so the
sweep over all pairs is repeated until they stop changing (as in Nagel et al.).

ReLU6 / Clip(0, M). A clip is not scale-equivariant: its ceiling does not move with s. It
becomes so with a per-channel ceiling, since Clip(0, M)(x) * s_c == Clip(0, M * s_c)(s_c * x).
ONNX Clip only takes scalar bounds, so every Clip(0, M) on the path of an equalised pair is
rewritten as ``Relu -> Min(., C)`` with C = M * s a constant of shape [1, channels, 1, ...]
([channels] for Gemm); the float function is unchanged. (MobileNetV2, ReLU6 everywhere, has no
CLE pair otherwise.) Only a lower bound of exactly 0 qualifies; a missing upper bound makes the
Clip a plain ReLU, which is left as it is. Bounds may be attributes (opset < 11) or constant
inputs (initializers or Constant nodes). The Min needs opset >= 8 (broadcasting). Clips of
skipped pairs are not touched.

Batch-norm is assumed to be folded already (onnxruntime's ``quant_pre_process`` does it).
The rescale is data-free: no calibration data is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ITERATIONS = 100
#: Converged when no pair's scale changes by more than this factor in a sweep (|s - 1|).
DEFAULT_THRESHOLD = 1e-4
DEFAULT_MAX_SCALE = 1e3

ACTIVATIONS = ("Relu", "LeakyRelu", "Clip")
#: |bound| at or above this is "no bound" (exporters write float32 max for a missing one).
_UNBOUNDED = 3.4e38
PASSTHROUGH = ("MaxPool", "Pad")


def _attr(node, name: str, default=None):
    from onnx import helper

    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def _clip_bounds(node, consts: dict, const_nodes: dict, opset: int):
    """``(lo, hi)`` of a Clip, ``None`` for a missing bound; ``None`` if a bound is not constant."""
    from onnx import numpy_helper

    def finite(v):
        return None if v is None or not np.isfinite(v) or abs(v) >= _UNBOUNDED else float(v)

    if opset < 11 and len(node.input) == 1:
        return finite(_attr(node, "min")), finite(_attr(node, "max"))
    bounds = []
    for k in (1, 2):
        name = node.input[k] if len(node.input) > k else ""
        if not name:
            bounds.append(None)
            continue
        if name in consts:
            arr = numpy_helper.to_array(consts[name])
        elif name in const_nodes:
            cn = const_nodes[name]
            if _attr(cn, "value") is not None:
                arr = numpy_helper.to_array(_attr(cn, "value"))
            elif _attr(cn, "value_float") is not None:
                arr = np.asarray(_attr(cn, "value_float"))
            else:
                return None
        else:
            return None
        if arr.size != 1:
            return None
        bounds.append(finite(float(arr.reshape(()))))
    return bounds[0], bounds[1]


@dataclass
class _Weight:
    """Where the channels of a weight tensor sit."""

    name: str
    axis: int  # the axis indexed by the pair's channel
    #: Depthwise consumer with channel multiplier m: input channel c owns outputs c*m..c*m+m-1.
    multiplier: int = 1


@dataclass
class CLEPairSpec:
    """A producer -> (activation / pooling / padding) -> consumer chain found in the graph."""

    producer: Any
    consumer: Any
    via: list[str]  # op types between them
    producer_weight: _Weight
    producer_bias: str | None  # bias tensor; scaled along its last axis
    consumer_weight: _Weight
    kind: str  # "conv", "depthwise" or "gemm"
    channels: int
    #: Clip(0, M) nodes on the path with a finite M, to become Relu -> Min(., M * s).
    ceilings: list[tuple[Any, float]] = field(default_factory=list)
    #: Rank of the activations on the path (4 for a 2-D conv, 2 for Gemm).
    rank: int = 4

    @property
    def id(self) -> str:
        return self.producer.name


@dataclass
class CLEPair:
    producer: str
    consumer: str
    via: list[str]
    kind: str
    channels: int
    scale_min: float
    scale_max: float
    #: Channels whose scale hit the [1/max_scale, max_scale] clamp.
    channels_clamped: int
    #: Median over channels of min(r1, r2) / max(r1, r2) before and after: 1 is balanced.
    balance_before: float
    balance_after: float
    #: Clip(0, M) nodes on the path rewritten as Relu -> Min(., per-channel ceiling).
    clips_converted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class CLEResult:
    pairs: list[CLEPair] = field(default_factory=list)
    #: Candidate pairs skipped because a weight or bias is shared with another node.
    skipped: list[str] = field(default_factory=list)
    #: Clip(0, M) nodes rewritten as Relu -> Min(., M * s) with a per-channel ceiling.
    clips_converted: list[str] = field(default_factory=list)
    iterations: int = 0
    converged: bool = True
    max_abs_output_change: float | None = None

    @property
    def max_scale(self) -> float:
        return max((max(p.scale_max, 1.0 / p.scale_min) for p in self.pairs), default=1.0)

    def summary(self) -> dict[str, Any]:
        return {
            "pairs": len(self.pairs),
            "skipped": len(self.skipped),
            "clips_converted": len(self.clips_converted),
            "iterations": self.iterations,
            "converged": self.converged,
            "max_scale": round(self.max_scale, 4),
            "max_abs_output_change": self.max_abs_output_change,
        }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def find_cle_pairs(model) -> list[CLEPairSpec]:
    """Producer -> ReLU/LeakyReLU/ReLU6 (+ MaxPool / zero Pad) -> consumer chains, in graph order.

    Producer: Conv (any group) or Gemm with a constant weight (and bias, if any). Between: any
    chain of Relu, LeakyRelu (alpha >= 0), Clip with constant bounds, lower bound exactly 0 and
    upper bound > 0 or missing (ReLU6; finite ceilings are listed in ``ceilings`` and need
    opset >= 8), MaxPool and constant-zero Pad that leaves the channel axis alone, with at least
    one activation. Consumer: a group-1 Conv (any kernel), a depthwise Conv (group == input
    channels) or a Gemm; Conv pairs with Conv and Gemm with Gemm. Every intermediate tensor
    must have exactly one consumer and not be a graph output. Any other Clip ends the chain.
    """
    from onnx import numpy_helper

    g = model.graph
    inits = {i.name: i for i in g.initializer}
    # An initializer that is also a graph input can be overridden at run time: not constant.
    graph_inputs = {i.name for i in g.input}
    consts = {k: v for k, v in inits.items() if k not in graph_inputs}
    const_nodes = {n.output[0]: n for n in g.node if n.op_type == "Constant" and n.output}
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 1)
    graph_outputs = {o.name for o in g.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            if i:
                consumers.setdefault(i, []).append(n)

    def only_consumer(tensor: str):
        outs = consumers.get(tensor, [])
        if len(outs) != 1 or tensor in graph_outputs:
            return None
        return outs[0]

    def const_zero(name: str) -> bool:
        return name in consts and not np.any(numpy_helper.to_array(consts[name]))

    def passthrough(node) -> bool:
        """True for ops that commute with a positive per-channel scale."""
        if node.op_type == "Relu":
            return True
        if node.op_type == "LeakyRelu":
            return float(_attr(node, "alpha", 0.01)) >= 0.0
        if node.op_type == "Clip":
            if node.domain not in ("", "ai.onnx"):
                return False
            bounds = _clip_bounds(node, consts, const_nodes, opset)
            if bounds is None or bounds[0] != 0.0:
                return False
            return bounds[1] is None or (bounds[1] > 0.0 and opset >= 8)
        if node.op_type == "MaxPool":
            # A used Indices output is fine too (unchanged by s > 0), but keep it simple.
            return len([o for o in node.output if o]) == 1
        if node.op_type == "Pad":
            mode = _attr(node, "mode", b"constant")
            if isinstance(mode, bytes):
                mode = mode.decode()
            if mode != "constant":
                return False
            if len(node.input) > 1 and node.input[1]:
                if node.input[1] not in consts:
                    return False
                pads = numpy_helper.to_array(consts[node.input[1]]).astype(np.int64).ravel()
                if len(node.input) > 3 and node.input[3]:
                    return False  # opset-18 axes: rare; not worth decoding here
                if len(node.input) > 2 and node.input[2] and not const_zero(node.input[2]):
                    return False
            else:
                pads = np.asarray(_attr(node, "pads", []), dtype=np.int64)
                if float(_attr(node, "value", 0.0)) != 0.0:
                    return False
            rank = len(pads) // 2
            # The channel axis (1) must not be padded: that would shift the channels.
            return rank >= 2 and pads[1] == 0 and pads[rank + 1] == 0
        return False

    def conv_weight(node):
        if node.op_type != "Conv" or len(node.input) < 2 or node.input[1] not in consts:
            return None
        w = consts[node.input[1]]
        return w if len(w.dims) >= 3 else None

    def gemm_weight(node):
        if node.op_type != "Gemm" or len(node.input) < 2 or node.input[1] not in consts:
            return None
        w = consts[node.input[1]]
        return w if len(w.dims) == 2 else None

    pairs: list[CLEPairSpec] = []
    for p in g.node:
        if p.op_type == "Conv":
            wp = conv_weight(p)
            if wp is None:
                continue
            channels = int(wp.dims[0])
            p_weight = _Weight(p.input[1], axis=0)
        elif p.op_type == "Gemm":
            wp = gemm_weight(p)
            if wp is None:
                continue
            trans_b = int(_attr(p, "transB", 0))
            channels = int(wp.dims[0] if trans_b else wp.dims[1])
            p_weight = _Weight(p.input[1], axis=0 if trans_b else 1)
        else:
            continue
        bias = p.input[2] if len(p.input) > 2 and p.input[2] else None
        if bias is not None:
            if bias not in consts:
                continue
            bdims = list(consts[bias].dims)
            # Scaled along its last axis, so that axis must be the channel axis.
            if not bdims or bdims[-1] != channels or (p.op_type == "Conv" and len(bdims) != 1):
                continue

        # Walk the chain of scale-equivariant ops.
        via: list[str] = []
        ceilings: list[tuple[Any, float]] = []
        tensor = p.output[0]
        node = only_consumer(tensor)
        while node is not None and passthrough(node) and node.input[0] == tensor:
            via.append(node.op_type)
            if node.op_type == "Clip":
                hi = _clip_bounds(node, consts, const_nodes, opset)[1]
                if hi is not None:
                    ceilings.append((node, hi))
            tensor = node.output[0]
            node = only_consumer(tensor)
        if node is None or not any(v in ACTIVATIONS for v in via) or node.input[0] != tensor:
            continue
        c = node
        if p.op_type == "Conv":
            wc = conv_weight(c)
            if wc is None:
                continue
            group = int(_attr(c, "group", 1))
            if group == 1 and wc.dims[1] == channels:
                kind, c_weight = "conv", _Weight(c.input[1], axis=1)
            elif group == channels and wc.dims[1] == 1 and wc.dims[0] % channels == 0:
                kind = "depthwise"
                c_weight = _Weight(c.input[1], axis=0, multiplier=int(wc.dims[0]) // channels)
            else:
                continue
        else:
            wc = gemm_weight(c)
            if wc is None or int(_attr(c, "transA", 0)):
                continue
            trans_b = int(_attr(c, "transB", 0))
            if (wc.dims[1] if trans_b else wc.dims[0]) != channels:
                continue
            kind, c_weight = "gemm", _Weight(c.input[1], axis=1 if trans_b else 0)
        rank = len(wp.dims) if p.op_type == "Conv" else 2
        pairs.append(CLEPairSpec(p, c, via, p_weight, bias, c_weight, kind, channels, ceilings, rank))
    return pairs


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------


def _channel_range(arr: np.ndarray, w: _Weight, channels: int) -> np.ndarray:
    """max |w| per pair channel."""
    a = np.moveaxis(np.abs(arr), w.axis, 0)
    return a.reshape(channels, -1).max(axis=1)


def _scale(arr: np.ndarray, w: _Weight, s: np.ndarray, divide: bool) -> np.ndarray:
    f = np.repeat(s, w.multiplier)
    f = 1.0 / f if divide else f
    shape = [1] * arr.ndim
    shape[w.axis] = -1
    return arr * f.reshape(shape)


def _balance(r1: np.ndarray, r2: np.ndarray) -> float:
    ok = (r1 > 0) & (r2 > 0)
    if not ok.any():
        return 1.0
    return float(np.median(np.minimum(r1[ok], r2[ok]) / np.maximum(r1[ok], r2[ok])))


def cle_scales(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """s = sqrt(r2 / r1): after producer * s and consumer / s both ranges are sqrt(r1 r2).

    Channels where either range is zero (a dead producer row or an unused input) keep 1.
    """
    s = np.ones(r1.shape, dtype=np.float64)
    ok = (r1 > 0) & (r2 > 0)
    s[ok] = np.sqrt(r2[ok] / r1[ok])
    return s


def cross_layer_equalise(
    src: Path,
    dst: Path,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    threshold: float = DEFAULT_THRESHOLD,
    max_scale: float = DEFAULT_MAX_SCALE,
    check_batch: np.ndarray | None = None,
    batches: list[np.ndarray] | None = None,
    t: float = 1.0,
) -> CLEResult:
    """Write a cross-layer-equalised copy of ``src`` to ``dst`` and describe what changed.

    Sweeps over every pair of :func:`find_cle_pairs`, rescaling producer rows (and bias) by
    ``s = sqrt(r2 / r1)`` and consumer input channels by ``1 / s``, until no scale in a sweep
    differs from 1 by more than ``threshold`` or ``iterations`` sweeps have run. Each pair's
    *cumulative* scale is clamped to ``[1 / max_scale, max_scale]``. Pairs whose weight or bias
    is shared with another node are skipped. Every Clip(0, M) on an equalised pair's path is
    rewritten as ``Relu -> Min(., M * s)`` with a per-channel ceiling (listed in
    ``clips_converted``), which keeps the float function exact. With ``check_batch`` the float outputs of both
    models are compared on it and the largest change recorded.

    ``t < 1`` (needs ``batches``) makes it activation-aware. Weight-only CLE balances the two
    weight tensors but can widen a channel's activation range hundreds of times; on TI's TDA4VM,
    whose feature maps are per tensor with power-of-two scales, it cost MobileNetV2 6.3pp (-10.5
    -> -16.7pp). The scale becomes ``s_act ** (1 - t) * s_cle ** t``, where ``s_act`` fills the
    producer output's shared range (:func:`anneal.core.equalize.choose_scales`, positive) and
    ``s_cle`` is the converged weight balance: the same geometric mix as ``equalise(mix=...)``.
    """
    import onnx
    from onnx import numpy_helper

    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations!r}")
    if not max_scale >= 1.0:
        raise ValueError(f"max_scale must be at least 1, got {max_scale!r}")

    from anneal.core.equalize import _name_unnamed_nodes

    model = onnx.load(str(src))
    _name_unnamed_nodes(model)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    uses: dict[str, int] = {}
    for n in g.node:
        for i in set(n.input):
            uses[i] = uses.get(i, 0) + 1

    result = CLEResult()
    pairs: list[CLEPairSpec] = []
    for pair in find_cle_pairs(model):
        names = [pair.producer_weight.name, pair.consumer_weight.name]
        if pair.producer_bias:
            names.append(pair.producer_bias)
        if len(set(names)) != len(names) or any(uses.get(n, 0) > 1 for n in names):
            result.skipped.append(pair.id)
            continue
        pairs.append(pair)

    arrays = {n: numpy_helper.to_array(inits[n]).astype(np.float64)
              for p in pairs for n in (p.producer_weight.name, p.consumer_weight.name, p.producer_bias) if n}
    original = {n: a.copy() for n, a in arrays.items()}
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t!r}")
    if t < 1.0 and batches is None:
        raise ValueError("activation-aware CLE (t < 1) needs calibration batches")

    def ranges(p: CLEPairSpec) -> tuple[np.ndarray, np.ndarray]:
        return (
            _channel_range(arrays[p.producer_weight.name], p.producer_weight, p.channels),
            _channel_range(arrays[p.consumer_weight.name], p.consumer_weight, p.channels),
        )

    before = [_balance(*ranges(p)) for p in pairs]
    total = [np.ones(p.channels) for p in pairs]
    lo, hi = 1.0 / max_scale, max_scale

    result.converged = not pairs
    for it in range(1, iterations + 1 if pairs else 1):
        result.iterations = it
        biggest = 0.0
        for k, p in enumerate(pairs):
            new = np.clip(total[k] * cle_scales(*ranges(p)), lo, hi)
            step = new / total[k]
            total[k] = new
            biggest = max(biggest, float(np.abs(step - 1.0).max()))
            pw = arrays[p.producer_weight.name]
            arrays[p.producer_weight.name] = _scale(pw, p.producer_weight, step, divide=False)
            if p.producer_bias:
                arrays[p.producer_bias] = arrays[p.producer_bias] * step
            cw = arrays[p.consumer_weight.name]
            arrays[p.consumer_weight.name] = _scale(cw, p.consumer_weight, step, divide=True)
        if biggest <= threshold:
            result.converged = True
            break

    if t < 1.0 and pairs:
        from anneal.core.equalize import DEFAULT_SLACK, channel_ranges, choose_scales

        outs = [p.producer.output[0] for p in pairs]
        ranges_act = channel_ranges(model, sorted(set(outs)), batches)
        arrays = {n: a.copy() for n, a in original.items()}
        for k, p in enumerate(pairs):
            s_act = choose_scales([ranges_act[p.producer.output[0]]], slack=DEFAULT_SLACK, allow_negative=False,
                                  max_scale=max_scale).astype(np.float64)
            total[k] = np.clip(s_act ** (1.0 - t) * total[k] ** t, lo, hi)
            arrays[p.producer_weight.name] = _scale(arrays[p.producer_weight.name], p.producer_weight, total[k], divide=False)
            if p.producer_bias:
                arrays[p.producer_bias] = arrays[p.producer_bias] * total[k]
            arrays[p.consumer_weight.name] = _scale(arrays[p.consumer_weight.name], p.consumer_weight, total[k], divide=True)
    for name, arr in arrays.items():
        inits[name].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name))
    for k, p in enumerate(pairs):
        s = total[k]
        result.pairs.append(
            CLEPair(
                producer=p.producer.name,
                consumer=p.consumer.name,
                via=list(p.via),
                kind=p.kind,
                channels=p.channels,
                scale_min=float(s.min()),
                scale_max=float(s.max()),
                channels_clamped=int(((s <= lo * (1 + 1e-9)) | (s >= hi * (1 - 1e-9))).sum()),
                balance_before=before[k],
                balance_after=_balance(*ranges(p)),
                clips_converted=[clip.name for clip, _ in p.ceilings],
            )
        )
    _rewrite_clips(model, pairs, total)
    result.clips_converted = [n for pair in result.pairs for n in pair.clips_converted]

    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    if check_batch is not None:
        from anneal.core.equalize import _max_output_change

        result.max_abs_output_change = _max_output_change(src, dst, check_batch)
    return result


def _rewrite_clips(model, pairs: list[CLEPairSpec], total: list[np.ndarray]) -> None:
    """Replace each pair's Clip(0, M) by ``Relu -> Min(., M * s)``; drop orphaned bound constants.

    Clip(0, M)(x) * s == Min(Relu(s * x), M * s) for s > 0, channel by channel.
    """
    from onnx import helper, numpy_helper

    g = model.graph
    names = {n.name for n in g.node} | {t for n in g.node for t in n.output}
    names |= {i.name for i in g.initializer} | {i.name for i in g.input}

    def fresh(base: str) -> str:
        name, k = base, 1
        while name in names:
            name, k = f"{base}_{k}", k + 1
        names.add(name)
        return name

    replace: dict[str, list] = {}
    bound_inputs: set[str] = set()
    for k, p in enumerate(pairs):
        shape = [p.channels] if p.rank == 2 else [1, p.channels] + [1] * (p.rank - 2)
        for clip, hi in p.ceilings:
            ceiling = fresh(f"{clip.name}_ceiling")
            relu_out = fresh(f"{clip.output[0]}_relu")
            g.initializer.append(
                numpy_helper.from_array((hi * total[k]).astype(np.float32).reshape(shape), ceiling)
            )
            replace[clip.output[0]] = [
                helper.make_node("Relu", [clip.input[0]], [relu_out], name=fresh(f"{clip.name}_relu")),
                helper.make_node("Min", [relu_out, ceiling], [clip.output[0]], name=fresh(f"{clip.name}_min")),
            ]
            bound_inputs.update(i for i in list(clip.input)[1:] if i)
    if not replace:
        return
    nodes = []
    for n in g.node:
        if n.op_type == "Clip" and n.output and n.output[0] in replace:
            nodes.extend(replace[n.output[0]])
        else:
            nodes.append(n)
    used = {i for n in nodes for i in n.input} | {o.name for o in g.output}
    orphans = bound_inputs - used
    nodes = [n for n in nodes if not (n.op_type == "Constant" and n.output and n.output[0] in orphans)]
    graph_inputs = {i.name for i in g.input}
    keep = [i for i in g.initializer if i.name not in orphans or i.name in graph_inputs]

    def refill(field, items) -> None:
        copies = []
        for item in items:  # copy first: clearing the field may detach the originals
            c = type(item)()
            c.CopyFrom(item)
            copies.append(c)
        del field[:]
        field.extend(copies)

    refill(g.node, nodes)
    refill(g.initializer, keep)
