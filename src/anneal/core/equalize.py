"""Exact channel equalisation across gated activations, before static INT8.

The failure this fixes. Static INT8 gives each activation tensor *one* scale and zero point,
shared by all of its channels. After batch-norm folding, the channels of an expand or stem
convolution can differ in range by 100x or more. On EfficientNet-B0's stem, one channel spans
169 while another lives entirely in SiLU's negative lobe (-0.27 to -0.10) and is left with
half a quantization level. The depthwise convolution that follows cannot average this error
across channels, and its folded weights are largest on exactly those small channels, so it
amplifies the rounding by 10-24 dB. Repeated in all 16 blocks, this takes EfficientNet-B0
from 74% to 22% top-1 on every CPU. MobileNetV3's Hardswish blocks fail the same way.

The rewrite. For the pattern

    x = ConvA(...);   y = x * gate(x);   z = DepthwiseConvB(y)        gate: Sigmoid, HardSigmoid

scale output channel c of A by s_c, feed the gate x / s_c (one extra element-wise Mul), and
divide channel c of B by s_c:

    x' = s * x,   y' = x' * gate(x' / s) = s * y,   B'(y') = B(y)

Nothing downstream of B changes, and the float model computes the same function (up to
rounding). Because the weights of A and B are quantized *per channel*, the per-channel weight
scales absorb s exactly: weight quantization is untouched. Only the activation tensors x and
y change, and s is chosen so that their channels fill the shared range evenly.

``s_c`` may be negative. SiLU's output is at least -0.278, so a channel that lives below zero
cannot be scaled up without dragging the whole tensor's range down; mirrored (s < 0) it lands
in the large positive side instead. This is exact for gated activations because the gate sees
x'/s = x. It is not exact for ReLU, which is also handled (``y = ReLU(x)``, s > 0 only, no gate
Mul needed: s * ReLU(x) = ReLU(s * x)) — that case is the classic cross-layer equalisation of
Nagel et al. (2019). Standard cross-layer equalisation needs ``f(s*x) = s*f(x)``, which gated
activations such as SiLU and Hardswish do not satisfy; HPTQ (Habi et al., 2021) names the missing
equalisation for Swish as a likely cause of EfficientNet's INT8 loss. The gate-side ``1/s``
removes that requirement at the cost of one element-wise Mul per block.

Choosing s. Each tensor's quantization range [lo, hi] (always including 0) may not grow at the
top; its bottom may extend by ``slack`` of the range. Within that budget every channel is scaled
up as far as it will go, in whichever sign lets it go further. Scaling up only ever *adds*
levels to a channel; the budget bounds what the other channels lose (at most ``slack``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

GATE_OPS = ("Sigmoid", "HardSigmoid")
#: Fused gated activations, decomposed into x * gate(x) when a site is rewritten.
FUSED_GATED_OPS = ("HardSwish",)
#: Name prefix of the Mul inserted before each gate; `gate_nodes` lists them for exclusion.
GATE_MUL_PREFIX = "anneal_eq_gate_mul_"
DEFAULT_SLACK = 0.1
DEFAULT_MAX_SCALE = 1e3


@dataclass
class _Site:
    kind: str  # "gated" or "relu"
    conv_a: Any
    act: Any  # the Mul (gated) or the Relu
    gate: Any | None
    conv_b: Any
    x: str  # A's output
    y: str  # the activation's output, B's input


@dataclass
class EqualisedSite:
    kind: str
    producer: str
    consumer: str
    gate: str | None
    channels: int
    scale_median: float
    scale_max: float
    channels_scaled: int  # |s| > 1.5
    channels_mirrored: int  # s < 0
    #: Median quantization levels per channel of the activation, before and after.
    levels_before: float
    levels_after: float

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class EqualisationResult:
    sites: list[EqualisedSite] = field(default_factory=list)
    #: Nodes on the gate branch (inserted Mul + gate op). Excluding them from quantization
    #: keeps the gate's input in float, which an imbalanced 8-bit tensor would otherwise be.
    gate_nodes: list[str] = field(default_factory=list)
    max_abs_logit_change: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "sites": len(self.sites),
            "by_kind": {k: sum(s.kind == k for s in self.sites) for k in ("gated", "relu")},
            "channels_mirrored": sum(s.channels_mirrored for s in self.sites),
            "median_levels_before": float(np.median([s.levels_before for s in self.sites])) if self.sites else None,
            "median_levels_after": float(np.median([s.levels_after for s in self.sites])) if self.sites else None,
            "max_abs_logit_change": self.max_abs_logit_change,
        }


# ---------------------------------------------------------------------------
# Scale choice (pure numpy; the part worth testing in isolation)
# ---------------------------------------------------------------------------


def channel_bound(lo: np.ndarray, hi: np.ndarray, sign: float, slack: float) -> np.ndarray:
    """Largest |s| per channel keeping ``s * channel`` inside the tensor's range budget.

    The budget is the tensor's current quantization range with its top fixed and its bottom
    extended by ``slack`` of the range. Channels with no extent get ``inf`` (unconstrained).
    """
    t_hi, t_lo = max(float(hi.max()), 0.0), min(float(lo.min()), 0.0)
    bottom = t_lo - slack * (t_hi - t_lo)
    c_lo, c_hi = (lo, hi) if sign > 0 else (-hi, -lo)
    bound = np.full(lo.shape, np.inf)
    pos, neg = c_hi > 0, c_lo < 0
    if t_hi > 0:
        bound[pos] = np.minimum(bound[pos], t_hi / c_hi[pos])
    else:
        bound[pos] = 1.0
    if bottom < 0:
        bound[neg] = np.minimum(bound[neg], bottom / c_lo[neg])
    else:
        bound[neg] = 1.0
    return bound


def choose_scales(
    tensors: list[tuple[np.ndarray, np.ndarray]],
    *,
    slack: float = DEFAULT_SLACK,
    allow_negative: bool = True,
    max_scale: float = DEFAULT_MAX_SCALE,
) -> np.ndarray:
    """Per-channel scales for channels that appear, multiplied by s, in every tensor given.

    ``tensors`` holds (per-channel min, per-channel max) for each tensor the scale multiplies
    (for a site, x and y). Returns s with |s| >= 1: channels are only ever scaled *up*.
    """

    def best(sign: float) -> np.ndarray:
        b = np.minimum.reduce([channel_bound(lo, hi, sign, slack) for lo, hi in tensors])
        b[~np.isfinite(b)] = 1.0
        return np.clip(b, 1.0, max_scale)

    s = best(1.0)
    if allow_negative:
        s_neg = best(-1.0)
        s = np.where(s_neg > s, -s_neg, s)
    return s.astype(np.float32)


def median_levels(lo: np.ndarray, hi: np.ndarray, levels: int = 255) -> float:
    """Median over channels of how many 8-bit steps a channel spans under a shared scale."""
    t_hi, t_lo = max(float(hi.max()), 0.0), min(float(lo.min()), 0.0)
    if t_hi - t_lo <= 0:
        return 0.0
    step = (t_hi - t_lo) / levels
    return float(np.median((hi - lo) / step))


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def _attr(node, name: str, default=None):
    from onnx import helper

    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def find_sites(model) -> list[_Site]:
    """Conv -> (gated activation | ReLU) -> depthwise Conv chains that can be equalised exactly.

    Every intermediate tensor must have exactly the consumers the rewrite accounts for; a
    tensor that also feeds a residual Add or a second branch is left alone.
    """
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def conv_with_const_weights(node) -> bool:
        return node is not None and node.op_type == "Conv" and len(node.input) > 1 and node.input[1] in inits

    def depthwise_only_consumer(tensor: str):
        outs = consumers.get(tensor, [])
        if len(outs) != 1 or not conv_with_const_weights(outs[0]):
            return None
        b = outs[0]
        w = inits[b.input[1]]
        if len(w.dims) < 2 or w.dims[1] != 1 or _attr(b, "group", 1) != w.dims[0]:
            return None
        return b

    sites: list[_Site] = []
    for node in g.node:
        if node.op_type == "Mul" and len(node.input) == 2:
            for x, gated in ((node.input[0], node.input[1]), (node.input[1], node.input[0])):
                gate = producer.get(gated)
                if gate is None or gate.op_type not in GATE_OPS or gate.input[0] != x:
                    continue
                a = producer.get(x)
                if not conv_with_const_weights(a):
                    continue
                if sorted(c.name for c in consumers.get(x, [])) != sorted([gate.name, node.name]):
                    continue
                if len(consumers.get(gate.output[0], [])) != 1:
                    continue
                b = depthwise_only_consumer(node.output[0])
                if b is not None:
                    sites.append(_Site("gated", a, node, gate, b, x, node.output[0]))
                break
        elif node.op_type == "HardSwish":
            # The fused form of x * HardSigmoid(x); decomposed when rewritten.
            x = node.input[0]
            a = producer.get(x)
            if not conv_with_const_weights(a) or len(consumers.get(x, [])) != 1:
                continue
            b = depthwise_only_consumer(node.output[0])
            if b is not None:
                sites.append(_Site("gated", a, node, None, b, x, node.output[0]))
        elif node.op_type == "Relu":
            x = node.input[0]
            a = producer.get(x)
            if not conv_with_const_weights(a) or len(consumers.get(x, [])) != 1:
                continue
            b = depthwise_only_consumer(node.output[0])
            if b is not None:
                sites.append(_Site("relu", a, node, None, b, x, node.output[0]))
    return sites


def _name_unnamed_nodes(model) -> None:
    """Exporters often leave nodes unnamed; quantizer exclusions work by name."""
    taken = {n.name for n in model.graph.node if n.name}
    for i, node in enumerate(model.graph.node):
        if not node.name:
            name = f"anneal_node_{i}"
            while name in taken:
                name += "_"
            node.name = name
            taken.add(name)


def channel_ranges(
    model, tensors: list[str], batches: Iterable[np.ndarray]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-channel (min, max) of NCHW tensors over the given input batches, in float."""
    import onnx
    import onnxruntime as ort
    from onnx import helper

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    existing = {o.name for o in probe.graph.output}
    probe.graph.output.extend(
        [helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None) for t in tensors if t not in existing]
    )
    opts = ort.SessionOptions()
    # Nothing may be fused away: every requested tensor must exist as computed.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(probe.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    lo: dict[str, np.ndarray] = {}
    hi: dict[str, np.ndarray] = {}
    seen = False
    for batch in batches:
        seen = True
        for t, v in zip(tensors, session.run(tensors, {input_name: batch})):
            axes = (0,) + tuple(range(2, v.ndim))
            mn, mx = v.min(axis=axes), v.max(axis=axes)
            lo[t] = mn if t not in lo else np.minimum(lo[t], mn)
            hi[t] = mx if t not in hi else np.maximum(hi[t], mx)
    if not seen:
        raise ValueError("equalisation needs calibration batches; none were given")
    return {t: (lo[t].astype(np.float64), hi[t].astype(np.float64)) for t in tensors}


def equalise(
    src: Path,
    dst: Path,
    batches: Iterable[np.ndarray],
    *,
    slack: float = DEFAULT_SLACK,
    allow_negative: bool = True,
    max_scale: float = DEFAULT_MAX_SCALE,
    check_batch: np.ndarray | None = None,
) -> EqualisationResult:
    """Write an equalised copy of ``src`` to ``dst`` and describe what changed.

    ``batches`` are calibration inputs (the ranges are measured on them). With ``check_batch``
    the float outputs of both models are compared on it and the largest change recorded.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(src))
    _name_unnamed_nodes(model)
    sites = find_sites(model)
    result = EqualisationResult()
    if not sites:
        onnx.save(model, str(dst))
        return result

    ranges = channel_ranges(model, sorted({t for s in sites for t in (s.x, s.y)}), batches)
    g = model.graph
    inits = {i.name: i for i in g.initializer}

    def rescale(name: str, factor: np.ndarray, divide: bool = False) -> None:
        arr = numpy_helper.to_array(inits[name]).astype(np.float64)
        shape = (-1,) + (1,) * (arr.ndim - 1)
        arr = arr / factor.reshape(shape) if divide else arr * factor.reshape(shape)
        inits[name].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name))

    shared: set[str] = set()
    for k, site in enumerate(sites):
        # A weight shared by two convs cannot be rescaled for one of them.
        names = [site.conv_a.input[1], site.conv_b.input[1]]
        if len(site.conv_a.input) > 2 and site.conv_a.input[2]:
            names.append(site.conv_a.input[2])
        if shared & set(names):
            continue
        shared.update(names)

        lo_x, hi_x = ranges[site.x]
        lo_y, hi_y = ranges[site.y]
        s = choose_scales(
            [(lo_x, hi_x), (lo_y, hi_y)],
            slack=slack,
            allow_negative=allow_negative and site.kind == "gated",
            max_scale=max_scale,
        )
        s64 = s.astype(np.float64)
        rescale(site.conv_a.input[1], s64)
        if len(site.conv_a.input) > 2 and site.conv_a.input[2]:
            rescale(site.conv_a.input[2], s64)
        rescale(site.conv_b.input[1], s64, divide=True)

        if site.kind == "gated" and site.gate is None:
            # HardSwish(x) = x * HardSigmoid(x) with alpha 1/6, beta 1/2: split it so the gate
            # can see x'/s.
            gate_out = f"anneal_eq_gate_{k}"
            site.gate = helper.make_node("HardSigmoid", [site.x], [gate_out],
                                         name=f"anneal_eq_hardsigmoid_{k}", alpha=1.0 / 6, beta=0.5)
            mul = helper.make_node("Mul", [site.x, gate_out], [site.y], name=site.act.name or f"anneal_eq_hswish_{k}")
            idx = list(g.node).index(site.act)
            g.node.remove(site.act)
            g.node.insert(idx, mul)
            g.node.insert(idx, site.gate)
            # protobuf copies on insert: keep handles to the nodes that are in the graph.
            site.gate, site.act = g.node[idx], g.node[idx + 1]
        if site.kind == "gated":
            inv = f"anneal_eq_inv_{k}"
            unscaled = f"anneal_eq_unscaled_{k}"
            mul_name = f"{GATE_MUL_PREFIX}{k}"
            g.initializer.append(
                numpy_helper.from_array((1.0 / s64).reshape(1, -1, 1, 1).astype(np.float32), inv)
            )
            g.node.insert(
                list(g.node).index(site.gate),
                helper.make_node("Mul", [site.x, inv], [unscaled], name=mul_name),
            )
            site.gate.input[0] = unscaled
            result.gate_nodes += [mul_name, site.gate.name]

        lo_y2 = np.minimum(s64 * lo_y, s64 * hi_y)
        hi_y2 = np.maximum(s64 * lo_y, s64 * hi_y)
        result.sites.append(
            EqualisedSite(
                kind=site.kind,
                producer=site.conv_a.name,
                consumer=site.conv_b.name,
                gate=site.gate.name if site.gate is not None else None,
                channels=int(s.shape[0]),
                scale_median=float(np.median(np.abs(s))),
                scale_max=float(np.abs(s).max()),
                channels_scaled=int((np.abs(s) > 1.5).sum()),
                channels_mirrored=int((s < 0).sum()),
                levels_before=median_levels(lo_y, hi_y),
                levels_after=median_levels(lo_y2, hi_y2),
            )
        )

    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))

    if check_batch is not None:
        result.max_abs_logit_change = _max_output_change(src, dst, check_batch)
    return result


def _max_output_change(a: Path, b: Path, batch: np.ndarray) -> float:
    import onnxruntime as ort

    def run(path: Path) -> np.ndarray:
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return s.run(None, {s.get_inputs()[0].name: batch})[0]

    return float(np.abs(run(a) - run(b)).max())
