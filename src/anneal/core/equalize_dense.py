"""Equalisation into *dense* consumers, with the trade-off measured per site.

:mod:`anneal.core.equalize` rescales channels across ``Conv -> x*gate(x) -> depthwise Conv``,
where the depthwise conv's per-channel weight scales absorb the rescale exactly. When the
consumer B is dense (a 1x1 group-1 Conv, or a MatMul) that is no longer free: B is quantized
per *output* channel, so dividing input channel c of B by s_c can cost that channel's weights
precision while the activation gains levels.

So the two errors are measured rather than assumed. For each site the scale is
``s = sign(s_budget) * |s_budget| ** alpha``, where ``s_budget`` is the full equalisation of
:func:`anneal.core.equalize.choose_scales` and alpha is chosen from ``DENSE_ALPHAS`` to minimise
the simulated INT8 error of B's output on captured calibration activations: activations
per-tensor asymmetric uint8, weights per-output-channel symmetric int8, as the quantizer will
do. alpha = 0 means "leave this site alone", so a site where the rewrite would hurt is not
rewritten.

Patterns covered:

    x = Conv(...) | MatMul(...) [+ bias]
    y = x * gate(x) [* scalar ...]      gate: any element-wise subgraph fed only by x
                                        (Sigmoid, HardSigmoid, GELU's Erf form, ...)
    z = Conv1x1(y) | MatMul(y, W)       dense, group 1

which is ConvNeXt's Linear -> GELU -> Linear (channels last), EfficientNetV2's Fused-MBConv
(3x3 Conv -> SiLU -> 1x1 Conv) and squeeze-excite MLPs. The float model is unchanged either
way: the gate sees x'/s = x.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anneal.core.equalize import (
    DEFAULT_MAX_SCALE,
    DEFAULT_SLACK,
    GATE_MUL_PREFIX,
    _attr,
    _max_output_change,
    _name_unnamed_nodes,
    choose_scales,
)

DENSE_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
#: Element-wise ops a gate subgraph may contain.
GATE_SUBGRAPH_OPS = {"Sigmoid", "HardSigmoid", "Erf", "Tanh", "Div", "Mul", "Add", "Sub"}
SAMPLE_ROWS = 4096


@dataclass
class DenseSite:
    a: Any  # Conv or MatMul
    a_weight: str
    a_weight_axis: int  # output-channel axis of A's weight
    a_bias: str | None
    x: str
    gate_nodes: list
    gate_entries: list  # gate nodes that read x
    mul: Any
    y: str  # B's input, after any scalar multiplies
    b: Any
    b_weight: str
    b_in_axis: int  # input-channel axis of B's weight
    channel_axis: int  # activation channel axis: 1 (NCHW) or -1 (channels last)


@dataclass
class DenseEqualisedSite:
    producer: str
    consumer: str
    channels: int
    alpha: float
    #: Simulated relative INT8 error of the consumer's output, without and with the rewrite.
    error_before: float
    error_after: float
    channels_mirrored: int

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Finding sites
# ---------------------------------------------------------------------------


def _constants(model) -> set[str]:
    names = {i.name for i in model.graph.initializer}
    return names | {o for n in model.graph.node if n.op_type == "Constant" for o in n.output}


def _const_value(model, name: str):
    from onnx import numpy_helper

    for i in model.graph.initializer:
        if i.name == name:
            return numpy_helper.to_array(i)
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    return None


def find_dense_sites(model) -> list[DenseSite]:
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    consts = _constants(model)
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def gate_subgraph(x: str, g_out: str):
        """The nodes computing g_out from x and constants alone, or None."""
        nodes: list = []
        stack = [g_out]
        while stack:
            t = stack.pop()
            n = producer.get(t)
            if n is None or n.op_type not in GATE_SUBGRAPH_OPS:
                return None
            if any(n is m for m in nodes):
                continue
            nodes.append(n)
            if len(nodes) > 8:
                return None
            stack += [i for i in n.input if i != x and i not in consts]
        return nodes if any(x in n.input for n in nodes) else None

    def producer_of(x: str):
        n = producer.get(x)
        if n is None:
            return None
        if n.op_type == "Conv" and len(n.input) > 1 and n.input[1] in inits:
            bias = n.input[2] if len(n.input) > 2 and n.input[2] else None
            return n, n.input[1], 0, bias, 1
        if n.op_type == "Add":
            const = [i for i in n.input if i in inits]
            other = [i for i in n.input if i not in inits]
            if len(const) == 1 and len(other) == 1 and len(inits[const[0]].dims) == 1:
                mm = producer.get(other[0])
                if (mm is not None and mm.op_type == "MatMul" and mm.input[1] in inits
                        and len(consumers.get(other[0], [])) == 1):
                    return mm, mm.input[1], 1, const[0], -1
        if n.op_type == "MatMul" and n.input[1] in inits:
            return n, n.input[1], 1, None, -1
        return None

    def dense_consumer(y: str, channel_axis: int):
        outs = consumers.get(y, [])
        if len(outs) != 1:
            return None
        b = outs[0]
        if b.input[0] != y or len(b.input) < 2 or b.input[1] not in inits:
            return None
        dims = list(inits[b.input[1]].dims)
        if b.op_type == "Conv" and channel_axis == 1:
            if _attr(b, "group", 1) != 1 or dims[2:] != [1] * (len(dims) - 2):
                return None
            return b, 1
        if b.op_type == "MatMul" and channel_axis == -1 and len(dims) == 2:
            return b, 0
        return None

    sites: list[DenseSite] = []
    for mul in g.node:
        if mul.op_type != "Mul" or len(mul.input) != 2:
            continue
        for x, g_out in ((mul.input[0], mul.input[1]), (mul.input[1], mul.input[0])):
            if x in consts or g_out in consts:
                continue
            gate = gate_subgraph(x, g_out)
            if gate is None:
                continue
            prod = producer_of(x)
            if prod is None:
                continue
            a, a_w, a_axis, a_bias, ch_axis = prod
            allowed = {n.name for n in gate} | {mul.name}
            if any(c.name not in allowed for c in consumers.get(x, [])):
                continue
            if any(c.name not in allowed for n in gate for c in consumers.get(n.output[0], [])):
                continue
            y = mul.output[0]
            while True:  # scalar multiplies (GELU's 0.5) commute with per-channel scaling
                outs = consumers.get(y, [])
                if len(outs) != 1 or outs[0].op_type not in ("Mul", "Div"):
                    break
                others = [i for i in outs[0].input if i != y]
                if len(others) != 1 or others[0] not in consts:
                    break
                val = _const_value(model, others[0])
                if val is None or np.size(val) != 1 or (outs[0].op_type == "Div" and outs[0].input[0] != y):
                    break
                y = outs[0].output[0]
            found = dense_consumer(y, ch_axis)
            if found is None:
                continue
            b, b_axis = found
            entries = [n for n in gate if x in n.input]
            sites.append(DenseSite(a, a_w, a_axis, a_bias, x, gate, entries, mul, y, b,
                                   b.input[1], b_axis, ch_axis))
            break
    return sites


# ---------------------------------------------------------------------------
# The measured trade-off
# ---------------------------------------------------------------------------


def fake_quant_activation(v: np.ndarray) -> np.ndarray:
    """Per-tensor asymmetric uint8, min/max range including zero."""
    lo, hi = min(float(v.min()), 0.0), max(float(v.max()), 0.0)
    step = (hi - lo) / 255 or 1.0
    zp = np.round(-lo / step)
    return (np.clip(np.round(v / step) + zp, 0, 255) - zp) * step


def fake_quant_columns(w: np.ndarray) -> np.ndarray:
    """Per-output-channel (column of a (C_in, C_out) matrix) symmetric int8."""
    scale = np.abs(w).max(axis=0, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    return np.clip(np.round(w / scale), -127, 127) * scale


def site_error(y: np.ndarray, w: np.ndarray, s: np.ndarray) -> float:
    """Relative INT8 error of ``y @ w`` after moving scale ``s`` from w's rows into y."""
    ref = y @ w
    out = fake_quant_activation(y * s) @ fake_quant_columns(w / s[:, None])
    return float(np.sum((out - ref) ** 2) / max(float(np.sum(ref ** 2)), 1e-30))


def _rows(v: np.ndarray, axis: int, rng) -> np.ndarray:
    v = np.moveaxis(v, axis, -1).reshape(-1, v.shape[axis])
    if len(v) > SAMPLE_ROWS:
        v = v[rng.choice(len(v), SAMPLE_ROWS, replace=False)]
    return v.astype(np.float64)


def equalise_dense(
    src: Path,
    dst: Path,
    batches: list[np.ndarray],
    *,
    slack: float = DEFAULT_SLACK,
    max_scale: float = DEFAULT_MAX_SCALE,
    sample_images: int = 8,
) -> tuple[list[DenseEqualisedSite], list[str], float | None]:
    """Rewrite the dense sites where it measurably lowers the consumer's INT8 error.

    Returns (sites rewritten, gate nodes to keep out of quantization, largest float output
    change on the first batch). ``batches`` is read twice, so it must be a list.
    """
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    model = onnx.load(str(src))
    _name_unnamed_nodes(model)
    sites = find_dense_sites(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not sites:
        onnx.save(model, str(dst))
        return [], [], None

    tensors = sorted({t for s in sites for t in (s.x, s.y)})
    axes = {s.x: s.channel_axis for s in sites} | {s.y: s.channel_axis for s in sites}
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    probe.graph.output.extend([helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None) for t in tensors])
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(probe.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    lo: dict[str, np.ndarray] = {}
    hi: dict[str, np.ndarray] = {}
    samples: dict[str, np.ndarray] = {}
    ys = {s.y for s in sites}
    rng = np.random.default_rng(0)
    for i, batch in enumerate(batches):
        for t, v in zip(tensors, sess.run(tensors, {inp: batch})):
            ax = axes[t] % v.ndim
            red = tuple(d for d in range(v.ndim) if d != ax)
            mn, mx = v.min(axis=red), v.max(axis=red)
            lo[t] = mn if t not in lo else np.minimum(lo[t], mn)
            hi[t] = mx if t not in hi else np.maximum(hi[t], mx)
            if i == 0 and t in ys:
                samples[t] = _rows(v[:sample_images], ax, rng)
    del sess

    g = model.graph
    inits = {i.name: i for i in g.initializer}

    def scale_along(name: str, axis: int, factor: np.ndarray, divide: bool = False) -> None:
        arr = numpy_helper.to_array(inits[name]).astype(np.float64)
        shape = [1] * arr.ndim
        shape[axis] = -1
        arr = arr / factor.reshape(shape) if divide else arr * factor.reshape(shape)
        inits[name].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name))

    done: list[DenseEqualisedSite] = []
    gate_nodes: list[str] = []
    used: set[str] = set()
    for k, site in enumerate(sites):
        names = {site.a_weight, site.b_weight} | ({site.a_bias} if site.a_bias else set())
        if used & names:
            continue
        s_budget = choose_scales(
            [(lo[site.x].astype(np.float64), hi[site.x].astype(np.float64)),
             (lo[site.y].astype(np.float64), hi[site.y].astype(np.float64))],
            slack=slack, allow_negative=True, max_scale=max_scale,
        ).astype(np.float64)
        wb = numpy_helper.to_array(inits[site.b_weight]).astype(np.float64)
        w2 = wb.reshape(wb.shape[0], wb.shape[1]).T if site.b.op_type == "Conv" else wb
        errors = {a: site_error(samples[site.y], w2, np.sign(s_budget) * np.abs(s_budget) ** a)
                  for a in DENSE_ALPHAS}
        alpha = min(errors, key=lambda a: (errors[a], a))
        if alpha == 0.0:
            continue
        used |= names
        s = np.sign(s_budget) * np.abs(s_budget) ** alpha

        scale_along(site.a_weight, site.a_weight_axis, s)
        if site.a_bias:
            scale_along(site.a_bias, 0, s)
        scale_along(site.b_weight, site.b_in_axis, s, divide=True)

        inv, unscaled, mul_name = f"anneal_eqd_inv_{k}", f"anneal_eqd_unscaled_{k}", f"{GATE_MUL_PREFIX}d{k}"
        shape = (1, -1, 1, 1) if site.channel_axis == 1 else (-1,)
        g.initializer.append(numpy_helper.from_array((1.0 / s).reshape(shape).astype(np.float32), inv))
        entry_names = {n.name for n in site.gate_entries}
        first = min(i for i, n in enumerate(g.node) if n.name in entry_names)
        g.node.insert(first, helper.make_node("Mul", [site.x, inv], [unscaled], name=mul_name))
        for node in g.node:
            if node.name in entry_names:
                for j, i in enumerate(node.input):
                    if i == site.x:
                        node.input[j] = unscaled
        gate_nodes += [mul_name] + [n.name for n in site.gate_nodes]
        done.append(DenseEqualisedSite(site.a.name, site.b.name, int(len(s)), float(alpha),
                                       errors[0.0], errors[alpha], int((s < 0).sum())))

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    change = _max_output_change(src, dst, batches[0]) if done else None
    return done, gate_nodes, change
