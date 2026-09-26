"""Exact channel equalisation across a Concat, for accelerators that give it one shared scale.

The failure. Accelerators such as TI's TIDL and Qualcomm's Hexagon HTP quantize a Concat's inputs
and output with one scale. A U-Net concatenates encoder skip features (Conv -> ReLU) with the
decoder's transposed-convolution output, whose range is far larger; under the shared scale the
skip channels keep a handful of levels. Measured on a Carvana U-Net (examples/segmentation/
unet_study.py): car IoU against FP32 0.993 with per-input scales, 0.082 with the shared one.

The rewrite. For every channel c of the Concat, find the Conv or ConvTranspose that produces it
through ops that commute with a positive per-channel scale (ReLU, LeakyReLU, MaxPool, average
pooling, zero Pad, Resize, Identity). Scale that producer's output channel (weights and bias) by
s_c > 0; the scale then travels through those ops into the Concat. Every Conv that consumes a
scaled tensor (the Concat's consumer, and e.g. the next encoder stage that also reads the skip
tensor through a MaxPool) divides the matching input channel by s_c. The float model is unchanged.

Choosing s. The Concat output's per-channel ranges are measured on calibration images and
:func:`anneal.core.equalize.choose_scales` fills the shared range: every channel is scaled up as
far as the tensor's range allows (positive scales only; ReLU does not commute with a sign flip).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from anneal.core.equalize import (
    DEFAULT_MAX_SCALE,
    DEFAULT_SLACK,
    _attr,
    _max_output_change,
    _name_unnamed_nodes,
    choose_scales,
)

#: Ops y = f(x) with f(s * x) = s * f(x) per channel for s > 0.
COMMUTING_OPS = {"Relu", "LeakyRelu", "MaxPool", "AveragePool", "GlobalAveragePool", "Pad", "Resize",
                 "Identity", "Upsample", "Dropout"}
PRODUCERS = {"Conv": 0, "ConvTranspose": 1}  # op -> output-channel axis of its weight


@dataclass
class ConcatSite:
    concat: Any
    #: per input: (producer node, channel count) in Concat order
    branches: list[tuple[Any, int]]
    #: (consumer node, weight input-channel axis, channel offset into the scale vector of the branch
    #: or of the Concat, which branch it follows (-1 = the Concat output))
    compensations: list[tuple[Any, int, int, int]] = field(default_factory=list)


def _pad_is_zero_constant(model, node) -> bool:
    mode = _attr(node, "mode", b"constant")
    mode = mode.decode() if isinstance(mode, bytes) else mode
    if mode != "constant":
        return False
    if len(node.input) > 2 and node.input[2]:
        inits = {i.name: i for i in model.graph.initializer}
        from onnx import numpy_helper

        if node.input[2] not in inits or np.any(numpy_helper.to_array(inits[node.input[2]]) != 0):
            return False
    return True


def find_concat_sites(model) -> list[ConcatSite]:
    """Channel Concats whose every input traces back to a scalable producer, and whose scaled
    tensors are consumed only by commuting ops, the Concat, or compensating convolutions."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    graph_outputs = {o.name for o in g.output}

    def commutes(node) -> bool:
        if node.op_type not in COMMUTING_OPS:
            return False
        if node.op_type == "LeakyRelu" and _attr(node, "alpha", 0.01) < 0:
            return False
        if node.op_type == "Pad" and not _pad_is_zero_constant(model, node):
            return False
        return True

    def trace_back(tensor: str):
        node = producer.get(tensor)
        while node is not None and commutes(node):
            node = producer.get(node.input[0])
        if node is not None and node.op_type in PRODUCERS and len(node.input) > 1 and node.input[1] in inits:
            if node.op_type == "Conv" and _attr(node, "group", 1) != 1:
                return None
            return node
        return None

    def compensable(node, tensor: str):
        """A Conv/ConvTranspose reading `tensor` as its data input: weight input-channel axis."""
        if node.op_type not in PRODUCERS or node.input[0] != tensor or node.input[1] not in inits:
            return None
        w = inits[node.input[1]]
        if node.op_type == "Conv":
            group = _attr(node, "group", 1)
            if group == 1:
                return 1
            if len(w.dims) > 1 and w.dims[1] == 1 and group == w.dims[0]:
                return 0  # depthwise: input channel c is output channel c
            return None
        return 0 if _attr(node, "group", 1) == 1 else None

    sites = []
    for cat in g.node:
        if cat.op_type != "Concat" or _attr(cat, "axis", 1) not in (1, -3):
            continue
        branches = []
        for t in cat.input:
            p = trace_back(t)
            if p is None:
                break
            ch = inits[p.input[1]].dims[PRODUCERS[p.op_type]] * (_attr(p, "group", 1) if p.op_type == "ConvTranspose" else 1)
            branches.append((p, ch))
        else:
            if len(branches) < 2 or len({b[0].name for b in branches}) != len(branches):
                continue
            site = ConcatSite(cat, branches)
            ok = True
            # forward from each producer (and from the Concat): every consumer must be accounted for
            starts = [(b[0].output[0], i) for i, b in enumerate(branches)] + [(cat.output[0], -1)]
            for start, which in starts:
                stack, seen = [start], set()
                while stack and ok:
                    t = stack.pop()
                    if t in seen:
                        continue
                    seen.add(t)
                    if t in graph_outputs:
                        ok = False
                        break
                    for c in consumers.get(t, []):
                        if c.name == cat.name and which >= 0:
                            continue
                        if commutes(c) and c.input[0] == t:
                            stack.append(c.output[0])
                            continue
                        axis = compensable(c, t)
                        if axis is None:
                            ok = False
                            break
                        site.compensations.append((c, axis, 0, which))
            if ok:
                sites.append(site)
    return sites


def equalise_concat(
    src: Path,
    dst: Path,
    batches: list[np.ndarray],
    *,
    slack: float = DEFAULT_SLACK,
    max_scale: float = DEFAULT_MAX_SCALE,
) -> list[dict[str, Any]]:
    """Rewrite every :func:`find_concat_sites` site; returns one report per site."""
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    model = onnx.load(str(src))
    _name_unnamed_nodes(model)
    sites = find_concat_sites(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not sites:
        onnx.save(model, str(dst))
        return []
    tensors = [s.concat.output[0] for s in sites]
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    probe.graph.output.extend([helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None) for t in tensors])
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.enable_cpu_mem_arena = False
    sess = ort.InferenceSession(probe.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    lo: dict[str, np.ndarray] = {}
    hi: dict[str, np.ndarray] = {}
    for b in batches:
        for t, v in zip(tensors, sess.run(tensors, {inp: b})):
            mn, mx = v.min(axis=(0, 2, 3)), v.max(axis=(0, 2, 3))
            lo[t] = mn if t not in lo else np.minimum(lo[t], mn)
            hi[t] = mx if t not in hi else np.maximum(hi[t], mx)
    del sess

    inits = {i.name: i for i in model.graph.initializer}

    def scale_axis(name: str, axis: int, factor: np.ndarray, divide: bool = False) -> None:
        arr = numpy_helper.to_array(inits[name]).astype(np.float64)
        shape = [1] * arr.ndim
        shape[axis] = -1
        arr = arr / factor.reshape(shape) if divide else arr * factor.reshape(shape)
        inits[name].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name))

    reports, touched = [], set()
    for site in sites:
        names = {p.input[1] for p, _ in site.branches} | {c.input[1] for c, *_ in site.compensations}
        if touched & names:
            continue  # weights already rescaled by an earlier site
        t = site.concat.output[0]
        s = choose_scales([(lo[t].astype(np.float64), hi[t].astype(np.float64))], slack=slack,
                          allow_negative=False, max_scale=max_scale).astype(np.float64)
        offsets = np.cumsum([0] + [ch for _, ch in site.branches])
        for i, (p, ch) in enumerate(site.branches):
            part = s[offsets[i]:offsets[i + 1]]
            scale_axis(p.input[1], PRODUCERS[p.op_type], part)
            if len(p.input) > 2 and p.input[2]:
                scale_axis(p.input[2], 0, part)
        for c, axis, _, which in site.compensations:
            part = s if which < 0 else s[offsets[which]:offsets[which + 1]]
            scale_axis(c.input[1], axis, part, divide=True)
        touched |= names
        before = float(np.median(255 * (hi[t] - lo[t]) / max(float(hi[t].max() - lo[t].min()), 1e-12)))
        lo2, hi2 = s * lo[t], s * hi[t]
        after = float(np.median(255 * (hi2 - lo2) / max(float(hi2.max() - lo2.min()), 1e-12)))
        reports.append({"concat": site.concat.name, "channels": int(len(s)), "branches": len(site.branches),
                        "compensated_convs": len(site.compensations), "scale_max": float(s.max()),
                        "median_levels_before": round(before, 2), "median_levels_after": round(after, 2)})
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    if reports:
        change = _max_output_change(src, dst, batches[0])
        for r in reports:
            r["max_abs_output_change"] = change
    return reports
