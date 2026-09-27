"""Analytic bias correction: cancel the mean shift that weight rounding puts on every channel.

Quantizing a layer's weights W to Q(W) changes its output by (Q(W) - W) x. Its expectation over
the data is a constant per output channel,

    delta_o = sum_{i,k} (W - Q(W))[o, i, k] * E[x_i],

which a bias absorbs exactly: with b' = b + delta, the quantized layer Q(W) x + b' has the float
layer's mean output. This is the data-driven half of Nagel et al.'s bias correction (2019), in the
analytic form (E[x] measured once on the float model, no quantized forward passes).

Why it matters here. Per-tensor weight quantization (TI's TIDL, AMD's XINT8) gives the small
channels of a layer few levels, so W - Q(W) is large relative to them and its mean does not
vanish. The error is systematic, not noise, and it compounds through depth: on EfficientNet-B1
under TIDL-like rules per-tensor weights alone cost ~17pp after equalisation. Per-channel weights
make it negligible, so this is for per-tensor targets.

Q must match the target's weight quantizer. Supported: symmetric int8 in [-127, 127], per tensor
or per output channel, scales float or rounded up to a power of two (XINT8). The weights are not
changed, so Q(W) after correction is the Q(W) the correction assumed: exact on average. The
approximation is E[x] from float activations and zero padding ignored (border outputs see fewer
inputs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class BiasCorrectionResult:
    layers: list[str] = field(default_factory=list)
    #: per layer: max |delta| / std of the layer's output bias scale, a size indication only
    max_shift: dict[str, float] = field(default_factory=dict)


def quantize_weights(w: np.ndarray, per_channel: bool, pow2: bool = False) -> np.ndarray:
    """Q(W): symmetric int8 in [-127, 127], dequantized, as the target quantizes weights."""
    w = np.asarray(w, np.float64)
    flat = w.reshape(w.shape[0], -1)
    amax = np.abs(flat).max(axis=1, keepdims=True) if per_channel else np.abs(flat).max(keepdims=True)
    scale = np.where(amax == 0, 1.0, amax / 127.0)
    if pow2:  # power-of-two scale covering the range
        scale = 2.0 ** np.ceil(np.log2(scale))
    return (np.clip(np.round(flat / scale), -127, 127) * scale).reshape(w.shape)


def _layers(g) -> list[tuple[int, str]]:
    """(node index, kind) of every Conv / Gemm(transB=1) whose weight is an initializer."""
    inits = {i.name for i in g.initializer}
    out = []
    for idx, n in enumerate(g.node):
        if n.op_type == "Conv" and len(n.input) > 1 and n.input[1] in inits:
            out.append((idx, "conv"))
        elif n.op_type == "Gemm" and len(n.input) > 1 and n.input[1] in inits:
            attrs = {a.name: a for a in n.attribute}
            if attrs.get("transB") is not None and attrs["transB"].i == 1 and (
                    attrs.get("transA") is None or attrs["transA"].i == 0):
                out.append((idx, "gemm"))
    return out


def input_means(model_path: Path, batches: Iterable[np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
    """Per-channel mean (axis 1) of the named float tensors over the batches."""
    import onnx
    import onnxruntime as ort

    m = onnx.load(str(model_path))
    have = {o.name for o in m.graph.output}
    for nm in names:
        if nm not in have:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0].name
    sums: dict[str, np.ndarray] = {}
    count: dict[str, int] = {}
    for x in batches:
        outs = s.run(names, {inp: x})
        for nm, t in zip(names, outs):
            t = np.asarray(t, np.float64)
            axes = tuple(a for a in range(t.ndim) if a != 1)
            n = int(np.prod([t.shape[a] for a in axes]))
            sums[nm] = sums.get(nm, 0.0) + t.sum(axis=axes)
            count[nm] = count.get(nm, 0) + n
    return {nm: sums[nm] / count[nm] for nm in names}


def correct_biases(src: Path, dst: Path, batches: Iterable[np.ndarray], *, per_channel: bool = False,
                   pow2: bool = False) -> BiasCorrectionResult:
    """Write ``dst``: ``src`` with every Conv/Gemm bias shifted by E[(W - Q(W)) x].

    Float outputs change by exactly that shift (the correction is for the quantized model).
    Weights that other nodes share are skipped.
    """
    import onnx
    from onnx import numpy_helper

    src, dst = Path(src), Path(dst)
    m = onnx.load(str(src))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    uses: dict[str, int] = {}
    for n in g.node:
        for i in n.input:
            uses[i] = uses.get(i, 0) + 1
    layers = [(idx, kind) for idx, kind in _layers(g) if uses.get(g.node[idx].input[1], 0) == 1]
    means = input_means(src, list(batches), sorted({g.node[idx].input[0] for idx, _ in layers}))
    result = BiasCorrectionResult()
    for idx, kind in layers:
        n = g.node[idx]
        w = numpy_helper.to_array(inits[n.input[1]]).astype(np.float64)
        err = w - quantize_weights(w, per_channel, pow2)
        mu = means[n.input[0]]
        if kind == "gemm":
            delta = err @ mu
        else:
            group = next((a.i for a in n.attribute if a.name == "group"), 1)
            c_out, c_in_g = w.shape[0], w.shape[1]
            e = err.reshape(c_out, c_in_g, -1).sum(axis=2)  # spatial taps all see E[x_i]
            mu_g = mu.reshape(group, c_in_g)
            delta = np.einsum("gok,gk->go", e.reshape(group, c_out // group, c_in_g), mu_g).reshape(c_out)
        if not np.any(delta):
            continue
        if len(n.input) > 2 and n.input[2]:
            if uses.get(n.input[2], 0) != 1:
                continue
            b = numpy_helper.to_array(inits[n.input[2]]).astype(np.float64)
            inits[n.input[2]].CopyFrom(numpy_helper.from_array((b + delta).astype(np.float32), n.input[2]))
        else:
            name = f"anneal_bc_bias_{idx}"
            g.initializer.append(numpy_helper.from_array(delta.astype(np.float32), name))
            if len(n.input) == 2:
                n.input.append(name)
            else:
                n.input[2] = name
        result.layers.append(n.name or f"node{idx}")
        result.max_shift[n.name or f"node{idx}"] = float(np.abs(delta).max())
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst))
    return result
