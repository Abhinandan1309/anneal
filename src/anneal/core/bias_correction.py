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

Q must match the target's weight quantizer, or be read from it: ``weights_from_qdq`` takes the
dequantized weights out of a QDQ model the target's own tool produced (measured, not assumed;
AMD Quark chooses its XINT8 weight scales by error minimisation, which a formula would miss).
Otherwise Supported: symmetric int8 in [-127, 127], per tensor
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


def weights_from_qdq(qdq_path: Path) -> dict[str, np.ndarray]:
    """Node name -> dequantized weight, for each Conv/Gemm in a QDQ model whose weight input is a
    DequantizeLinear of an initializer (the tool's own Q(W))."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(qdq_path))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    out = {}
    for n in g.node:
        if n.op_type not in ("Conv", "Gemm") or len(n.input) < 2 or not n.name:
            continue
        dq = prod.get(n.input[1])
        # DequantizeLinear, or a vendor's (Quark: VitisDequantizeLinear) with the same inputs
        if dq is None or not dq.op_type.endswith("DequantizeLinear") or dq.input[1] not in inits:
            continue
        scale = numpy_helper.to_array(inits[dq.input[1]]).astype(np.float64)
        zp_init = inits.get(dq.input[2]) if len(dq.input) > 2 else None
        zp = numpy_helper.to_array(zp_init).astype(np.float64) if zp_init is not None else np.zeros_like(scale)
        axis = next((a.i for a in dq.attribute if a.name == "axis"), 1)
        if dq.input[0] in inits:  # weight stored quantized
            q = numpy_helper.to_array(inits[dq.input[0]]).astype(np.float64)
        else:  # weight stored float behind a QuantizeLinear: apply it
            ql = prod.get(dq.input[0])
            if ql is None or not ql.op_type.endswith("QuantizeLinear") or ql.input[0] not in inits:
                continue
            w = numpy_helper.to_array(inits[ql.input[0]]).astype(np.float64)
            lo, hi = (-128, 127) if zp_init is None or zp_init.data_type == onnx.TensorProto.INT8 else (0, 255)
            if zp_init is not None and zp_init.data_type in (onnx.TensorProto.INT16, onnx.TensorProto.UINT16):
                lo, hi = (-32768, 32767) if zp_init.data_type == onnx.TensorProto.INT16 else (0, 65535)
            s_b, z_b = scale, zp
            if scale.ndim == 1 and scale.size > 1:
                shape = [1] * w.ndim
                shape[axis] = -1
                s_b, z_b = scale.reshape(shape), zp.reshape(shape)
            q = np.clip(np.round(w / s_b) + z_b, lo, hi)
        if scale.ndim == 1 and scale.size > 1:
            shape = [1] * q.ndim
            shape[axis] = -1
            scale, zp = scale.reshape(shape), zp.reshape(shape)
        out[n.name] = (q - zp) * scale
    return out


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
                   pow2: bool = False, quantized: dict[str, np.ndarray] | None = None) -> BiasCorrectionResult:
    """Write ``dst``: ``src`` with every Conv/Gemm bias shifted by E[(W - Q(W)) x].

    ``quantized`` (node name -> Q(W), from ``weights_from_qdq``) replaces the formula; layers it
    does not cover are left alone.

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
    layers = [(idx, kind) for idx, kind in _layers(g) if uses.get(g.node[idx].input[1], 0) == 1
              and (quantized is None or g.node[idx].name in quantized)]
    means = input_means(src, list(batches), sorted({g.node[idx].input[0] for idx, _ in layers}))
    result = BiasCorrectionResult()
    for idx, kind in layers:
        n = g.node[idx]
        w = numpy_helper.to_array(inits[n.input[1]]).astype(np.float64)
        qw = quantized[n.name] if quantized is not None else quantize_weights(w, per_channel, pow2)
        if qw.shape != w.shape:
            continue
        err = w - qw
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
