"""Which activation tensors lose the most at 8 bits? Measured by fake-quantizing one at a time.

Not to be confused with :mod:`anneal.core.sensitivity`, which measures *layers* by quantizing
one layer's *weights* at a time. This module is about *activation tensors*: static INT8 gives
each one a single per-tensor scale and zero point, and a few tensors (a stem output whose
channels differ in range by orders of magnitude, say) can carry most of a model's accuracy
loss even after equalisation. Keeping just those at 16 bits is cheap mixed precision.

The measurement (the core of ``examples/advise/tensor_sensitivity.py``): in the float model,
insert a per-tensor uint8 asymmetric QuantizeLinear/DequantizeLinear pair on one tensor only,
with its range the min/max over the calibration batches, and count the share of probe
predictions (argmax over axis 1 of output 0) that differ from the float model's. Everything
else stays in float, so the damage is that tensor's alone. The cost is one float inference
pass over the probe per candidate tensor, plus one pass over the calibration batches for the
ranges.

Memory: batches are streamed, only one inference session exists at a time, the CPU memory
arena is disabled (so a session's peak does not stay reserved), and the fake-quantized copies
are written to a temporary directory and deleted.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np

#: Ops whose data inputs are the candidate tensors by default: the ones static INT8 quantizes
#: on every backend, and whose inputs are what the integer kernels actually consume.
CANDIDATE_OPS = ("Conv", "Gemm", "MatMul")
#: Bias inputs are quantized with the weights, never as activations.
_BIAS_INPUT = {"Conv": 2, "Gemm": 2}


class TensorDamage(NamedTuple):
    """An activation tensor and the share of probe predictions its 8-bit version flips."""

    tensor: str
    damage: float


def candidate_tensors(model) -> list[str]:
    """Data (non-constant) inputs of every Conv/Gemm/MatMul, in graph order, each once."""
    g = model.graph
    constant = {i.name for i in g.initializer}
    constant |= {o for n in g.node if n.op_type == "Constant" for o in n.output}
    seen: dict[str, None] = {}
    for n in g.node:
        if n.op_type not in CANDIDATE_OPS:
            continue
        for j, t in enumerate(n.input):
            if t and t not in constant and j != _BIAS_INPUT.get(n.op_type):
                seen.setdefault(t, None)
    return list(seen)


def _session(model_or_path, *, keep_all: bool = True):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    if keep_all:
        # Nothing may be fused away: the Q/DQ pair and every probed tensor must exist as written.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    src = model_or_path if isinstance(model_or_path, (bytes, str)) else str(model_or_path)
    return ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])


def tensor_ranges(
    model_path: Path, tensors: Sequence[str], batches: Iterable[np.ndarray]
) -> dict[str, tuple[float, float]]:
    """Per-tensor (min, max) over the batches, streamed: only two floats per tensor are kept."""
    import onnx
    from onnx import helper

    model = onnx.load(str(model_path))
    existing = {o.name for o in model.graph.output}
    model.graph.output.extend(
        [helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None) for t in tensors if t not in existing]
    )
    blob = model.SerializeToString()
    del model
    session = _session(blob)
    del blob
    name = session.get_inputs()[0].name
    lo = {t: np.inf for t in tensors}
    hi = {t: -np.inf for t in tensors}
    seen = False
    for batch in batches:
        seen = True
        for t, v in zip(tensors, session.run(list(tensors), {name: batch})):
            lo[t] = min(lo[t], float(v.min()))
            hi[t] = max(hi[t], float(v.max()))
    if not seen:
        raise ValueError("activation sensitivity needs calibration batches; none were given")
    return {t: (lo[t], hi[t]) for t in tensors}


def uint8_qparams(lo: float, hi: float) -> tuple[float, int]:
    """Asymmetric per-tensor uint8 scale and zero point for [lo, hi], widened to include 0."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = max(hi - lo, 1e-12) / 255.0
    return scale, int(np.clip(round(-lo / scale), 0, 255))


def _predict(session, xs: Sequence[np.ndarray]) -> np.ndarray:
    """Argmax over axis 1 of output 0, flattened: top-1 per image for a classifier."""
    name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    return np.concatenate([session.run([out_name], {name: x})[0].argmax(1).ravel() for x in xs])


def rank_activation_tensors(
    model_path: Path,
    calib_batches: Iterable[np.ndarray],
    probe_batches: Iterable[np.ndarray],
    tensors: Sequence[str] | None = None,
) -> list[TensorDamage]:
    """Every candidate tensor with its 8-bit damage, most damaging first.

    ``calib_batches`` set each tensor's range (streamed once). ``probe_batches`` are the
    inputs whose predictions are compared; they are held in memory and run once per candidate,
    so keep them small (a few dozen images resolve the tensors that matter). ``tensors``
    defaults to :func:`candidate_tensors`; an unknown name is an error. Ties keep graph order.
    """
    import onnx
    from onnx import helper, numpy_helper

    model_path = Path(model_path)
    model = onnx.load(str(model_path))
    g = model.graph
    known = {i.name for i in g.input} | {o for n in g.node for o in n.output}
    if tensors is None:
        tensors = candidate_tensors(model)
    else:
        tensors = list(dict.fromkeys(tensors))
        unknown = [t for t in tensors if t not in known]
        if unknown:
            raise ValueError(f"not tensors of the model: {unknown[:3]}")
    if not tensors:
        return []
    xs = [np.ascontiguousarray(x, dtype=np.float32) for x in probe_batches]
    if not xs:
        raise ValueError("activation sensitivity needs probe batches; none were given")

    ranges = tensor_ranges(model_path, tensors, calib_batches)
    session = _session(model_path)
    ref = _predict(session, xs)
    del session

    rows: list[TensorDamage] = []
    with tempfile.TemporaryDirectory(prefix="anneal-act-sens-") as tmp:
        path = Path(tmp) / "one.onnx"
        for t in tensors:
            scale, zp = uint8_qparams(*ranges[t])
            s_name, z_name = "anneal_fq_scale", "anneal_fq_zp"
            q, dq = f"{t}__anneal_fq_q", f"{t}__anneal_fq"
            # Rewire in place, save, undo: no per-candidate copy of the model in memory.
            rewired = [(node, j) for node in g.node for j, inp in enumerate(node.input) if inp == t]
            for node, j in rewired:
                node.input[j] = dq
            g.initializer.extend([numpy_helper.from_array(np.array(scale, np.float32), s_name),
                                  numpy_helper.from_array(np.array(zp, np.uint8), z_name)])
            g.node.extend([helper.make_node("QuantizeLinear", [t, s_name, z_name], [q], name="anneal_fq_q"),
                           helper.make_node("DequantizeLinear", [q, s_name, z_name], [dq], name="anneal_fq_dq")])
            onnx.save(model, str(path))
            del g.node[-2:]
            del g.initializer[-2:]
            for node, j in rewired:
                node.input[j] = t
            session = _session(path)
            rows.append(TensorDamage(t, float(np.mean(_predict(session, xs) != ref))))
            del session
    return sorted(rows, key=lambda r: -r.damage)
