"""Predict which convolutions per-tensor INT8 activations will starve — from the float model.

Static INT8 gives each activation tensor one quantization step Δ, shared by all its channels.
Rounding adds noise of variance Δ²/12 to every input element of the next convolution, so the
noise that reaches output channel o is

    noise_o = Δ²/12 · ‖W_o‖²          (W_o: every weight feeding output channel o)

and its signal-to-quantization-noise ratio is SQNR_o = Var(z_o) / noise_o. A dense convolution
mixes all input channels, so a starved input channel costs it little. A *depthwise*
convolution maps channel c to channel c alone: when channel c spans a few steps of a range set
by some other channel — and batch-norm folding has left large weights on exactly those small
channels — its output is mostly noise. That is the mechanism behind EfficientNet-B0 and
MobileNetV3 losing 50pp and 12pp top-1 under static INT8 on every CPU.

The Δ²/12 noise model holds when a channel spans several quantization steps. A channel that
spans less than one step is rounded to one or two grid points: its error is a bias, not noise,
and the number reported for it is only a flag (the channel is starved either way).

This needs only the float model and a few calibration images; nothing is quantized. It models
activation rounding only: weight quantization, clipping and hardware arithmetic (see
:mod:`anneal.core.saturation`) are out of scope, and errors compound through the network in
ways a per-layer estimate does not capture. It says *where* to look; the audit says whether
it mattered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

#: Output channels below this predicted SQNR (dB) count as starved.
STARVED_DB = 10.0
#: A layer is flagged when at least this fraction of its output channels are starved.
FLAG_FRACTION = 0.05
#: Reported SQNRs are capped here, so layers without noise do not produce inf/nan summaries.
CAP_DB = 200.0


@dataclass
class LayerImbalance:
    node: str
    input_tensor: str
    depthwise: bool
    channels: int
    #: Input quantization step under a shared per-tensor scale (range / 255).
    step: float
    sqnr_median_db: float
    sqnr_p10_db: float
    sqnr_min_db: float
    starved_fraction: float
    #: Depthwise only: median and minimum quantization levels spanned by an input channel.
    levels_median: float | None = None
    levels_min: float | None = None

    @property
    def flagged(self) -> bool:
        return self.starved_fraction >= FLAG_FRACTION

    def to_dict(self) -> dict[str, Any]:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}
        d["flagged"] = self.flagged
        return d


def _db(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.maximum(x, 1e-30))


def analyse(model_path: Path, batches: Iterable[np.ndarray], *, levels: int = 255) -> list[LayerImbalance]:
    """Predicted per-channel activation-rounding SQNR at the output of every Conv."""
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    model = onnx.load(str(model_path))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    convs = [
        n for n in g.node
        if n.op_type == "Conv" and len(n.input) > 1 and n.input[1] in inits and n.input[0] not in inits
    ]
    if not convs:
        return []

    tensors = sorted({t for n in convs for t in (n.input[0], n.output[0])})
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    existing = {o.name for o in probe.graph.output}
    probe.graph.output.extend(
        [helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None) for t in tensors if t not in existing]
    )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(probe.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    inputs = {n.input[0] for n in convs}
    outputs = {n.output[0] for n in convs}
    t_lo: dict[str, float] = {}
    t_hi: dict[str, float] = {}
    c_lo: dict[str, np.ndarray] = {}
    c_hi: dict[str, np.ndarray] = {}
    s1: dict[str, np.ndarray] = {}
    s2: dict[str, np.ndarray] = {}
    count: dict[str, int] = {}
    for batch in batches:
        for t, v in zip(tensors, session.run(tensors, {input_name: batch})):
            v = v.astype(np.float64)
            axes = (0,) + tuple(range(2, v.ndim))
            if t in inputs:
                t_lo[t] = min(t_lo.get(t, 0.0), float(v.min()))
                t_hi[t] = max(t_hi.get(t, 0.0), float(v.max()))
                mn, mx = v.min(axis=axes), v.max(axis=axes)
                c_lo[t] = mn if t not in c_lo else np.minimum(c_lo[t], mn)
                c_hi[t] = mx if t not in c_hi else np.maximum(c_hi[t], mx)
            if t in outputs:
                s1[t] = s1.get(t, 0.0) + v.sum(axis=axes)
                s2[t] = s2.get(t, 0.0) + (v * v).sum(axis=axes)
                count[t] = count.get(t, 0) + int(np.prod([v.shape[a] for a in axes]))
    if not count:
        raise ValueError("imbalance analysis needs calibration batches; none were given")

    results: list[LayerImbalance] = []
    for node in convs:
        x, z = node.input[0], node.output[0]
        w = numpy_helper.to_array(inits[node.input[1]]).astype(np.float64)
        group = 1
        for a in node.attribute:
            if a.name == "group":
                group = helper.get_attribute_value(a)
        depthwise = group > 1 and w.shape[1] == 1 and group == w.shape[0]
        step = (t_hi[x] - t_lo[x]) / levels
        var = s2[z] / count[z] - (s1[z] / count[z]) ** 2
        noise = step * step / 12.0 * (w.reshape(w.shape[0], -1) ** 2).sum(axis=1)
        live = var > 1e-12  # dead output channels carry no signal to lose
        sqnr = _db(var[live] / np.maximum(noise[live], 1e-30)) if live.any() else np.array([CAP_DB])
        sqnr = np.minimum(sqnr, CAP_DB)  # a noiseless layer (zero weights) is not infinitely good
        lv = None
        if depthwise and step > 0:
            lv = (c_hi[x] - c_lo[x]) / step
        results.append(
            LayerImbalance(
                node=node.name or z,
                input_tensor=x,
                depthwise=depthwise,
                channels=int(w.shape[0]),
                step=float(step),
                sqnr_median_db=float(np.median(sqnr)),
                sqnr_p10_db=float(np.percentile(sqnr, 10)),
                sqnr_min_db=float(sqnr.min()),
                starved_fraction=float((sqnr < STARVED_DB).mean()),
                levels_median=None if lv is None else float(np.median(lv)),
                levels_min=None if lv is None else float(lv.min()),
            )
        )
    return results


def summarise(results: list[LayerImbalance]) -> dict[str, Any]:
    flagged = [r for r in results if r.flagged]
    return {
        "layers": len(results),
        "flagged": len(flagged),
        "flagged_depthwise": sum(r.depthwise for r in flagged),
        "worst": sorted(results, key=lambda r: r.sqnr_p10_db)[0].node if results else None,
        "worst_p10_sqnr_db": min((r.sqnr_p10_db for r in results), default=None),
        "median_layer_p10_sqnr_db": float(np.median([r.sqnr_p10_db for r in results])) if results else None,
    }
