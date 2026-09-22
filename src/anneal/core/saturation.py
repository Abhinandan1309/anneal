"""Predict INT8 accumulator saturation on x86 CPUs without VNNI — without owning one.

Background. The AVX2 path onnxruntime uses for u8×s8 matrix products (``VPMADDUBSW``)
multiplies pairs of unsigned activations by signed weights and adds each *pair* into a
**saturating 16-bit** integer before widening to 32 bits:

    pair = a[k]·w[k] + a[k+1]·w[k+1]      clipped to [-32768, 32767]

One product can reach 255 × 127 = 32,385, so a pair can reach 64,770 and be clipped.
With ``reduce_range`` the weights are 7-bit (|w| ≤ 63), and 255 × 63 × 2 = 32,130 can
never exceed the limit. VNNI (``VPDPBUSD``) and ARM's dot-product instructions (``SDOT``)
accumulate straight into 32 bits and cannot saturate this way.

This module emulates the 16-bit pair arithmetic on a quantized model's *real* int8 weights
and *real* u8 activations (captured by running the model on a few images) and reports, per
layer, how often pairs saturate and how much that corrupts the accumulators.

Assumptions, stated because the result depends on them:

* **Pairing order.** Pairs are formed from consecutive elements of the reduction axis with
  input channels innermost (the NHWC layout onnxruntime's quantized convolution uses). A
  different packing would pair different elements; the *worst-case* check
  (:func:`weight_pair_risk`) does not depend on it.
* **S8S8 on x86 without VNNI** is modelled as the u8 path on activations shifted by +128,
  the usual way such kernels handle signed activations. That S8S8 per-channel models broke
  exactly as badly as U8S8 on those CPUs is consistent with this, not proof of it.

The emulation predicts *damage to accumulators*, not end-to-end accuracy; the hardware lab
is what measured accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

I16_MIN, I16_MAX = -32768, 32767
#: A weight pair whose magnitudes sum past this can saturate when both activations are 255.
PAIR_WEIGHT_LIMIT = I16_MAX / 255.0

QUANTIZED_MATMUL_OPS = ("Conv", "Gemm", "MatMul")


@dataclass
class LayerSaturation:
    node: str
    op_type: str
    reduction_len: int
    weight_max_abs: int
    #: Fraction of weight pairs that *could* saturate given the worst activations (255, 255).
    risky_weight_pairs: float
    pairs_checked: int = 0
    saturated_pairs: int = 0
    accumulators_checked: int = 0
    accumulators_affected: int = 0
    #: Mean |sat - exact| / |exact| over accumulators touched by saturation.
    mean_relative_error: float = 0.0
    bias_saturated_channels: int = 0
    #: Convolution groups; equal to the input channels for a depthwise convolution.
    groups: int = 1
    depthwise: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def pair_rate(self) -> float:
        return self.saturated_pairs / self.pairs_checked if self.pairs_checked else 0.0

    @property
    def accumulator_rate(self) -> float:
        return self.accumulators_affected / self.accumulators_checked if self.accumulators_checked else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "op_type": self.op_type,
            "reduction_len": self.reduction_len,
            "weight_max_abs": self.weight_max_abs,
            "risky_weight_pairs": self.risky_weight_pairs,
            "pairs_checked": self.pairs_checked,
            "saturated_pairs": self.saturated_pairs,
            "pair_rate": self.pair_rate,
            "accumulators_checked": self.accumulators_checked,
            "accumulators_affected": self.accumulators_affected,
            "accumulator_rate": self.accumulator_rate,
            "mean_relative_error": self.mean_relative_error,
            "bias_saturated_channels": self.bias_saturated_channels,
            "groups": self.groups,
            "depthwise": self.depthwise,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# pure arithmetic — testable without onnx
# ---------------------------------------------------------------------------


def pair_sums(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Adjacent-pair sums of a·w along the last axis, exactly (int32), before any clipping.

    ``a`` is (S, K) activations, ``w`` is (O, K) weights; returns (S, O, ceil(K/2)).
    An odd trailing element forms a pair with an implicit zero.
    """
    a = a.astype(np.int32)
    w = w.astype(np.int32)
    k = a.shape[-1]
    if k % 2:
        a = np.concatenate([a, np.zeros(a.shape[:-1] + (1,), np.int32)], axis=-1)
        w = np.concatenate([w, np.zeros(w.shape[:-1] + (1,), np.int32)], axis=-1)
    prod = a[:, None, :] * w[None, :, :]
    return prod[..., 0::2] + prod[..., 1::2]


def saturation_stats(a: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Emulate saturating 16-bit pair accumulation and compare with exact arithmetic."""
    pairs = pair_sums(a, w)
    clipped = np.clip(pairs, I16_MIN, I16_MAX)
    saturated = pairs != clipped
    exact = pairs.sum(axis=-1, dtype=np.int64)
    approx = clipped.sum(axis=-1, dtype=np.int64)
    touched = saturated.any(axis=-1)
    rel = np.abs(approx - exact) / np.maximum(np.abs(exact), 1)
    return {
        "pairs": int(pairs.size),
        "saturated_pairs": int(saturated.sum()),
        "accumulators": int(exact.size),
        "accumulators_affected": int(touched.sum()),
        "relative_error_sum": float(rel[touched].sum()),
    }


def weight_pair_risk(w: np.ndarray) -> float:
    """Fraction of adjacent weight pairs that can saturate if both activations are 255."""
    w = np.abs(w.astype(np.int32))
    if w.shape[-1] % 2:
        w = np.concatenate([w, np.zeros(w.shape[:-1] + (1,), np.int32)], axis=-1)
    s = w[..., 0::2] + w[..., 1::2]
    return float((s > PAIR_WEIGHT_LIMIT).mean()) if s.size else 0.0


# ---------------------------------------------------------------------------
# graph inspection
# ---------------------------------------------------------------------------


@dataclass
class _QLayer:
    node: Any
    act_tensor: str  # name of the quantized (u8/s8) activation tensor feeding the op
    act_zero_point: int
    act_signed: bool
    weight_q: np.ndarray
    bias_q: np.ndarray | None
    attrs: dict[str, Any]


def _find_layers(model) -> list[_QLayer]:
    from onnx import helper, numpy_helper

    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    producer = {out: n for n in model.graph.node for out in n.output}
    layers: list[_QLayer] = []

    def dq_source(name: str):
        node = producer.get(name)
        if node is None or node.op_type != "DequantizeLinear":
            return None
        return node

    for node in model.graph.node:
        if node.op_type not in QUANTIZED_MATMUL_OPS or len(node.input) < 2:
            continue
        a_dq, w_dq = dq_source(node.input[0]), dq_source(node.input[1])
        if a_dq is None or w_dq is None or w_dq.input[0] not in inits:
            continue
        zp_name = a_dq.input[2] if len(a_dq.input) > 2 else None
        zp = inits.get(zp_name) if zp_name else None
        weight = inits[w_dq.input[0]]
        if weight.dtype != np.int8:
            continue
        bias = None
        if len(node.input) > 2:
            b_dq = dq_source(node.input[2])
            if b_dq is not None and b_dq.input[0] in inits:
                bias = inits[b_dq.input[0]]
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        layers.append(
            _QLayer(
                node=node,
                act_tensor=a_dq.input[0],
                act_zero_point=int(zp) if zp is not None else 0,
                act_signed=bool(zp is not None and zp.dtype == np.int8),
                weight_q=weight,
                bias_q=bias,
                attrs=attrs,
            )
        )
    return layers


def _capture_activations(model_path: Path, tensors: list[str], batches: Iterable[np.ndarray]):
    """Run the model with the quantized activation tensors exposed as extra outputs."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(model_path))
    existing = {o.name for o in model.graph.output}
    for name in tensors:
        if name not in existing:
            model.graph.output.append(onnx.ValueInfoProto(name=name))
    opts = ort.SessionOptions()
    # Plain execution: the QuantizeLinear outputs are what the fused kernels would consume.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    captured: dict[str, list[np.ndarray]] = {t: [] for t in tensors}
    for x in batches:
        outs = session.run(tensors, {input_name: x})
        for name, value in zip(tensors, outs):
            captured[name].append(value)
    return {k: np.concatenate(v, axis=0) for k, v in captured.items()}


def _conv_patches(
    a: np.ndarray, layer: _QLayer, rng: np.random.Generator, n_positions: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample output positions; return per-group (S, K) activations and (O, K) weights.

    K is ordered (kh, kw, cin) with input channels innermost.
    """
    attrs = layer.attrs
    w = layer.weight_q  # (O, C/g, kh, kw)
    groups = int(attrs.get("group", 1))
    kh, kw = w.shape[2], w.shape[3]
    sh, sw = (attrs.get("strides") or [1, 1])[:2]
    dh, dw = (attrs.get("dilations") or [1, 1])[:2]
    pads = attrs.get("pads") or [0, 0, 0, 0]
    n, c, h, wd = a.shape
    fill = layer.act_zero_point
    padded = np.full((n, c, h + pads[0] + pads[2], wd + pads[1] + pads[3]), fill, dtype=np.int32)
    padded[:, :, pads[0]:pads[0] + h, pads[1]:pads[1] + wd] = a
    ho = (padded.shape[2] - dh * (kh - 1) - 1) // sh + 1
    wo = (padded.shape[3] - dw * (kw - 1) - 1) // sw + 1
    total = n * ho * wo
    pick = rng.choice(total, size=min(n_positions, total), replace=False)
    ni, rem = np.divmod(pick, ho * wo)
    yi, xi = np.divmod(rem, wo)
    rows = yi[:, None] * sh + np.arange(kh)[None, :] * dh  # (S, kh)
    cols = xi[:, None] * sw + np.arange(kw)[None, :] * dw  # (S, kw)
    # patch[s, c, i, j] = padded[n_s, c, rows[s, i], cols[s, j]]
    patch = padded[ni[:, None, None, None], np.arange(c)[None, :, None, None],
                   rows[:, None, :, None], cols[:, None, None, :]]
    return patch, w, groups


def analyse(
    model_path: Path,
    batches: Iterable[np.ndarray],
    *,
    n_positions: int = 256,
    seed: int = 0,
) -> list[LayerSaturation]:
    """Per-layer saturation of a QDQ-quantized model under x86 non-VNNI arithmetic."""
    import onnx

    model = onnx.load(str(model_path))
    layers = _find_layers(model)
    if not layers:
        return []
    acts = _capture_activations(model_path, sorted({l.act_tensor for l in layers}), batches)
    rng = np.random.default_rng(seed)
    results: list[LayerSaturation] = []

    for layer in layers:
        node = layer.node
        a = acts[layer.act_tensor].astype(np.int32)
        if layer.act_signed:
            a = a + 128  # modelled as the u8 path on shifted activations (see module docstring)
        w = layer.weight_q.astype(np.int32)
        stats_total = {"pairs": 0, "saturated_pairs": 0, "accumulators": 0,
                       "accumulators_affected": 0, "relative_error_sum": 0.0}
        notes: list[str] = []

        groups, depthwise = 1, False
        if node.op_type == "Conv" and w.ndim == 4:
            patch, w4, groups = _conv_patches(a, layer, rng, n_positions)
            depthwise = groups > 1 and w4.shape[1] == 1
            cpg, opg = w4.shape[1], w4.shape[0] // groups
            k_len = cpg * w4.shape[2] * w4.shape[3]
            risky = []
            for g in range(groups):
                act_k = patch[:, g * cpg:(g + 1) * cpg].transpose(0, 2, 3, 1).reshape(len(patch), -1)
                w_k = w4[g * opg:(g + 1) * opg].transpose(0, 2, 3, 1).reshape(opg, -1)
                risky.append(weight_pair_risk(w_k))
                for start in range(0, len(act_k), 16):
                    s = saturation_stats(act_k[start:start + 16], w_k)
                    for key in stats_total:
                        stats_total[key] += s[key]
            risky_frac = float(np.mean(risky))
        else:
            # Gemm / MatMul: activations (N, K); weights (K, O) or (O, K) with transB.
            x = a.reshape(a.shape[0], -1)
            w2 = w.T if (node.op_type == "MatMul" or not layer.attrs.get("transB", 0)) else w
            k_len = w2.shape[1]
            risky_frac = weight_pair_risk(w2)
            for start in range(0, len(x), 16):
                s = saturation_stats(x[start:start + 16], w2)
                for key in stats_total:
                    stats_total[key] += s[key]

        bias_hits = 0
        if layer.bias_q is not None:
            info = np.iinfo(np.int32)
            bias_hits = int(np.sum((layer.bias_q <= info.min + 1) | (layer.bias_q >= info.max - 1)))
            if bias_hits:
                notes.append(
                    f"{bias_hits} bias value(s) pinned at the int32 limit: a near-zero per-channel "
                    f"weight scale made the bias unrepresentable. This is in the model file and "
                    f"affects every CPU, not only non-VNNI x86."
                )
        affected = stats_total["accumulators_affected"]
        results.append(
            LayerSaturation(
                node=node.name or node.output[0],
                op_type=node.op_type,
                reduction_len=int(k_len),
                weight_max_abs=int(np.abs(w).max()),
                risky_weight_pairs=risky_frac,
                pairs_checked=stats_total["pairs"],
                saturated_pairs=stats_total["saturated_pairs"],
                accumulators_checked=stats_total["accumulators"],
                accumulators_affected=affected,
                mean_relative_error=(stats_total["relative_error_sum"] / affected) if affected else 0.0,
                bias_saturated_channels=bias_hits,
                groups=int(groups),
                depthwise=bool(depthwise),
                notes=notes,
            )
        )
    return results


def summarise(results: list[LayerSaturation]) -> dict[str, Any]:
    pairs = sum(r.pairs_checked for r in results)
    sat = sum(r.saturated_pairs for r in results)
    accs = sum(r.accumulators_checked for r in results)
    hit = sum(r.accumulators_affected for r in results)
    worst = max(results, key=lambda r: r.accumulator_rate, default=None)
    can_saturate = any(r.risky_weight_pairs > 0 for r in results)
    return {
        "layers": len(results),
        "layers_saturating": sum(r.saturated_pairs > 0 for r in results),
        "pair_rate": sat / pairs if pairs else 0.0,
        "accumulator_rate": hit / accs if accs else 0.0,
        "worst_layer": worst.node if worst and worst.saturated_pairs else None,
        "worst_layer_accumulator_rate": worst.accumulator_rate if worst else 0.0,
        "saturation_possible": can_saturate,
        "bias_saturated_channels": sum(r.bias_saturated_channels for r in results),
    }
