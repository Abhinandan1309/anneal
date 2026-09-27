"""Equalisation scales derived from a quantization-noise model, for per-tensor weights.

The rewrite of :mod:`anneal.core.equalize` is exact for any per-channel scale s. Which s is best
depends on the target: with per-channel weights only the activations share a scale, and
activation equalisation (every channel's range brought up to the tensor's) is optimal. With
per-tensor weights (TIDL, AMD XINT8) s also moves precision between the rows of the producer's
weights and the columns of the consumers' weights. Until now that trade-off was a fixed blend,
``s = s_act**0.5 * s_cle**0.5``. This module derives it.

The model. Every tensor T at a site is quantized with one step ``Delta_T = M_T / L``, where
``M_T`` is its largest channel range after the rescale. A tensor the scale multiplies (the
producer's weights, the pre-activation x', the activation y', ...) has ``M_T = max_c s_c r_Tc``;
a tensor it divides (a consumer's weights) has ``M_T = max_c r_Tc / s_c``. Rounding adds
independent noise of variance ``Delta^2 / 12`` per element. Measured at the outputs of the
consumers that divide s out (where the function is back to the original one), the noise of
channel c is

    U_c / s_c**2 + D_c * s_c**2,   U_c = sum_{T up} w_Tc M_T**2,   D_c = sum_{T down} w_Tc M_T**2

(common factor ``1 / (12 L**2)`` dropped). The weights ``w_Tc`` are the channel's sensitivities:

* activation y' feeding consumer weights W: ``sum_{o,k} W[o, c, k]**2`` (a depthwise consumer's
  kernel, or a dense consumer's input column), times ``E[e_c**2]`` when y reaches W through a
  squeeze-excite product ``y * e``;
* pre-activation x' (quantized, then gated): the same times ``E[g'(x_c)**2]``, the mean squared
  slope of the activation (SiLU, Hardswish, ReLU);
* the producer's weights: that times ``K_A * E[in**2]`` (fan-in times the mean square of its
  input): every weight's rounding error reaches x_c through one input element;
* a consumer's weights (divided): ``K_W * C_out * E[y_c**2]`` for the elements it multiplies.

The total ``E(s) = sum_c U_c / s_c**2 + D_c s_c**2`` is invariant under s -> t s. For fixed
maxima M the problem separates by channel, with the minimiser

    s_c = (U_c / D_c) ** (1/4),  clipped to  max_{T down} r_Tc / M_T <= s_c <= min_{T up} M_T / r_Tc

(the box keeps every channel inside the maxima assumed). What remains is a search over the few
maxima (two to five per site, one of them fixed by the invariance), done here by Nelder-Mead on
their logarithms against the true objective, from the blend's own maxima: the result is never
worse than the blend under the model. Special cases: with no divided terms (per-channel
weights) every channel rises to the box, which is activation equalisation; a channel whose
activation is a constant (a dead weight row, the bias only) has a small sensitivity to the
producer's weights and gets no inflated scale, which the blend had to special-case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _slope_sq(x: np.ndarray, act: str) -> np.ndarray:
    """Squared derivative of the activation at x (elementwise)."""
    if act == "silu":
        sg = 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
        return (sg * (1.0 + x * (1.0 - sg))) ** 2
    if act == "hswish":
        return np.where(x < -3.0, 0.0, np.where(x > 3.0, 1.0, (2.0 * x + 3.0) / 6.0)) ** 2
    if act == "relu":
        return (x > 0).astype(np.float64)
    raise ValueError(act)


def channel_moments(model, requests: dict[str, set[str]], batches) -> dict[str, dict[str, np.ndarray]]:
    """Moments of NCHW tensors over calibration batches, in one pass.

    ``requests`` maps a tensor to the statistics wanted: ``"ms"`` (overall mean square, a scalar),
    ``"ms_c"`` (per-channel mean square), ``"absmax_c"`` (per-channel max |value|) and
    ``"slope_<act>"`` (per-channel mean squared activation slope, act in silu/hswish/relu).
    """
    import onnx
    import onnxruntime as ort
    from onnx import helper

    tensors = sorted(requests)
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    existing = {o.name for o in probe.graph.output}
    probe.graph.output.extend(
        [helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None) for t in tensors if t not in existing])
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(probe.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    sums: dict[str, dict[str, np.ndarray]] = {t: {} for t in tensors}
    counts: dict[str, float] = {t: 0.0 for t in tensors}
    for batch in batches:
        for t, v in zip(tensors, session.run(tensors, {input_name: batch}), strict=True):
            v = v.astype(np.float64)
            axes = (0,) + tuple(range(2, v.ndim))
            n_per_c = v.size / v.shape[1]
            counts[t] += n_per_c
            acc = sums[t]
            for stat in requests[t]:
                if stat in ("ms", "ms_c"):
                    val = (v * v).sum(axis=axes)
                elif stat == "absmax_c":
                    val = np.abs(v).max(axis=axes)
                    acc[stat] = val if stat not in acc else np.maximum(acc[stat], val)
                    continue
                elif stat.startswith("slope_"):
                    val = _slope_sq(v, stat[6:]).sum(axis=axes)
                else:
                    raise ValueError(stat)
                acc[stat] = val if stat not in acc else acc[stat] + val
    out: dict[str, dict[str, np.ndarray]] = {}
    for t in tensors:
        out[t] = {}
        for stat, val in sums[t].items():
            if stat == "absmax_c":
                out[t][stat] = val
            elif stat == "ms":
                out[t][stat] = np.asarray(val.sum() / (counts[t] * len(val)))
            else:
                out[t][stat] = val / counts[t]
    return out


def output_sensitivities(model, tensors: list[str], batches, probes: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    """Per-channel sensitivity of the network output to independent noise in each tensor.

    For noise of variance v per element in channel c of tensor T, the expected squared error of
    the output is ``v * S_Tc`` with ``S_Tc = sum over the channel's elements of ||d out / d T||^2``
    (per image). This is the trace of J J^T restricted to the channel, estimated with
    Hutchinson's method: for random normal v on the output, ``E ||J^T v||^2 = trace(J J^T)``, so a
    few backward passes give every tensor's and channel's value at once. Exact where the
    per-site formulas are not: through residual chains and squeeze-excite products, and with
    everything downstream of the site. Needs torch and onnx2torch; raises ImportError without.
    """
    import torch
    from onnx2torch import convert

    producer = {}
    for n in model.graph.node:
        if n.output:
            producer[n.output[0]] = n.name
    target_of = {t: producer[t].lstrip("/").replace(".", "/") for t in tensors if t in producer}
    gm = convert(model).eval()
    by_target = {n.target: n for n in gm.graph.nodes if n.op == "call_module"}
    wanted = {by_target[tg]: t for t, tg in target_of.items() if tg in by_target}
    missing = sorted(set(tensors) - set(wanted.values()))
    if missing:
        raise KeyError(f"no converted node for {missing[:3]}")

    captured: dict[str, torch.Tensor] = {}

    class Capture(torch.fx.Interpreter):
        def run_node(self, n):
            out = super().run_node(n)
            if n in wanted and isinstance(out, torch.Tensor) and out.requires_grad:
                out.retain_grad()
                captured[wanted[n]] = out
            return out

    gen = torch.Generator().manual_seed(seed)
    acc: dict[str, np.ndarray] = {}
    images = 0
    for batch in batches:
        x = torch.from_numpy(np.asarray(batch, np.float32))
        for _ in range(probes):
            captured.clear()
            gm.zero_grad(set_to_none=True)
            out = Capture(gm).run(x)
            out = out[0] if isinstance(out, (tuple, list)) else out
            v = torch.randn(out.shape, generator=gen)
            (out * v).sum().backward()
            for t, ten in captured.items():
                g = ten.grad
                if g is None:
                    continue
                axes = (0,) + tuple(range(2, g.ndim))
                val = (g.double() ** 2).sum(dim=axes).numpy()
                acc[t] = val if t not in acc else acc[t] + val
        images += x.shape[0]
    return {t: v / (images * probes) for t, v in acc.items()}


@dataclass
class Term:
    """One quantized tensor at a site: per-channel ranges and sensitivities, and the direction."""

    name: str
    r: np.ndarray  # per-channel range (max |value|) before the rescale
    w: np.ndarray  # per-channel noise sensitivity at the site's outputs
    up: bool  # True: the tensor carries s (multiplied); False: it divides s out


def objective(terms: list[Term], s: np.ndarray) -> float:
    """The model's total output noise for scales s (up to the common factor)."""
    s = np.abs(np.asarray(s, np.float64))
    total = 0.0
    for t in terms:
        if t.up:
            m = float(np.max(s * t.r))
            total += float(np.sum(t.w * m * m / (s * s)))
        else:
            m = float(np.max(t.r / s))
            total += float(np.sum(t.w * m * m * s * s))
    return total


def _scales_for_maxima(terms: list[Term], log_m: np.ndarray) -> np.ndarray:
    m = np.exp(log_m)
    c = len(terms[0].r)
    u, d = np.zeros(c), np.zeros(c)
    lo, hi = np.zeros(c), np.full(c, np.inf)
    for t, mt in zip(terms, m):
        if t.up:
            u += t.w * mt * mt
            with np.errstate(divide="ignore"):
                hi = np.minimum(hi, np.where(t.r > 0, mt / np.maximum(t.r, 1e-300), np.inf))
        else:
            d += t.w * mt * mt
            lo = np.maximum(lo, t.r / mt)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(d > 0, (u / np.where(d > 0, d, 1.0)) ** 0.25, np.inf)
    s = np.minimum(np.maximum(s, lo), hi)
    # no bound either way (a channel no term constrains): leave it at 1
    s = np.where(np.isfinite(s) & (s > 0), s, np.where(np.isfinite(hi), hi, np.where(lo > 0, lo, 1.0)))
    return s


def _nelder_mead(f, x0: np.ndarray, step: float = 0.5, iters: int = 400, tol: float = 1e-9) -> np.ndarray:
    n = len(x0)
    pts = [x0] + [x0 + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * max(abs(vals[0]), 1e-300):
            break
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (pts[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    return pts[int(np.argmin(vals))]


def optimal_scales(terms: list[Term], s0: np.ndarray, max_spread: float = 1e3) -> np.ndarray:
    """Positive per-channel scales minimising :func:`objective`, started from ``s0``.

    Returns scales normalised to min 1, with spread (max/min) at most ``max_spread``. Never worse
    than ``|s0|`` under the model: the better of the two is returned.
    """
    s0 = np.abs(np.asarray(s0, np.float64))
    live = [t for t in terms if np.any(t.w > 0) and np.any(t.r > 0)]
    if not live or not any(not t.up for t in live):
        return s0 / s0.min()

    def maxima(s: np.ndarray) -> np.ndarray:
        return np.log(np.array([np.max(s * t.r) if t.up else np.max(t.r / s) for t in live]))

    def f(free: np.ndarray) -> float:
        log_m = np.concatenate([[anchor], free])
        return objective(live, _scales_for_maxima(live, log_m))

    # The maxima search is not convex: start it from several natural scale choices (the given
    # one, uniform, pure activation balance, pure weight balance) and keep the best.
    r_up = np.max([t.r for t in live if t.up], axis=0)
    r_dn = np.max([t.r for t in live if not t.up], axis=0)
    starts = [s0, np.ones_like(s0), 1.0 / np.maximum(r_up, 1e-12),
              np.sqrt(np.maximum(r_dn, 1e-12) / np.maximum(r_up, 1e-12))]
    candidates = [s0]
    for st in starts:
        m0 = maxima(st)
        anchor = m0[0]
        best = _nelder_mead(f, m0[1:]) if len(m0) > 1 else m0[1:]
        s = _scales_for_maxima(live, np.concatenate([[anchor], best]))
        # fixed-point polish: maxima from s, then the per-channel closed form, while it improves
        for _ in range(50):
            s_new = _scales_for_maxima(live, maxima(s))
            if objective(live, s_new) >= objective(live, s) * (1 - 1e-12):
                break
            s = s_new
        candidates.append(s)
    s = min(candidates, key=lambda c: objective(live, c))
    s = s / s.min()
    return np.minimum(s, max_spread)
