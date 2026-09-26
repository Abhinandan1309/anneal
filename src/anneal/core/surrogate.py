"""A sigmoid built from the HardSigmoids an AMD NPU/DPU actually runs.

The failure this fixes. AMD's NPU/DPU toolchain (AMD Quark, its XINT8 preset) replaces every
Sigmoid with the one gate the hardware has, HardSigmoid ``h(u) = clip(u/6 + 1/2, 0, 1)``, with
alpha = 1/6 and beta = 1/2 fixed. That swap happens before any quantization and alone costs
EfficientNet-B0 48.8pp and EfficientNet-B1 75.7pp top-1 in *float*: h is a poor sigmoid, and
EfficientNet applies it ~90 times (every SiLU and every squeeze-excite gate).

The rewrite. Replace each Sigmoid by a sum of the hardware's own HardSigmoids,

    s(x) = sum_{i=1..K} w_i * h(k_i * x + b_i),     w = softmax(logits),  k_i > 0,

built from element-wise Mul/Add and the fixed-alpha HardSigmoid only, so the DPU runs it
natively. The asymptotes are exact: sum(w) = 1, k > 0 and no constant offset, so s is exactly 0
far below zero and exactly 1 far above it, as sigmoid tends to. This matters more than the fit in
between: a 1-5% error at saturation compounds over ~90 gates and destroyed EfficientNet-B1.

k, b, w are scalars fitted *per gate* on that gate's own input distribution, measured on
calibration images. A SiLU gate (a Sigmoid whose output multiplies its own input) propagates
``x * (sigmoid(x) - s(x))``, so its loss is weighted by x^2; a squeeze-excite gate propagates the
error itself, unweighted. Measured in float on Imagenette against FP32 (examples/vitis/
surrogate_float.py): K=3 recovers EfficientNet-B0 to +0.00pp and EfficientNet-B1 to -2.0pp,
from AMD's -48.8pp and -75.7pp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

#: HardSigmoid as the DPU fixes it: h(u) = clip(ALPHA * u + BETA, 0, 1).
ALPHA, BETA = 1.0 / 6.0, 0.5
#: Gate input samples kept per gate for fitting (uniform over calibration batches).
MAX_POINTS = 20_000
#: Values drawn from each gate tensor per calibration batch before the reservoir.
PER_BATCH = 4096
#: Prefix of every tensor, node and initializer the rewrite adds.
PREFIX = "anneal_sur_"


def hard_sigmoid(u: np.ndarray) -> np.ndarray:
    return np.clip(u * ALPHA + BETA, 0.0, 1.0)


def surrogate(x: np.ndarray, w: np.ndarray, k: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``sum_i w_i * h(k_i x + b_i)`` evaluated in numpy (float64)."""
    x = np.asarray(x, dtype=np.float64)
    return sum(float(wi) * hard_sigmoid(float(ki) * x + float(bi)) for wi, ki, bi in zip(w, k, b))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


@dataclass
class Gate:
    node: str  # the Sigmoid node's name (may be empty in unnamed graphs)
    index: int  # its position in graph.node
    input: str
    output: str
    #: The output multiplies the gate's own input (possibly rescaled by a constant): a SiLU.
    silu: bool


def _scaled_sources(tensor: str, producer: dict[str, Any], consts: set[str]) -> set[str]:
    """``tensor`` and the tensors it is a constant multiple of (through Mul/Div by constants).

    Equalisation feeds a SiLU's gate ``x' * (1/s)`` while the Mul multiplies by ``x'``
    (see :mod:`anneal.core.equalize`); that is still a SiLU, up to a per-channel scale.
    """
    out = {tensor}
    while tensor in producer:
        nd = producer[tensor]
        if nd.op_type == "Mul" and len(nd.input) == 2:
            a, c = nd.input
            if c in consts:
                tensor = a
            elif a in consts:
                tensor = c
            else:
                break
        elif nd.op_type == "Div" and len(nd.input) == 2 and nd.input[1] in consts:
            tensor = nd.input[0]
        else:
            break
        out.add(tensor)
    return out


def sigmoid_gates(model) -> list[Gate]:
    """Every Sigmoid node: its input tensor, its output, and whether it is a SiLU gate.

    A SiLU gate's output is multiplied by the gate's own input (``x * sigmoid(x)``). A
    squeeze-excite gate's output multiplies a different tensor (the feature map it scales).
    """
    g = model.graph
    consts = {i.name for i in g.initializer} | {
        n.output[0] for n in g.node if n.op_type == "Constant"
    }
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list[Any]] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    gates = []
    for idx, nd in enumerate(g.node):
        if nd.op_type != "Sigmoid":
            continue
        x, y = nd.input[0], nd.output[0]
        sources = _scaled_sources(x, producer, consts)
        silu = False
        for c in consumers.get(y, []):
            if c.op_type == "Mul" and len(c.input) == 2:
                other = c.input[1] if c.input[0] == y else c.input[0]
                if other in sources:
                    silu = True
        gates.append(Gate(nd.name, idx, x, y, silu))
    return gates


# ---------------------------------------------------------------------------
# The fit (per gate)
# ---------------------------------------------------------------------------


def fit_surrogate(
    samples: np.ndarray,
    silu: bool,
    k_terms: int,
    seed: int = 0,
    *,
    restarts: int = 5,
    max_iter: int = 300,
    max_points: int = MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``(w, k, b)`` of a K-term surrogate to sigmoid on ``samples`` (a gate's inputs).

    Least squares on the samples, weighted by x^2 for a SiLU gate. Parameterised so the
    asymptotes are exact whatever the optimiser does: ``w = softmax(logits)`` (sum 1, all
    positive), ``k = exp(log_k)`` (positive, so every term saturates to 0 below and to 1 above).

    Levenberg-Marquardt with the analytic Jacobian (h is piecewise linear), from ``restarts``
    seeded random starts, the best kept: numpy only and deterministic for a given seed. On a
    20k-point gate distribution it takes under a second for K=3 and reached a lower loss than
    the Adam fit of the prototype (scratch/pwl_sigmoid_pergate.py), which took ~10 s.
    """
    if k_terms < 1:
        raise ValueError(f"k_terms must be at least 1, got {k_terms}")
    xs = np.asarray(samples, dtype=np.float64).ravel()
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        raise ValueError("fit_surrogate needs at least one finite sample")
    rng = np.random.default_rng(seed)
    if xs.size > max_points:
        xs = rng.choice(xs, max_points, replace=False)
    K, n = int(k_terms), xs.size
    weight = xs * xs if silu else np.ones_like(xs)
    # Normalised so the damping means the same on every gate.
    sw = np.sqrt(weight / max(float(weight.mean()), 1e-12))
    target = _sigmoid(xs)

    def evaluate(theta):
        z, log_k, b = theta[:K], theta[K:2 * K], theta[2 * K:]
        w = np.exp(z - z.max())
        w /= w.sum()
        k = np.exp(log_k)
        v = (k[:, None] * xs[None] + b[:, None]) * ALPHA + BETA
        h = np.clip(v, 0.0, 1.0)
        s = w @ h
        r = sw * (s - target)
        return w, k, v, h, s, r, float(r @ r) / n

    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for _ in range(restarts):
        theta = np.concatenate([np.zeros(K), np.log(rng.random(K) * 2 + 0.5), rng.standard_normal(K) * 2])
        w, k, v, h, s, r, f = evaluate(theta)
        lam = 1e-2
        for _ in range(max_iter):
            slope = ((v > 0) & (v < 1)) * ALPHA  # dh/du, zero where the term saturates
            jac = np.empty((n, 3 * K))
            jac[:, :K] = (w[:, None] * (h - s[None])).T  # softmax: d s / d z_j = w_j (h_j - s)
            jac[:, K:2 * K] = (w[:, None] * slope * k[:, None] * xs[None]).T
            jac[:, 2 * K:] = (w[:, None] * slope).T
            jac *= sw[:, None]
            a, g = jac.T @ jac, jac.T @ r
            damp = np.diag(np.diag(a)) + 1e-9 * np.eye(3 * K)
            for _ in range(10):
                trial = theta + np.linalg.solve(a + lam * damp, -g)
                cand = evaluate(trial)
                if cand[-1] < f:
                    break
                lam *= 4
            else:
                break  # no step reduces the loss: converged (or stuck at a kink)
            gain = (f - cand[-1]) / max(f, 1e-300)
            theta, (w, k, v, h, s, r, f) = trial, cand
            lam = max(lam / 3, 1e-9)
            if gain < 1e-10:
                break
        if best is None or f < best[0]:
            best = (f, w.copy(), k.copy(), theta[2 * K:].copy())
    assert best is not None
    _, w, k, b = best
    return w, k, b


def _loss(xs: np.ndarray, silu: bool, w, k, b) -> float:
    """The fit's objective, unnormalised: mean of (x * err)^2 for SiLU, err^2 otherwise."""
    err = surrogate(xs, w, k, b) - _sigmoid(xs)
    return float(np.mean((xs * err) ** 2 if silu else err ** 2))


# ---------------------------------------------------------------------------
# The rewrite
# ---------------------------------------------------------------------------


@dataclass
class SurrogateGate:
    node: str
    input: str
    silu: bool
    w: list[float]
    k: list[float]
    b: list[float]
    #: The fit's objective on the samples (x^2-weighted for SiLU gates).
    loss: float
    #: max |sigmoid(x) - s(x)| over the samples.
    max_abs_error: float
    #: max |x * (sigmoid(x) - s(x))| over the samples: the error a SiLU passes on.
    max_abs_silu_error: float
    x_min: float
    x_max: float
    samples: int
    #: Nodes that replaced the Sigmoid (to keep in float, like the gate they replace).
    nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def r(v):
            if isinstance(v, float):
                return float(f"{v:.6g}")
            if isinstance(v, list) and v and isinstance(v[0], float):
                return [float(f"{x:.6g}") for x in v]
            return v

        return {k: r(v) for k, v in self.__dict__.items()}


@dataclass
class SurrogateResult:
    k_terms: int
    gates: list[SurrogateGate] = field(default_factory=list)
    #: max |output change| of the model on the check batch (float, before quantization).
    max_abs_output_change: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "k_terms": self.k_terms,
            "gates": len(self.gates),
            "silu_gates": sum(g.silu for g in self.gates),
            "median_loss": float(np.median([g.loss for g in self.gates])) if self.gates else None,
            "max_abs_error": max((g.max_abs_error for g in self.gates), default=None),
            "max_abs_output_change": self.max_abs_output_change,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary(), "per_gate": [g.to_dict() for g in self.gates]}


class _Reservoir:
    """Uniform sample of at most ``size`` values from a stream of equal-sized chunks."""

    def __init__(self, size: int, rng: np.random.Generator):
        self.size, self.rng = size, rng
        self.buf = np.empty(0, dtype=np.float32)
        self.seen = 0

    def add(self, v: np.ndarray) -> None:
        room = self.size - self.buf.size
        if room > 0:
            self.buf = np.concatenate([self.buf, v[:room]])
            self.seen += min(room, v.size)
            v = v[room:]
        if v.size == 0:
            return
        idx = self.seen + np.arange(1, v.size + 1)  # 1-based stream position of each value
        self.seen += v.size
        slot = (self.rng.random(v.size) * idx).astype(np.int64)  # uniform in [0, idx)
        keep = slot < self.size
        self.buf[slot[keep]] = v[keep]


def gate_samples(
    model, gates: list[Gate], batches: Iterable[np.ndarray], *, seed: int = 0
) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]], np.ndarray | None]:
    """Each gate input's values on the calibration batches (subsampled), its full range, and
    the first batch (kept for checking the rewrite)."""
    import onnx
    import onnxruntime as ort
    from onnx import helper

    tensors = list(dict.fromkeys(g.input for g in gates))
    graph_inputs = {i.name for i in model.graph.input}
    fetch = [t for t in tensors if t not in graph_inputs]
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    existing = {o.name for o in probe.graph.output}
    probe.graph.output.extend(
        [helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None) for t in fetch if t not in existing]
    )
    opts = ort.SessionOptions()
    # Nothing may be fused away: every gate input must exist as computed.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(probe.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(seed)
    res = {t: _Reservoir(MAX_POINTS, rng) for t in tensors}
    lo = {t: np.inf for t in tensors}
    hi = {t: -np.inf for t in tensors}
    first = None
    for batch in batches:
        if first is None:
            first = batch
        values = dict(zip(fetch, session.run(fetch, {input_name: batch}) if fetch else []))
        if input_name in tensors:
            values[input_name] = batch
        for t in tensors:
            v = np.asarray(values[t], dtype=np.float32).ravel()
            if v.size == 0:
                continue
            lo[t], hi[t] = min(lo[t], float(v.min())), max(hi[t], float(v.max()))
            if v.size > PER_BATCH:
                v = v[rng.choice(v.size, PER_BATCH, replace=False)]
            res[t].add(v)
    if first is None:
        raise ValueError("the sigmoid surrogate needs calibration batches; none were given")
    return {t: r.buf for t, r in res.items()}, {t: (lo[t], hi[t]) for t in tensors}, first


def _unique(base: str, taken: set[str]) -> str:
    name = base
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def replace_sigmoids(
    src: Path,
    dst: Path,
    batches: Iterable[np.ndarray],
    k_terms: int = 3,
    *,
    seed: int = 0,
    check: bool = True,
) -> dict[str, Any]:
    """Replace every Sigmoid in ``src`` by its fitted HardSigmoid-sum surrogate; save ``dst``.

    Each Sigmoid becomes, per term i, ``Mul(x, k_i) -> Add(b_i) -> HardSigmoid(1/6, 1/2) ->
    Mul(w_i)``, the terms summed by a chain of Adds whose last output keeps the Sigmoid's output
    name, so nothing downstream changes. (For K=1, w=1 exactly and its Mul is left out.) The
    scale and shift are explicit Mul/Add rather than folded into HardSigmoid's alpha/beta,
    because the DPU fixes alpha = 1/6 and beta = 1/2.

    Returns a report: per gate the fitted parameters, the fit loss and the max abs error on the
    samples, plus (``check``) the max abs output change on the first calibration batch.
    """
    import onnx
    from onnx import helper, numpy_helper

    src, dst = Path(src), Path(dst)
    model = onnx.load(str(src))
    gates = sigmoid_gates(model)
    result = SurrogateResult(k_terms=int(k_terms))
    first = None
    if gates:
        samples, ranges, first = gate_samples(model, gates, batches, seed=seed)
    g = model.graph
    taken = {n.name for n in g.node} | {o for n in g.node for o in n.output} | {
        i.name for i in g.initializer} | {i.name for i in g.input}
    fits: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    replacement: dict[int, list[Any]] = {}
    for gi, gate in enumerate(gates):
        xs = samples[gate.input]
        key = (gate.input, gate.silu)
        if key not in fits:
            fits[key] = fit_surrogate(xs, gate.silu, k_terms, seed=seed)
        w, k, b = fits[key]
        w = (w / w.sum()).astype(np.float32)  # sum exactly 1 as stored
        k, b = k.astype(np.float32), b.astype(np.float32)
        base = f"{PREFIX}{gi}"
        nodes, terms = [], []
        for j in range(len(w)):
            p = f"{base}_{j}"
            names = {nm: _unique(f"{p}_{nm}", taken) for nm in ("k", "b", "w", "kx", "u", "h", "t")}
            g.initializer.extend([numpy_helper.from_array(np.array(k[j], np.float32), names["k"]),
                                  numpy_helper.from_array(np.array(b[j], np.float32), names["b"])])
            last = len(w) == 1
            h_out = gate.output if last else names["h"]
            nodes += [
                helper.make_node("Mul", [gate.input, names["k"]], [names["kx"]], name=_unique(f"{p}_mul_k", taken)),
                helper.make_node("Add", [names["kx"], names["b"]], [names["u"]], name=_unique(f"{p}_add_b", taken)),
                helper.make_node("HardSigmoid", [names["u"]], [h_out], name=_unique(f"{p}_hardsigmoid", taken),
                                 alpha=ALPHA, beta=BETA),
            ]
            if not last:
                g.initializer.append(numpy_helper.from_array(np.array(w[j], np.float32), names["w"]))
                nodes.append(helper.make_node("Mul", [names["h"], names["w"]], [names["t"]],
                                              name=_unique(f"{p}_mul_w", taken)))
                terms.append(names["t"])
        acc = terms[0] if terms else None
        for j, t in enumerate(terms[1:], start=1):
            out = gate.output if j == len(terms) - 1 else _unique(f"{base}_acc{j}", taken)
            nodes.append(helper.make_node("Add", [acc, t], [out], name=_unique(f"{base}_sum{j}", taken)))
            acc = out
        replacement[gate.index] = nodes
        xs64 = xs.astype(np.float64)
        err = surrogate(xs64, w, k, b) - _sigmoid(xs64)
        result.gates.append(SurrogateGate(
            node=gate.node, input=gate.input, silu=gate.silu,
            w=[float(v) for v in w], k=[float(v) for v in k], b=[float(v) for v in b],
            loss=_loss(xs64, gate.silu, w, k, b),
            max_abs_error=float(np.abs(err).max()),
            max_abs_silu_error=float(np.abs(xs64 * err).max()),
            x_min=ranges[gate.input][0], x_max=ranges[gate.input][1], samples=int(xs.size),
            nodes=[n.name for n in nodes],
        ))

    new_nodes = []
    for i, nd in enumerate(g.node):
        new_nodes.extend(replacement.get(i, [nd]))
    del g.node[:]
    g.node.extend(new_nodes)
    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    if check and first is not None:
        result.max_abs_output_change = _output_change(src, dst, first)
    return result.to_dict()


def _output_change(a: Path, b: Path, batch: np.ndarray) -> float:
    import onnxruntime as ort

    outs = []
    for p in (a, b):
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        outs.append(s.run(None, {s.get_inputs()[0].name: batch}))
    return float(max(np.abs(x - y).max() for x, y in zip(*outs)))
