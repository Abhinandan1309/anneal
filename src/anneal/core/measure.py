"""The measurement harness.

This module is the reason the rest of the project is trustworthy. Optimisation tools
love to report "2.3x faster" from a single timed call on a warm cache. Here:

* warm-up iterations are run and **discarded** before anything is recorded;
* latency is reported as p50/p90/p99, not a mean, because tail latency is what an edge
  deadline actually cares about;
* the execution provider that *actually* ran is recorded, not the one that was requested
  (onnxruntime silently falls back to CPU, and silent fallback has ruined more edge
  benchmarks than bad kernels have);
* every candidate is checked for numerical agreement against the baseline, so a transform
  that makes the model fast by making it wrong gets caught rather than celebrated.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence

import numpy as np

from anneal.core.artifact import ModelArtifact, concrete_input_shape, describe_io
from anneal.core.targets import Target

#: Discarded iterations before timing starts. ORT allocates arenas and picks kernels on
#: the first few calls; timing those measures the allocator, not the model.
DEFAULT_WARMUP = 10
DEFAULT_RUNS = 60


@dataclass(frozen=True)
class Measurement:
    """Everything measured about one candidate. No field here is an estimate."""

    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p99: float
    latency_ms_mean: float
    latency_ms_std: float
    n_runs: int
    n_warmup: int
    batch_size: int
    size_bytes: int
    providers_used: tuple[str, ...]
    load_time_ms: float
    accuracy: float | None = None
    n_eval: int = 0
    top1_agreement: float | None = None
    logit_cosine: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def throughput_ips(self) -> float:
        """Inferences per second at the measured median latency."""
        if self.latency_ms_p50 <= 0:
            return float("nan")
        return self.batch_size * 1000.0 / self.latency_ms_p50

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["providers_used"] = list(self.providers_used)
        d["size_mb"] = round(self.size_mb, 4)
        d["throughput_ips"] = round(self.throughput_ips, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Measurement:
        known = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in d.items() if k in known}
        payload["providers_used"] = tuple(payload.get("providers_used", ()))
        return cls(**payload)


class MeasurementError(RuntimeError):
    """A candidate could not be measured (usually: the graph does not load or run)."""


def _percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. Avoids interpolating between two real observations."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(q / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


class Benchmarker:
    """Measures artifacts against one target with a fixed, reused input tensor.

    The same input bytes are fed to every candidate. That removes input variance as an
    explanation for a latency difference between two trials.
    """

    def __init__(
        self,
        target: Target,
        *,
        batch_size: int = 1,
        warmup: int = DEFAULT_WARMUP,
        runs: int = DEFAULT_RUNS,
        seed: int = 0,
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        target.ensure_available()
        self.target = target
        self.batch_size = batch_size
        self.warmup = warmup
        self.runs = runs
        self.seed = seed
        self._input_shape_override = input_shape
        self._bench_input: dict[str, np.ndarray] | None = None
        self._baseline_logits: np.ndarray | None = None
        #: The batch size actually fed, which a model with a fixed batch axis dictates
        #: regardless of what was requested. Reported instead of the request so
        #: throughput numbers stay true.
        self._effective_batch: int | None = None

    # ----- session --------------------------------------------------------

    def _session(self, artifact: ModelArtifact):
        import onnxruntime as ort

        try:
            return ort.InferenceSession(
                str(artifact.path),
                sess_options=self.target.session_options(),
                providers=list(self.target.providers),
            )
        except Exception as exc:  # onnxruntime raises a zoo of exception types
            raise MeasurementError(f"could not load {artifact.path.name}: {exc}") from exc

    def _make_input(self, artifact: ModelArtifact) -> dict[str, np.ndarray]:
        """Build (once) the fixed synthetic tensor used for latency timing."""
        if self._bench_input is not None:
            return self._bench_input

        io = describe_io(artifact.path)
        if len(io["inputs"]) != 1:
            raise MeasurementError(
                f"expected exactly one model input, found {len(io['inputs'])}: "
                f"{[i['name'] for i in io['inputs']]}"
            )
        spec = io["inputs"][0]
        shape = self._input_shape_override or concrete_input_shape(spec, self.batch_size)
        rng = np.random.default_rng(self.seed)
        # Standard-normal matches the distribution of normalised image tensors well
        # enough that denormal-float slow paths do not distort the timing.
        self._bench_input = {spec["name"]: rng.standard_normal(shape, dtype=np.float32)}
        self._effective_batch = int(shape[0])
        return self._bench_input

    # ----- core measurement ----------------------------------------------

    def measure(
        self,
        artifact: ModelArtifact,
        *,
        evalset: "EvalSet | None" = None,
        record_baseline: bool = False,
    ) -> Measurement:
        """Load, time, and (optionally) score one artifact."""
        t0 = time.perf_counter()
        session = self._session(artifact)
        load_ms = (time.perf_counter() - t0) * 1000.0

        providers_used = tuple(session.get_providers())
        requested = self.target.providers[0]
        if requested not in providers_used:
            # Silent EP fallback is the single most common cause of bogus edge numbers.
            raise MeasurementError(
                f"target {self.target.name!r} requested {requested} but onnxruntime "
                f"is running {providers_used}; refusing to report this as a "
                f"{self.target.name} measurement"
            )

        feed = self._make_input(artifact)
        output_names = [o.name for o in session.get_outputs()]

        for _ in range(self.warmup):
            session.run(output_names, feed)

        samples: list[float] = []
        for _ in range(self.runs):
            start = time.perf_counter_ns()
            session.run(output_names, feed)
            samples.append((time.perf_counter_ns() - start) / 1e6)

        accuracy: float | None = None
        n_eval = 0
        agreement: float | None = None
        cosine: float | None = None

        if evalset is not None:
            try:
                accuracy, n_eval, logits = self._score(session, evalset, output_names)
            except Exception as exc:  # a broken candidate must not kill the whole run
                raise MeasurementError(
                    f"{artifact.path.name} ran the latency benchmark but failed during "
                    f"evaluation: {exc}"
                ) from exc
            if record_baseline:
                self._baseline_logits = logits
            elif self._baseline_logits is not None and logits is not None:
                agreement, cosine = _compare_to_baseline(self._baseline_logits, logits)

        del session  # free the arena before the next candidate is loaded

        return Measurement(
            latency_ms_p50=_percentile(samples, 50),
            latency_ms_p90=_percentile(samples, 90),
            latency_ms_p99=_percentile(samples, 99),
            latency_ms_mean=statistics.fmean(samples),
            latency_ms_std=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
            n_runs=len(samples),
            n_warmup=self.warmup,
            batch_size=self._effective_batch or self.batch_size,
            size_bytes=artifact.size_bytes,
            providers_used=providers_used,
            load_time_ms=load_ms,
            accuracy=accuracy,
            n_eval=n_eval,
            top1_agreement=agreement,
            logit_cosine=cosine,
        )

    def _score(
        self, session, evalset: "EvalSet", output_names: list[str]
    ) -> tuple[float, int, np.ndarray | None]:
        """Run the eval set and return (top-1 accuracy, n, stacked logits)."""
        input_name = session.get_inputs()[0].name
        correct = 0
        total = 0
        collected: list[np.ndarray] = []

        for x, y in evalset.batches():
            out = session.run(output_names, {input_name: x})[0]
            logits = np.asarray(out, dtype=np.float32)
            collected.append(logits)
            pred = evalset.decode(logits)
            correct += int((pred == y).sum())
            total += int(y.shape[0])

        if total == 0:
            return 0.0, 0, None
        return correct / total, total, np.concatenate(collected, axis=0)


def _compare_to_baseline(
    baseline: np.ndarray, candidate: np.ndarray
) -> tuple[float | None, float | None]:
    """How much did the transform actually change the model's behaviour?

    ``top1_agreement`` is the fraction of eval samples where the candidate picks the same
    class as the baseline; ``logit_cosine`` is the mean cosine similarity of the raw
    output vectors. Accuracy alone can hide a model that got a different set of answers
    right by luck — these two catch that.
    """
    if baseline.shape != candidate.shape:
        return None, None

    b_top = baseline.argmax(axis=-1)
    c_top = candidate.argmax(axis=-1)
    agreement = float((b_top == c_top).mean())

    b = baseline.reshape(baseline.shape[0], -1).astype(np.float64)
    c = candidate.reshape(candidate.shape[0], -1).astype(np.float64)
    denom = np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    safe = denom > 0
    if not safe.any():
        return agreement, None
    cos = np.einsum("ij,ij->i", b[safe], c[safe]) / denom[safe]
    return agreement, float(cos.mean())


class EvalSet:
    """Minimal protocol for a labelled evaluation set.

    ``batches()`` yields ``(x, y)`` where ``x`` is a float32 array shaped for the model
    input and ``y`` is an int array of ground-truth labels. ``decode()`` maps raw model
    output to predicted labels, which is where a class-subset mapping lives.
    """

    def batches(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:  # pragma: no cover
        raise NotImplementedError

    def decode(self, logits: np.ndarray) -> np.ndarray:  # pragma: no cover
        return logits.argmax(axis=-1)

    def calibration_batches(self, limit: int) -> Iterable[np.ndarray]:  # pragma: no cover
        """Unlabelled inputs for static quantization calibration."""
        seen = 0
        for x, _ in self.batches():
            if seen >= limit:
                return
            yield x
            seen += x.shape[0]

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError
