"""Search policies: what to try next, given what has been measured.

Two implementations ship:

:class:`HeuristicPolicy`
    A curriculum distilled from how edge engineers actually work: probe the families of
    transform, look at what the measurements say, then escalate selective quantization
    if accuracy broke or tune for speed if it did not. Deterministic, needs no API key,
    and is the honest baseline the LLM policy has to beat.

:class:`ClaudePolicy`
    Hands the measured ledger to Claude each turn and lets it choose. It sees the same
    numbers the heuristic does and nothing else — no hidden hints — so a comparison
    between the two is meaningful.

Keeping both is the point. An agentic system that cannot be compared against a competent
non-agentic baseline is a demo, not an engineering result.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from anneal.core.artifact import ModelArtifact, TransformRecord
from anneal.core.ledger import Ledger, Trial
from anneal.core.transforms import TransformSpec

BASELINE = -1


@dataclass(frozen=True)
class Proposal:
    """One move: apply ``transform`` to the model produced by trial ``base_index``."""

    transform: str
    params: dict[str, Any] = field(default_factory=dict)
    base_index: int = BASELINE
    rationale: str = ""

    def preview_key(self, base: ModelArtifact) -> str:
        record = TransformRecord(self.transform, self.params)
        return "|".join([*(str(t) for t in base.lineage), str(record)])


@dataclass(frozen=True)
class Constraints:
    """The engineer's actual requirements."""

    max_accuracy_drop_pp: float | None = 1.0
    min_accuracy: float | None = None
    max_size_bytes: int | None = None
    max_latency_ms: float | None = None

    def describe(self) -> str:
        parts = []
        if self.max_accuracy_drop_pp is not None:
            parts.append(f"accuracy may drop at most {self.max_accuracy_drop_pp:.2f}pp vs baseline")
        if self.min_accuracy is not None:
            parts.append(f"absolute accuracy >= {self.min_accuracy * 100:.2f}%")
        if self.max_size_bytes is not None:
            parts.append(f"size <= {self.max_size_bytes / 1e6:.1f}MB")
        if self.max_latency_ms is not None:
            parts.append(f"p50 latency <= {self.max_latency_ms:.2f}ms")
        return "; ".join(parts) or "none (explore the whole frontier)"


@dataclass
class SearchState:
    """What a policy is allowed to see."""

    ledger: Ledger
    transforms: dict[str, TransformSpec]
    budget_remaining: int
    constraints: Constraints
    baseline_artifact: ModelArtifact
    #: Whether a measured layer-sensitivity ranking can be used this run — an eval set to
    #: sweep against, or a saved sweep. The policy prefers it when it can.
    can_measure_sensitivity: bool = False
    #: The target's INT8 arithmetic, from anneal.core.environment.cpu_features():
    #: "x86-avx2-16bit" is the path whose 16-bit pair sums can saturate.
    int8_path: str = "unknown"
    #: Conv -> gated activation/ReLU -> depthwise chains in the baseline that equalisation
    #: can rebalance (anneal.core.equalize). Zero for nets without depthwise convolutions.
    equalisable_sites: int = 0

    def artifact(self, index: int) -> ModelArtifact | None:
        if index == BASELINE:
            return self.baseline_artifact
        for trial in self.ledger.trials:
            if trial.index == index and trial.ok:
                return trial.artifact
        return None

    def accuracy_floor(self) -> float | None:
        """Lowest acceptable absolute accuracy, combining both constraint forms."""
        floors = []
        if self.constraints.min_accuracy is not None:
            floors.append(self.constraints.min_accuracy)
        base = self.ledger.baseline
        if (
            self.constraints.max_accuracy_drop_pp is not None
            and base is not None
            and base.measurement is not None
            and base.measurement.accuracy is not None
        ):
            floors.append(
                base.measurement.accuracy - self.constraints.max_accuracy_drop_pp / 100.0
            )
        return max(floors) if floors else None

    def meets_floor(self, trial: Trial) -> bool:
        floor = self.accuracy_floor()
        if floor is None:
            return True
        if trial.measurement is None or trial.measurement.accuracy is None:
            return False
        return trial.measurement.accuracy >= floor


class Policy(Protocol):
    name: str

    def propose(self, state: SearchState) -> Proposal | None:
        """Next move, or ``None`` to stop the search."""
        ...


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------


class HeuristicPolicy:
    """Deterministic, measurement-driven curriculum. No API key required."""

    name = "heuristic"

    def __init__(self) -> None:
        self._queue: list[Proposal] = []
        self._generation = 0
        self._stop_reason = ""

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    def propose(self, state: SearchState) -> Proposal | None:
        while True:
            found = self._pop_fresh(state)
            if found is not None:
                return found
            if not self._advance(state):
                self._stop_reason = self._stop_reason or "curriculum exhausted"
                return None

    # ----- queue management ----------------------------------------------

    def _pop_fresh(self, state: SearchState) -> Proposal | None:
        attempted = state.ledger.attempted_recipes()
        while self._queue:
            proposal = self._queue.pop(0)
            base = state.artifact(proposal.base_index)
            if base is None:
                continue
            if proposal.transform not in state.transforms:
                continue
            if proposal.preview_key(base) in attempted:
                continue
            return proposal
        return None

    def _advance(self, state: SearchState) -> bool:
        """Generate the next batch of proposals from what has been measured."""
        gen = self._generation
        self._generation += 1
        if gen == 0:
            self._queue.extend(self._probe())
            return True
        if gen == 1:
            self._queue.extend(self._react(state))
            return True
        if gen == 2:
            self._queue.extend(self._escalate(state))
            return True
        if gen == 3:
            self._queue.extend(self._polish(state))
            return True
        return False

    # ----- generations ----------------------------------------------------

    def _probe(self) -> list[Proposal]:
        """One representative from each family, all applied to the baseline."""
        return [
            Proposal(
                "graph_optimize",
                {"level": "all"},
                BASELINE,
                "Lossless baseline: establish what fusion alone buys before touching precision.",
            ),
            Proposal(
                "quantize_dynamic_int8",
                {"per_channel": True, "reduce_range": False, "weight_type": "int8"},
                BASELINE,
                "Cheapest INT8 path; no calibration needed. Probes whether this target has "
                "INT8 kernels worth using.",
            ),
            Proposal(
                "quantize_static_int8",
                {
                    "per_channel": True,
                    "reduce_range": False,
                    "calibrate_method": "minmax",
                    "calib_samples": 64,
                    "activation_type": "uint8",
                },
                BASELINE,
                "Full INT8 with real calibration — the largest potential latency win and "
                "the largest accuracy risk.",
            ),
        ]

    def _react(self, state: SearchState) -> list[Proposal]:
        out: list[Proposal] = []
        opt_idx = self._index_of(state, "graph_optimize")

        # A quantized model that came out *slower* than FP32 is the most informative
        # failure available: it means this target's INT8 path is not what the textbook
        # assumes. Per-channel scales are the usual culprit, so retry per-tensor before
        # concluding anything about INT8 on this hardware.
        for trial in state.ledger.successful():
            if not trial.artifact.lineage:
                continue
            speedup = state.ledger.speedup_of(trial)
            if speedup is None or speedup >= 0.95:
                continue
            head = trial.artifact.lineage[-1]
            if head.params.get("per_channel") is not True:
                continue
            out.append(
                Proposal(
                    head.name,
                    {**head.params, "per_channel": False},
                    BASELINE,
                    f"Trial [{trial.index}] came out {1 / speedup:.1f}x SLOWER than FP32. "
                    f"Per-channel dynamic scales are the usual cause on CPU targets; "
                    f"retry with per-tensor scales before writing off this transform.",
                )
            )

        broke_accuracy = [
            t for t in state.ledger.successful() if t.artifact.lineage and not state.meets_floor(t)
        ]

        if broke_accuracy:
            # A static INT8 model that broke accuracy is retried with 7-bit weights and with
            # per-tensor scales first. On ResNet-18 / Zen 2 the full-range per-channel recipe
            # lost 4.2pp while the reduce_range variant lost nothing at the same speed —
            # consistent with the intermediate saturation onnxruntime documents for x86 CPUs
            # without VNNI. These are fast kernels, so they are tried before the slow
            # selective-dynamic path.
            for trial in broke_accuracy:
                head = trial.artifact.lineage[-1]
                if head.name != "quantize_static_int8" or len(trial.artifact.lineage) != 1:
                    continue
                if (
                    state.equalisable_sites
                    and head.params.get("per_channel")
                    and not head.params.get("equalize")
                ):
                    # Tried first: unlike the guard, this failure is not specific to one CPU.
                    for float_gates in (False, True):
                        out.append(
                            Proposal(
                                "quantize_static_int8",
                                {**head.params, "equalize": True, "equalize_slack": 0.1,
                                 "float_gates": float_gates},
                                BASELINE,
                                f"Trial [{trial.index}] broke accuracy, and the model has "
                                f"{state.equalisable_sites} conv -> activation -> depthwise "
                                f"chains. One shared activation scale starves the small channels "
                                f"there and the depthwise conv amplifies it, on every CPU. "
                                f"Rescale channels exactly (float model unchanged) first"
                                + ("; keep the gate branches in float." if float_gates else "."),
                            )
                        )
                if (
                    state.int8_path == "x86-avx2-16bit"
                    and not head.params.get("reduce_range")
                    and not head.params.get("guard_saturation")
                ):
                    out.append(
                        Proposal(
                            "quantize_static_int8",
                            {**head.params, "guard_saturation": True, "saturation_tolerance": 0.02},
                            BASELINE,
                            f"Trial [{trial.index}] broke accuracy on an x86 CPU without VNNI, "
                            f"whose INT8 path sums pairs in saturating 16-bit arithmetic. Emulate "
                            f"that arithmetic and keep only the layers that saturate in FP32, "
                            f"leaving every other layer at full 8-bit precision.",
                        )
                    )
                if head.params.get("per_channel") and not head.params.get("reduce_range"):
                    out.append(
                        Proposal(
                            "quantize_static_int8",
                            {**head.params, "reduce_range": True},
                            BASELINE,
                            f"Trial [{trial.index}] (static INT8, full-range per-channel "
                            f"weights) fell below the accuracy floor. 7-bit weights avoid the "
                            f"int16 saturation that full-range per-channel scales invite on "
                            f"CPUs without VNNI; retry with reduce_range.",
                        )
                    )
                if head.params.get("per_channel"):
                    out.append(
                        Proposal(
                            "quantize_static_int8",
                            {**head.params, "per_channel": False},
                            BASELINE,
                            f"Trial [{trial.index}] broke accuracy with per-channel scales; "
                            f"per-tensor scales keep most weights well below full range.",
                        )
                    )

            # Then spare the worst layers and re-measure.
            for k in (1, 2, 4):
                out.append(
                    Proposal(
                        "quantize_dynamic_sensitive",
                        {"skip_top_k": k, "skip_first_last": False, "per_channel": True,
                         "ranking": self._ranking(state)},
                        BASELINE,
                        f"{len(broke_accuracy)} quantized candidate(s) fell below the accuracy "
                        f"floor; spare the {k} most quantization-sensitive layer(s) and re-measure.",
                    )
                )
        else:
            # Accuracy held. Push harder on speed.
            out.append(
                Proposal(
                    "quantize_dynamic_sensitive",
                    {"skip_top_k": 0, "skip_first_last": False, "per_channel": True,
                     "ranking": self._ranking(state)},
                    BASELINE,
                    "Accuracy held everywhere; check whether sparing zero layers (blanket "
                    "quantization via the selective path) is faster still.",
                )
            )
            out.append(
                Proposal(
                    "quantize_static_int8",
                    {
                        "per_channel": False,
                        "reduce_range": False,
                        "calibrate_method": "minmax",
                        "calib_samples": 64,
                        "activation_type": "uint8",
                    },
                    BASELINE,
                    "Per-tensor scales drop the per-channel dequant overhead; accuracy budget "
                    "has room for it.",
                )
            )

        # Either way, stacking fusion under the best quantization is worth one trial. The
        # candidate must be a *quantization*: when fusion itself is the fastest thing found
        # (as on MobileNetV3, where every INT8 variant was slower than FP32), "stack the
        # winner on fusion" would mean fusing twice — a guaranteed-wasted trial.
        quantized = [
            t
            for t in state.ledger.successful()
            if t.artifact.lineage and t.artifact.lineage[-1].name != "graph_optimize"
        ]
        best_quant = min(
            quantized, key=lambda t: t.measurement.latency_ms_p50, default=None  # type: ignore[union-attr]
        )
        if opt_idx is not None and best_quant is not None:
            head = best_quant.artifact.lineage[-1]
            out.append(
                Proposal(
                    head.name,
                    dict(head.params),
                    opt_idx,
                    f"Trial [{best_quant.index}] was the fastest quantization; re-apply it on "
                    f"top of the graph-optimised graph to see if fusion and quantization compose.",
                )
            )
        return out

    def _escalate(self, state: SearchState) -> list[Proposal]:
        out: list[Proposal] = []
        passing = [
            t for t in state.ledger.successful() if t.artifact.lineage and state.meets_floor(t)
        ]

        if not passing:
            # Nothing has cleared the bar yet — spare aggressively.
            for k in (8, 16):
                out.append(
                    Proposal(
                        "quantize_dynamic_sensitive",
                        {"skip_top_k": k, "skip_first_last": True, "per_channel": True,
                         "ranking": self._ranking(state)},
                        BASELINE,
                        f"Still nothing inside the accuracy budget; spare the top {k} "
                        f"sensitive layers plus stem and classifier.",
                    )
                )
        else:
            out.append(
                Proposal(
                    "quantize_static_int8",
                    {
                        "per_channel": True,
                        "reduce_range": False,
                        "calibrate_method": "entropy",
                        "calib_samples": 64,
                        "activation_type": "uint8",
                    },
                    BASELINE,
                    "Entropy calibration clips activation outliers that MinMax stretches the "
                    "scale to cover; usually recovers accuracy at equal latency.",
                )
            )
            out.append(
                Proposal(
                    "quantize_dynamic_sensitive",
                    {"skip_top_k": 1, "skip_first_last": True, "per_channel": True,
                     "ranking": self._ranking(state)},
                    BASELINE,
                    "Cheap frontier point: spare only the stem, classifier and single worst "
                    "layer — should sit between blanket INT8 and FP32.",
                )
            )
        return out

    def _polish(self, state: SearchState) -> list[Proposal]:
        """One last attempt to extend the frontier at the fast end."""
        opt_idx = self._index_of(state, "graph_optimize")
        if opt_idx is None:
            return []
        return [
            Proposal(
                "quantize_static_int8",
                {
                    "per_channel": True,
                    "reduce_range": True,
                    "calibrate_method": "minmax",
                    "calib_samples": 64,
                    "activation_type": "uint8",
                },
                opt_idx,
                "Final probe: reduced-range static INT8 over the fused graph, the most "
                "aggressive combination in the action space.",
            )
        ]

    # ----- ledger queries -------------------------------------------------

    @staticmethod
    def _ranking(state: SearchState) -> str:
        # The weight-error proxy scored Spearman +0.33 against a measured sweep on
        # ResNet-18 and ranked the most damaging layer last. Use measurement when it exists.
        return "measured" if state.can_measure_sensitivity else "proxy"

    @staticmethod
    def _index_of(state: SearchState, transform: str) -> int | None:
        for trial in state.ledger.successful():
            if len(trial.artifact.lineage) == 1 and trial.artifact.lineage[0].name == transform:
                return trial.index
        return None


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class ClaudePolicy:
    """Lets Claude choose the next transform from the measured ledger."""

    name = "claude"

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        api_key: str | None = None,
        max_tokens: int = 1024,
        client: Any = None,
    ) -> None:
        #: ``client`` is injectable so the proposal-parsing logic can be tested without
        #: network access — that parsing is where the bugs live, not in the HTTP call.
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "the Claude policy needs the optional llm extra: pip install 'anneal[llm]'"
                ) from exc

            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Either export it, or run with "
                    "--policy heuristic, which needs no API access."
                )
            client = anthropic.Anthropic(api_key=key)

        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._stop_reason = ""
        #: Kept for the report, so a reader can audit what the model was thinking.
        self.transcript: list[dict[str, Any]] = []

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    def propose(self, state: SearchState) -> Proposal | None:
        from anneal.agent import prompts

        user = prompts.render_state(
            state.ledger,
            state.transforms,
            state.budget_remaining,
            state.constraints.describe(),
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=prompts.SYSTEM,
            tools=prompts.build_tools(state.transforms),
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user}],
        )

        block = next((b for b in response.content if b.type == "tool_use"), None)
        if block is None:
            self._stop_reason = "model returned no tool call"
            return None

        self.transcript.append(
            {
                "trial_index": state.ledger.next_index(),
                "tool": block.name,
                "input": block.input,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }
        )

        if block.name == "stop":
            self._stop_reason = str(block.input.get("reason", "model chose to stop"))
            return None

        data = block.input
        params = data.get("params") or {}
        if isinstance(params, str):
            # Models occasionally hand back a JSON string for an object-typed field.
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}

        base_index = int(data.get("base_trial", BASELINE))
        return Proposal(
            transform=str(data["transform"]),
            params=dict(params),
            base_index=base_index,
            rationale=str(data.get("rationale", "")),
        )


def build_policy(name: str, *, model: str = "claude-sonnet-5") -> Policy:
    if name == "heuristic":
        return HeuristicPolicy()
    if name == "claude":
        return ClaudePolicy(model=model)
    raise ValueError(f"unknown policy {name!r}; expected 'heuristic' or 'claude'")
