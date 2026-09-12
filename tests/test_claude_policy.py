"""The Claude policy's parsing layer, exercised with a stub client.

The HTTP call is not the interesting part. What matters is that a tool call from the
model becomes a well-formed Proposal, that a malformed one degrades safely, and that the
rendered state the model sees actually contains the measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from anneal.agent import prompts
from anneal.agent.policy import BASELINE, ClaudePolicy, Constraints, SearchState
from anneal.core.artifact import ModelArtifact, TransformRecord
from anneal.core.ledger import Ledger
from anneal.core.transforms import REGISTRY
from conftest import make_trial


@dataclass
class _Block:
    type: str
    name: str
    input: dict[str, Any]


@dataclass
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class _Response:
    content: list[Any]
    usage: _Usage


class StubClient:
    """Returns canned tool-use responses and records what it was asked."""

    def __init__(self, *blocks: Any) -> None:
        self._blocks = list(blocks)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        block = self._blocks.pop(0)
        return _Response(content=[block], usage=_Usage())


def state_with_measurements() -> SearchState:
    ledger = Ledger(target_fingerprint={"target": "cpu-1t", "providers": ["CPUExecutionProvider"]})
    ledger.add(make_trial(0, latency=46.74, accuracy=0.668))
    ledger.add(
        make_trial(
            1,
            lineage=(TransformRecord("quantize_dynamic_int8", {"per_channel": True}),),
            latency=627.9,
            accuracy=0.6797,
        )
    )
    return SearchState(
        ledger=ledger,
        transforms=REGISTRY,
        budget_remaining=8,
        constraints=Constraints(max_accuracy_drop_pp=1.0),
        baseline_artifact=ModelArtifact(path=Path("m.onnx")),
    )


def policy_with(*blocks: Any) -> tuple[ClaudePolicy, StubClient]:
    client = StubClient(*blocks)
    return ClaudePolicy(client=client), client


# ----- parsing -------------------------------------------------------------


def test_tool_call_becomes_a_proposal():
    policy, _ = policy_with(
        _Block(
            "tool_use",
            "propose_transform",
            {
                "transform": "quantize_static_int8",
                "params": {"calibrate_method": "entropy"},
                "base_trial": -1,
                "rationale": "Trial 1 was 13x slower; static QDQ avoids ConvInteger.",
            },
        )
    )

    proposal = policy.propose(state_with_measurements())
    assert proposal is not None
    assert proposal.transform == "quantize_static_int8"
    assert proposal.params == {"calibrate_method": "entropy"}
    assert proposal.base_index == BASELINE
    assert "13x slower" in proposal.rationale


def test_params_returned_as_a_json_string_are_recovered():
    # Models occasionally serialise an object-typed field as a string.
    policy, _ = policy_with(
        _Block(
            "tool_use",
            "propose_transform",
            {
                "transform": "quantize_dynamic_sensitive",
                "params": '{"skip_top_k": 4}',
                "base_trial": -1,
                "rationale": "spare the worst layers",
            },
        )
    )
    assert policy.propose(state_with_measurements()).params == {"skip_top_k": 4}


def test_unparseable_params_fall_back_to_defaults_rather_than_crashing():
    policy, _ = policy_with(
        _Block(
            "tool_use",
            "propose_transform",
            {
                "transform": "graph_optimize",
                "params": "not json at all",
                "base_trial": -1,
                "rationale": "x",
            },
        )
    )
    assert policy.propose(state_with_measurements()).params == {}


def test_missing_params_field_is_treated_as_defaults():
    policy, _ = policy_with(
        _Block(
            "tool_use", "propose_transform",
            {"transform": "graph_optimize", "base_trial": -1, "rationale": "x"},
        )
    )
    assert policy.propose(state_with_measurements()).params == {}


def test_base_trial_selects_a_chained_recipe():
    policy, _ = policy_with(
        _Block(
            "tool_use", "propose_transform",
            {"transform": "quantize_static_int8", "base_trial": 1, "rationale": "chain it"},
        )
    )
    assert policy.propose(state_with_measurements()).base_index == 1


def test_stop_tool_ends_the_search_with_a_reason():
    policy, _ = policy_with(
        _Block("tool_use", "stop", {"reason": "frontier is covered"})
    )
    assert policy.propose(state_with_measurements()) is None
    assert policy.stop_reason == "frontier is covered"


def test_a_response_with_no_tool_call_stops_rather_than_crashing():
    policy, _ = policy_with(_Block("text", "", {}))
    assert policy.propose(state_with_measurements()) is None
    assert "no tool call" in policy.stop_reason


def test_transcript_records_the_decision_for_auditing():
    policy, _ = policy_with(
        _Block(
            "tool_use", "propose_transform",
            {"transform": "graph_optimize", "base_trial": -1, "rationale": "baseline first"},
        )
    )
    policy.propose(state_with_measurements())

    entry = policy.transcript[0]
    assert entry["tool"] == "propose_transform"
    assert entry["usage"]["input_tokens"] == 100


# ----- what the model is shown --------------------------------------------


def test_the_model_is_forced_to_call_a_tool():
    policy, client = policy_with(
        _Block("tool_use", "stop", {"reason": "done"})
    )
    policy.propose(state_with_measurements())
    assert client.calls[0]["tool_choice"] == {"type": "any"}


def test_rendered_state_contains_the_actual_measurements():
    state = state_with_measurements()
    rendered = prompts.render_state(state.ledger, REGISTRY, 8, state.constraints.describe())

    assert "627.90ms" in rendered
    assert "0.07x" in rendered          # the slowdown must be visible as a speedup figure
    assert "cpu-1t" in rendered
    assert "8 trial(s) remaining" in rendered


def test_rendered_state_lists_every_available_transform():
    state = state_with_measurements()
    rendered = prompts.render_state(state.ledger, REGISTRY, 8, "none")
    for name in REGISTRY:
        assert name in rendered


def test_rendered_state_reports_failures_to_the_model():
    state = state_with_measurements()
    state.ledger.add(
        make_trial(2, lineage=(TransformRecord("cast_fp16", {}),), error="no fp16 kernels")
    )
    rendered = prompts.render_state(state.ledger, REGISTRY, 7, "none")
    assert "FAILED: no fp16 kernels" in rendered


def test_tool_schema_restricts_transform_to_known_names():
    tools = prompts.build_tools(REGISTRY)
    propose = next(t for t in tools if t["name"] == "propose_transform")
    assert propose["input_schema"]["properties"]["transform"]["enum"] == sorted(REGISTRY)
    assert set(propose["input_schema"]["required"]) == {"transform", "base_trial", "rationale"}


def test_empty_ledger_renders_without_error():
    rendered = prompts.render_state(Ledger(), REGISTRY, 10, "none")
    assert "nothing measured yet" in rendered


def test_missing_api_key_gives_an_actionable_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pytest.importorskip("anthropic")
    with pytest.raises(RuntimeError, match="--policy heuristic"):
        ClaudePolicy()
