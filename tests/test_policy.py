"""The heuristic policy's reasoning: does it actually react to the measurements?"""

from __future__ import annotations

from pathlib import Path

import pytest

from anneal.agent.policy import BASELINE, Constraints, HeuristicPolicy, Proposal, SearchState
from anneal.core.artifact import ModelArtifact, TransformRecord
from anneal.core.ledger import Ledger
from anneal.core.transforms import REGISTRY
from conftest import make_trial

BASE = ModelArtifact(path=Path("model.onnx"))


def state_for(ledger: Ledger, *, budget: int = 10, constraints: Constraints | None = None):
    return SearchState(
        ledger=ledger,
        transforms=REGISTRY,
        budget_remaining=budget,
        constraints=constraints or Constraints(max_accuracy_drop_pp=1.0),
        baseline_artifact=BASE,
    )


def fresh_ledger(baseline_accuracy: float = 0.90) -> Ledger:
    ledger = Ledger()
    ledger.add(make_trial(0, latency=40.0, accuracy=baseline_accuracy))
    return ledger


# ----- constraints ---------------------------------------------------------


def test_accuracy_floor_derives_from_the_measured_baseline():
    state = state_for(fresh_ledger(0.90))
    assert state.accuracy_floor() == pytest.approx(0.89)


def test_accuracy_floor_takes_the_stricter_of_the_two_constraint_forms():
    state = state_for(
        fresh_ledger(0.90), constraints=Constraints(max_accuracy_drop_pp=1.0, min_accuracy=0.95)
    )
    assert state.accuracy_floor() == pytest.approx(0.95)


def test_no_constraints_means_no_floor():
    state = state_for(fresh_ledger(0.90), constraints=Constraints(max_accuracy_drop_pp=None))
    assert state.accuracy_floor() is None


def test_a_trial_without_accuracy_cannot_meet_a_floor():
    ledger = fresh_ledger(0.90)
    unscored = make_trial(1, lineage=(TransformRecord("x", {}),), accuracy=None)
    assert not state_for(ledger).meets_floor(unscored)


# ----- opening moves -------------------------------------------------------


def test_first_proposal_is_the_lossless_one():
    policy = HeuristicPolicy()
    first = policy.propose(state_for(fresh_ledger()))
    assert first is not None
    assert first.transform == "graph_optimize"
    assert first.base_index == BASELINE


def test_probe_phase_covers_distinct_transform_families():
    policy = HeuristicPolicy()
    ledger = fresh_ledger()
    seen = []
    for _ in range(3):
        proposal = policy.propose(state_for(ledger))
        assert proposal is not None
        seen.append(proposal.transform)
        ledger.add(
            make_trial(
                ledger.next_index(),
                lineage=(TransformRecord(proposal.transform, proposal.params),),
                latency=30.0,
            )
        )
    assert len(set(seen)) == 3


def test_every_proposal_carries_a_rationale():
    policy = HeuristicPolicy()
    proposal = policy.propose(state_for(fresh_ledger()))
    assert proposal is not None and proposal.rationale.strip()


# ----- reacting to measurements -------------------------------------------


def test_accuracy_regression_triggers_selective_quantization():
    ledger = fresh_ledger(0.90)
    # A quantized candidate that blew through the 1pp budget.
    ledger.add(
        make_trial(
            1,
            lineage=(TransformRecord("quantize_dynamic_int8", {"per_channel": True}),),
            latency=20.0,
            accuracy=0.80,
        )
    )

    policy = HeuristicPolicy()
    policy._generation = 1  # skip the probe phase
    proposals = policy._react(state_for(ledger))

    selective = [p for p in proposals if p.transform == "quantize_dynamic_sensitive"]
    assert selective, "policy should try sparing sensitive layers after an accuracy break"
    assert all(p.params["skip_top_k"] > 0 for p in selective)
    assert sorted(p.params["skip_top_k"] for p in selective) == [1, 2, 4]


def test_a_slowdown_triggers_a_per_tensor_retry():
    ledger = fresh_ledger(0.90)
    # Quantization that came out far slower than FP32 — the real ORT dynamic-quant trap.
    ledger.add(
        make_trial(
            1,
            lineage=(TransformRecord("quantize_dynamic_int8", {"per_channel": True}),),
            latency=600.0,
            accuracy=0.895,
        )
    )

    policy = HeuristicPolicy()
    policy._generation = 1
    proposals = policy._react(state_for(ledger))

    retries = [
        p
        for p in proposals
        if p.transform == "quantize_dynamic_int8" and p.params.get("per_channel") is False
    ]
    assert retries, "a regression should prompt a per-tensor retry before giving up"
    assert "SLOWER" in retries[0].rationale


def test_healthy_accuracy_pushes_for_more_speed():
    ledger = fresh_ledger(0.90)
    ledger.add(
        make_trial(
            1,
            lineage=(TransformRecord("quantize_static_int8", {"per_channel": True}),),
            latency=25.0,
            accuracy=0.899,
        )
    )

    policy = HeuristicPolicy()
    policy._generation = 1
    proposals = policy._react(state_for(ledger))

    assert not any(
        p.transform == "quantize_dynamic_sensitive" and p.params["skip_top_k"] > 0
        for p in proposals
    ), "nothing broke accuracy, so there is no reason to spare layers"


# ----- not repeating itself ------------------------------------------------


def test_policy_never_reproposes_a_recipe_in_the_ledger():
    ledger = fresh_ledger()
    policy = HeuristicPolicy()

    for _ in range(12):
        proposal = policy.propose(state_for(ledger))
        if proposal is None:
            break
        record = TransformRecord(proposal.transform, proposal.params)
        base = BASE if proposal.base_index == BASELINE else ledger.trials[proposal.base_index].artifact
        key = "|".join([*(str(t) for t in base.lineage), str(record)])
        assert key not in ledger.attempted_recipes()
        ledger.add(
            make_trial(
                ledger.next_index(), lineage=base.lineage + (record,), latency=30.0, accuracy=0.895
            )
        )


def test_policy_eventually_stops():
    ledger = fresh_ledger()
    policy = HeuristicPolicy()

    for _ in range(40):
        proposal = policy.propose(state_for(ledger))
        if proposal is None:
            assert policy.stop_reason
            return
        record = TransformRecord(proposal.transform, proposal.params)
        base = BASE if proposal.base_index == BASELINE else ledger.trials[proposal.base_index].artifact
        ledger.add(
            make_trial(
                ledger.next_index(), lineage=base.lineage + (record,), latency=30.0, accuracy=0.895
            )
        )
    pytest.fail("heuristic policy never terminated")


def test_proposal_referencing_a_missing_trial_resolves_to_nothing():
    assert state_for(fresh_ledger()).artifact(99) is None


def test_proposal_referencing_a_failed_trial_resolves_to_nothing():
    ledger = fresh_ledger()
    ledger.add(make_trial(1, lineage=(TransformRecord("x", {}),), error="boom"))
    assert state_for(ledger).artifact(1) is None


def test_constraints_describe_is_human_readable():
    text = Constraints(max_accuracy_drop_pp=1.0, max_size_bytes=8_000_000).describe()
    assert "1.00pp" in text and "8.0MB" in text
    assert Constraints(max_accuracy_drop_pp=None).describe() == "none (explore the whole frontier)"


def test_preview_key_composes_lineage():
    base = ModelArtifact(
        path=Path("m.onnx"), lineage=(TransformRecord("graph_optimize", {"level": "all"}),)
    )
    key = Proposal("quantize_dynamic_int8", {"per_channel": True}).preview_key(base)
    assert key == "graph_optimize(level=all)|quantize_dynamic_int8(per_channel=True)"


def test_policy_asks_for_measured_ranking_when_it_can():
    ledger = fresh_ledger(0.90)
    ledger.add(
        make_trial(
            1,
            lineage=(TransformRecord("quantize_static_int8", {"per_channel": True}),),
            latency=20.0,
            accuracy=0.80,
        )
    )
    state = state_for(ledger)
    state.can_measure_sensitivity = True

    policy = HeuristicPolicy()
    policy._generation = 1
    selective = [p for p in policy._react(state) if p.transform == "quantize_dynamic_sensitive"]
    assert selective and all(p.params["ranking"] == "measured" for p in selective)


def test_policy_falls_back_to_the_proxy_without_data():
    ledger = fresh_ledger(0.90)
    ledger.add(
        make_trial(
            1,
            lineage=(TransformRecord("quantize_static_int8", {"per_channel": True}),),
            latency=20.0,
            accuracy=0.80,
        )
    )
    policy = HeuristicPolicy()
    policy._generation = 1
    selective = [
        p for p in policy._react(state_for(ledger)) if p.transform == "quantize_dynamic_sensitive"
    ]
    assert selective and all(p.params["ranking"] == "proxy" for p in selective)


def test_policy_never_proposes_fusing_twice_when_fusion_is_the_fastest_result():
    # MobileNetV3 regression: fusion was the fastest candidate, and "stack the winner on
    # the fused graph" proposed graph_optimize on top of graph_optimize — a wasted trial.
    ledger = fresh_ledger(0.90)
    fuse = TransformRecord("graph_optimize", {"level": "all"})
    ledger.add(make_trial(1, lineage=(fuse,), latency=38.0, accuracy=0.90))
    ledger.add(
        make_trial(
            2,
            lineage=(TransformRecord("quantize_static_int8", {"per_channel": True}),),
            latency=45.0,
            accuracy=0.60,
        )
    )

    policy = HeuristicPolicy()
    policy._generation = 1
    proposals = policy._react(state_for(ledger))

    stacked = [p for p in proposals if p.base_index == 1]
    assert all(p.transform != "graph_optimize" for p in stacked)
    # It should still try the fastest *quantization* on the fused graph.
    assert any(p.transform == "quantize_static_int8" for p in stacked)


def test_broken_static_int8_is_retried_with_reduce_range_and_per_tensor_first():
    # ResNet-18 on a Zen 2 CPU: full-range per-channel static INT8 lost 4.2pp, while the
    # same recipe with reduce_range lost nothing at the same speed. The search never
    # tried that. It must now, before the slow selective-dynamic path.
    ledger = fresh_ledger(0.90)
    params = {
        "per_channel": True, "reduce_range": False, "calibrate_method": "minmax",
        "calib_samples": 64, "activation_type": "uint8",
    }
    ledger.add(
        make_trial(1, lineage=(TransformRecord("quantize_static_int8", params),),
                   latency=20.0, accuracy=0.85)
    )

    policy = HeuristicPolicy()
    policy._generation = 1
    proposals = policy._react(state_for(ledger))

    static = [p for p in proposals if p.transform == "quantize_static_int8"]
    assert {"reduce_range": True}.items() <= static[0].params.items()
    assert any(p.params["per_channel"] is False for p in static)
    first_dynamic = next(i for i, p in enumerate(proposals) if p.transform == "quantize_dynamic_sensitive")
    assert all(proposals.index(p) < first_dynamic for p in static)


def _broken_static_ledger():
    ledger = fresh_ledger(0.90)
    params = {
        "per_channel": True, "reduce_range": False, "calibrate_method": "minmax",
        "calib_samples": 64, "activation_type": "uint8",
    }
    ledger.add(make_trial(1, lineage=(TransformRecord("quantize_static_int8", params),),
                          latency=20.0, accuracy=0.85))
    return ledger


def test_on_a_saturating_cpu_the_guard_is_tried_before_reduce_range():
    state = state_for(_broken_static_ledger())
    state.int8_path = "x86-avx2-16bit"
    policy = HeuristicPolicy()
    policy._generation = 1
    static = [p for p in policy._react(state) if p.transform == "quantize_static_int8"]
    assert static[0].params.get("guard_saturation") is True
    assert any(p.params.get("reduce_range") for p in static[1:])


def test_off_the_saturating_path_no_guard_is_proposed():
    for path in ("x86-vnni", "arm-dotprod", "unknown"):
        state = state_for(_broken_static_ledger())
        state.int8_path = path
        policy = HeuristicPolicy()
        policy._generation = 1
        assert not any(p.params.get("guard_saturation") for p in policy._react(state))


def test_a_net_with_depthwise_chains_tries_equalisation_first():
    state = state_for(_broken_static_ledger())
    state.int8_path = "x86-avx2-16bit"
    state.equalisable_sites = 16
    policy = HeuristicPolicy()
    policy._generation = 1
    static = [p for p in policy._react(state) if p.transform == "quantize_static_int8"]
    assert static[0].params.get("equalize") is True
    assert {p.params.get("float_gates") for p in static[:2]} == {False, True}
    # The CPU-specific fixes still follow.
    assert any(p.params.get("guard_saturation") for p in static[2:])


def test_equalisation_is_not_proposed_without_sites_or_twice():
    state = state_for(_broken_static_ledger())
    policy = HeuristicPolicy()
    policy._generation = 1
    assert not any(p.params.get("equalize") for p in policy._react(state))

    ledger = fresh_ledger(0.90)
    params = {"per_channel": True, "calib_samples": 64, "activation_type": "uint8",
              "equalize": True, "equalize_slack": 0.1, "float_gates": False}
    ledger.add(make_trial(1, lineage=(TransformRecord("quantize_static_int8", params),),
                          latency=20.0, accuracy=0.85))
    state = state_for(ledger)
    state.equalisable_sites = 16
    assert not any(
        p.params.get("equalize") and p.params.get("float_gates") is False
        and not p.params.get("guard_saturation") and not p.params.get("reduce_range")
        and p.params.get("per_channel")
        for p in policy._react(state)
    )
