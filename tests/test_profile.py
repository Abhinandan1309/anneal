"""Parsing onnxruntime traces into operator attribution."""

from __future__ import annotations

import pytest

from anneal.core.profile import Profile, diff_profiles, parse_profile_events


def kernel_event(node: str, op: str, dur: float) -> dict:
    return {"cat": "Node", "name": f"{node}_kernel_time", "dur": dur, "args": {"op_name": op}}


def test_aggregates_time_by_operator_type():
    profile = parse_profile_events(
        [
            kernel_event("conv1", "Conv", 100.0),
            kernel_event("conv2", "Conv", 300.0),
            kernel_event("fc", "Gemm", 100.0),
        ],
        "m",
        runs=1,
    )

    assert [o.op_type for o in profile.by_op] == ["Conv", "Gemm"]
    assert profile.by_op[0].total_us == 400.0
    assert profile.by_op[0].count == 2
    assert profile.by_op[0].share == pytest.approx(0.8)
    assert profile.total_us == 500.0


def test_shares_sum_to_one():
    profile = parse_profile_events(
        [kernel_event("a", "Conv", 7.0), kernel_event("b", "Gemm", 3.0)], "m", runs=1
    )
    assert sum(o.share for o in profile.by_op) == pytest.approx(1.0)


def test_nodes_are_ranked_by_cost():
    profile = parse_profile_events(
        [
            kernel_event("slow", "Conv", 900.0),
            kernel_event("fast", "Conv", 100.0),
        ],
        "m",
        runs=1,
    )
    assert [n.name for n in profile.by_node] == ["slow", "fast"]
    assert profile.by_node[0].share == pytest.approx(0.9)


def test_repeated_runs_accumulate_into_one_node():
    profile = parse_profile_events(
        [kernel_event("conv1", "Conv", 10.0), kernel_event("conv1", "Conv", 30.0)], "m", runs=2
    )
    assert len(profile.by_node) == 1
    assert profile.by_node[0].total_us == 40.0
    assert profile.by_node[0].count == 2


def test_non_node_events_are_ignored():
    profile = parse_profile_events(
        [
            {"cat": "Session", "name": "model_loading_uri", "dur": 99999.0, "args": {}},
            kernel_event("conv1", "Conv", 10.0),
        ],
        "m",
        runs=1,
    )
    assert profile.total_us == 10.0


def test_fence_events_are_ignored():
    # onnxruntime emits <node>_fence_before/_fence_after alongside kernel times; counting
    # them would double-count the graph.
    profile = parse_profile_events(
        [
            {"cat": "Node", "name": "conv1_fence_before", "dur": 5.0, "args": {}},
            kernel_event("conv1", "Conv", 10.0),
            {"cat": "Node", "name": "conv1_fence_after", "dur": 5.0, "args": {}},
        ],
        "m",
        runs=1,
    )
    assert profile.total_us == 10.0


def test_missing_op_name_becomes_unknown_rather_than_crashing():
    profile = parse_profile_events(
        [{"cat": "Node", "name": "x_kernel_time", "dur": 1.0, "args": {}}], "m", runs=1
    )
    assert profile.by_op[0].op_type == "Unknown"


def test_empty_trace_does_not_divide_by_zero():
    profile = parse_profile_events([], "m", runs=1)
    assert profile.total_us == 0.0
    assert profile.by_op == ()
    assert profile.mean_ms_per_run == 0.0


def test_lookup_by_op_type():
    profile = parse_profile_events([kernel_event("a", "Conv", 5.0)], "m", runs=1)
    assert profile.op("Conv").total_us == 5.0
    assert profile.op("ConvInteger") is None


# ----- diffing -------------------------------------------------------------


def make(op_durations: dict[str, float], runs: int) -> Profile:
    events = [kernel_event(f"n{i}", op, dur) for i, (op, dur) in enumerate(op_durations.items())]
    return parse_profile_events(events, "m", runs=runs)


def test_diff_detects_an_operator_that_replaced_another():
    before = make({"Conv": 100.0}, runs=1)
    after = make({"ConvInteger": 1900.0}, runs=1)

    deltas = {d.op_type: d for d in diff_profiles(before, after)}
    assert deltas["ConvInteger"].appeared
    assert deltas["Conv"].vanished
    assert deltas["ConvInteger"].delta_us == pytest.approx(1900.0)


def test_diff_is_sorted_worst_regression_first():
    before = make({"Conv": 100.0, "Gemm": 50.0}, runs=1)
    after = make({"Conv": 900.0, "Gemm": 10.0}, runs=1)
    assert [d.op_type for d in diff_profiles(before, after)] == ["Conv", "Gemm"]


def test_diff_normalises_per_run_so_unequal_iteration_counts_compare_honestly():
    # Same per-run cost, different iteration counts: the delta must be zero.
    before = make({"Conv": 100.0}, runs=1)
    after = make({"Conv": 1000.0}, runs=10)
    assert diff_profiles(before, after)[0].delta_us == pytest.approx(0.0)


def test_diff_of_identical_profiles_is_all_zero():
    profile = make({"Conv": 100.0, "Gemm": 20.0}, runs=1)
    assert all(d.delta_us == 0 for d in diff_profiles(profile, profile))


def test_unchanged_operator_is_neither_new_nor_gone():
    delta = next(d for d in diff_profiles(make({"Conv": 10.0}, 1), make({"Conv": 20.0}, 1)))
    assert not delta.appeared and not delta.vanished


def test_profile_serialises_for_the_record():
    profile = make({"Conv": 100.0}, runs=4)
    d = profile.to_dict()
    assert d["by_op"][0]["op_type"] == "Conv"
    assert d["runs"] == 4
    assert d["mean_ms_per_run"] == pytest.approx(0.025)
