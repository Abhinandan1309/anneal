"""Detecting a machine that is not fit to be benchmarked."""

from __future__ import annotations

import math

from anneal.core.environment import drift, snapshot, warnings_for


def test_snapshot_always_returns_every_key():
    facts = snapshot()
    for key in ("on_ac", "battery_percent", "battery_saver", "max_mhz", "limit_mhz"):
        assert key in facts


def test_a_healthy_machine_raises_no_warnings():
    assert warnings_for(
        {"on_ac": True, "has_battery": True, "battery_saver": False,
         "max_mhz": 2900, "limit_mhz": 2900}
    ) == []


def test_a_desktop_without_a_battery_is_not_flagged():
    assert warnings_for({"on_ac": None, "has_battery": False}) == []


def test_battery_power_is_flagged_with_the_charge_level():
    warnings = warnings_for({"on_ac": False, "has_battery": True, "battery_percent": 21})
    assert any("battery (21%)" in w for w in warnings)


def test_battery_saver_is_flagged():
    assert any("battery saver" in w for w in warnings_for({"battery_saver": True}))


def test_a_clock_ceiling_below_rated_speed_is_flagged():
    warnings = warnings_for({"max_mhz": 2900, "limit_mhz": 1700})
    assert any("1700 MHz" in w for w in warnings)


def test_unknown_facts_produce_no_false_alarms():
    assert warnings_for({}) == []


def test_drift_is_relative_and_symmetric():
    assert drift(100.0, 110.0) == 0.1
    assert drift(100.0, 90.0) == 0.1
    assert math.isnan(drift(0.0, 5.0))
