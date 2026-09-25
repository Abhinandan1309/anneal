"""CLI wiring, and the generated reproduction scripts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from anneal.cli import build_parser, main
from anneal.core.artifact import TransformRecord
from anneal.core.ledger import Ledger
from conftest import make_trial


@pytest.fixture
def saved_ledger(tmp_path: Path, ledger: Ledger) -> Path:
    ledger.config["constraints"] = {"max_accuracy_drop_pp": 1.0}
    return ledger.save(tmp_path / "ledger.json")


# ----- parser --------------------------------------------------------------


def test_every_subcommand_is_registered():
    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "command")
    assert set(action.choices) == {
        "run", "targets", "transforms", "sensitivity", "audit", "saturation", "imbalance", "advise",
        "profile", "compare", "validate", "export", "report",
    }


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_run_defaults_are_sane():
    args = build_parser().parse_args(["run"])
    assert args.model == "torchvision:resnet18"
    assert args.warmup > 0 and args.runs > args.warmup
    assert args.max_accuracy_drop == 1.0


def test_compare_requires_a_target():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["compare", "ledger.json"])


# ----- export --------------------------------------------------------------


def test_export_generates_valid_python(saved_ledger: Path, tmp_path: Path):
    out = tmp_path / "repro.py"
    assert main(["export", str(saved_ledger), "--trial", "2", "--out", str(out)]) == 0
    ast.parse(out.read_text(encoding="utf-8"))


def test_exported_script_applies_the_exact_recipe(saved_ledger: Path, tmp_path: Path):
    out = tmp_path / "repro.py"
    main(["export", str(saved_ledger), "--trial", "2", "--out", str(out)])
    text = out.read_text(encoding="utf-8")

    assert "'quantize_dynamic_int8'" in text
    assert "'per_channel': True" in text
    # The measured numbers travel with the recipe so a reader can tell whether their
    # rebuild matches.
    assert "p50 latency" in text
    assert "20.00 ms" in text


def test_exported_script_uses_posix_paths(saved_ledger: Path, tmp_path: Path):
    out = tmp_path / "repro.py"
    main(["export", str(saved_ledger), "--trial", "2", "--out", str(out)])
    baseline_line = next(
        line for line in out.read_text(encoding="utf-8").splitlines()
        if line.startswith("BASELINE")
    )
    assert "\\\\" not in baseline_line


def test_export_chains_multiple_transforms_in_order(tmp_path: Path):
    led = Ledger()
    led.add(make_trial(0, latency=40.0, accuracy=0.90))
    led.add(
        make_trial(
            1,
            lineage=(
                TransformRecord("graph_optimize", {"level": "all"}),
                TransformRecord("quantize_static_int8", {"per_channel": True}),
            ),
            latency=20.0,
            accuracy=0.895,
        )
    )
    path = led.save(tmp_path / "ledger.json")

    out = tmp_path / "repro.py"
    assert main(["export", str(path), "--trial", "1", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")

    assert text.index("graph_optimize") < text.index("quantize_static_int8")
    assert "# step 1" in text and "# step 2" in text
    # A static-quantization recipe must carry calibration data with it, or it cannot run.
    assert "load_calibset" in text


def test_export_omits_calibration_when_no_transform_needs_it(saved_ledger: Path, tmp_path: Path):
    out = tmp_path / "repro.py"
    main(["export", str(saved_ledger), "--trial", "1", "--out", str(out)])
    assert "load_calibset" not in out.read_text(encoding="utf-8")


def test_exporting_the_baseline_is_refused(saved_ledger: Path, tmp_path: Path):
    # There is no recipe to reproduce; saying so beats emitting an empty script.
    assert main(["export", str(saved_ledger), "--trial", "0", "--out", str(tmp_path / "x.py")]) == 1


def test_exporting_a_failed_trial_is_refused(tmp_path: Path):
    led = Ledger()
    led.add(make_trial(0, latency=40.0, accuracy=0.9))
    led.add(make_trial(1, lineage=(TransformRecord("q"),), error="boom"))
    path = led.save(tmp_path / "ledger.json")
    assert main(["export", str(path), "--trial", "1", "--out", str(tmp_path / "x.py")]) == 2


def test_exporting_an_unknown_trial_is_refused(saved_ledger: Path, tmp_path: Path):
    assert main(["export", str(saved_ledger), "--trial", "99", "--out", str(tmp_path / "x.py")]) == 2


def test_export_without_a_trial_picks_the_recommendation(saved_ledger: Path, tmp_path: Path):
    # Baseline is 90%; the 1pp budget rules out trial 2 (87%) and leaves trial 1.
    out = tmp_path / "repro.py"
    assert main(["export", str(saved_ledger), "--out", str(out)]) == 0
    assert "Reproduce trial 1" in out.read_text(encoding="utf-8")


def test_missing_ledger_is_an_error(tmp_path: Path):
    assert main(["export", str(tmp_path / "nope.json"), "--out", str(tmp_path / "x.py")]) == 2


# ----- other commands ------------------------------------------------------


def test_targets_command_runs():
    assert main(["targets"]) == 0


def test_transforms_command_runs():
    assert main(["transforms"]) == 0


def test_report_command_rewrites_from_a_ledger(saved_ledger: Path):
    assert main(["report", str(saved_ledger)]) == 0
    assert (saved_ledger.parent / "report.md").exists()
    assert (saved_ledger.parent / "frontier.svg").exists()


def test_report_on_a_missing_ledger_is_an_error(tmp_path: Path):
    assert main(["report", str(tmp_path / "nope.json")]) == 2


def test_sensitivity_on_a_missing_model_is_an_error(tmp_path: Path):
    assert main(["sensitivity", str(tmp_path / "nope.onnx")]) == 2


def test_sensitivity_ranks_a_real_graph(tiny_onnx: Path):
    assert main(["sensitivity", str(tiny_onnx), "--top", "5"]) == 0
