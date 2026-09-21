"""Paired statistics and the audit of an optimised model against its original."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anneal.core.artifact import ModelArtifact
from anneal.core.audit import audit, mcnemar_exact, paired_delta_ci
from anneal.core.dataset import SyntheticEvalSet
from anneal.core.targets import get_target
from anneal.core.transforms import TransformContext, apply_transform


# ----- McNemar -------------------------------------------------------------


def test_no_discordant_images_means_no_evidence_of_a_difference():
    assert mcnemar_exact(0, 0) == 1.0


def test_balanced_discordance_is_not_significant():
    assert mcnemar_exact(5, 5) == 1.0


def test_one_sided_discordance_matches_the_hand_computed_value():
    # 10 regressions, 0 fixes: p = 2 * (1/2)^10
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


def test_moderate_imbalance_matches_the_hand_computed_value():
    # n=20, k=5: sum C(20, 0..5) = 21700; p = 2 * 21700 / 2^20
    assert mcnemar_exact(15, 5) == pytest.approx(2 * 21700 / 2**20)


def test_mcnemar_is_symmetric():
    assert mcnemar_exact(15, 5) == mcnemar_exact(5, 15)


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)


# ----- paired CI -----------------------------------------------------------


def test_equal_regressions_and_fixes_give_a_zero_delta_with_a_symmetric_interval():
    delta, lo, hi = paired_delta_ci(10, 10, 1000)
    assert delta == 0.0
    assert lo == pytest.approx(-hi)
    assert lo < 0 < hi


def test_no_discordant_images_gives_a_zero_width_interval():
    assert paired_delta_ci(0, 0, 500) == (0.0, 0.0, 0.0)


def test_net_regressions_give_a_negative_delta():
    delta, lo, hi = paired_delta_ci(30, 10, 1000)
    assert delta == pytest.approx(-2.0)
    assert lo < delta < hi


def test_interval_narrows_with_more_images():
    _, lo_small, hi_small = paired_delta_ci(3, 1, 100)
    _, lo_big, hi_big = paired_delta_ci(30, 10, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_zero_images_is_rejected():
    with pytest.raises(ValueError):
        paired_delta_ci(0, 0, 0)


# ----- end to end ----------------------------------------------------------


@pytest.fixture
def evalset() -> SyntheticEvalSet:
    return SyntheticEvalSet(shape=(3, 16, 16), n=32, batch_size=8, n_classes=4)


@pytest.fixture
def quantized(tiny_onnx: Path, tmp_path: Path) -> ModelArtifact:
    return apply_transform(
        "quantize_dynamic_int8",
        {"per_channel": False, "weight_type": "uint8"},
        ModelArtifact(path=tiny_onnx),
        TransformContext(workdir=tmp_path / "q"),
    )


def test_auditing_a_model_against_itself_finds_nothing(tiny_onnx: Path, evalset):
    model = ModelArtifact(path=tiny_onnx)
    result = audit(model, model, get_target("cpu-1t"), evalset, warmup=1, runs=3)

    assert result.n == 32
    assert result.changed == result.regressions == result.fixes == 0
    assert result.p_value == 1.0
    assert result.delta_pp == 0.0


def test_audit_counts_are_internally_consistent(tiny_onnx: Path, quantized, evalset):
    result = audit(
        ModelArtifact(path=tiny_onnx), quantized, get_target("cpu-1t"), evalset, warmup=1, runs=3
    )
    # An image can only regress or be fixed if its predicted class changed.
    assert result.regressions + result.fixes <= result.changed <= result.n
    assert result.candidate_acc - result.original_acc == pytest.approx(
        (result.fixes - result.regressions) / result.n
    )
    assert sum(c.n for c in result.classes) == result.n


def test_audit_serialises_and_renders(tiny_onnx: Path, quantized, evalset):
    result = audit(
        ModelArtifact(path=tiny_onnx), quantized, get_target("cpu-1t"), evalset, warmup=1, runs=3
    )
    json.dumps(result.to_dict())
    md = result.markdown()
    for section in ("## Verdict", "## Measurements", "## Paired accuracy test", "## Per class"):
        assert section in md
    assert result.verdict()


def test_audit_command_writes_its_report(tiny_onnx: Path, quantized, tmp_path: Path):
    from anneal.cli import main

    out = tmp_path / "audit"
    code = main(
        [
            "audit", str(tiny_onnx), str(quantized.path),
            "--target", "cpu-1t", "--eval", "synthetic", "--eval-limit", "16",
            "--eval-batch", "8", "--warmup", "1", "--runs", "3", "--out", str(out),
        ]
    )
    # 3 means the audit found a problem (e.g. the candidate is slower) — a valid outcome.
    assert code in (0, 3)
    assert (out / "audit.md").exists()
    assert json.loads((out / "audit.json").read_text(encoding="utf-8"))["n"] == 16


def test_audit_command_rejects_a_missing_model(tmp_path: Path):
    from anneal.cli import main

    assert main(["audit", str(tmp_path / "a.onnx"), str(tmp_path / "b.onnx")]) == 2


def test_latency_is_flagged_untrustworthy_when_the_machine_is_unstable(tiny_onnx: Path, evalset):
    model = ModelArtifact(path=tiny_onnx)
    result = audit(model, model, get_target("cpu-1t"), evalset, warmup=1, runs=3)
    result.environment_warnings = ["running on battery (20%)"]
    assert result.verdict()[0].startswith("LATENCY UNTRUSTWORTHY")

    result.environment_warnings = []
    result.latency_drift = 0.5
    assert "moved 50%" in result.verdict()[0]
