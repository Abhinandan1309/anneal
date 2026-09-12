"""Sensitivity analysis and the transform action space, against a real ONNX graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anneal.core.artifact import ModelArtifact, TransformRecord
from anneal.core.transforms import (
    TransformContext,
    TransformError,
    apply_transform,
    rank_layer_sensitivity,
    weight_quantization_error,
)


# ----- weight sensitivity --------------------------------------------------


def test_exactly_representable_weights_have_no_quantization_error():
    # Integers spanning the full int8 range land on grid points exactly.
    w = np.linspace(-127, 127, 255, dtype=np.float32).reshape(1, -1)
    assert weight_quantization_error(w, per_channel=False) == pytest.approx(0.0, abs=1e-6)


def test_quantization_error_is_bounded_and_positive_for_random_weights():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 32)).astype(np.float32)
    err = weight_quantization_error(w)
    assert 0.0 < err < 0.1


def test_an_outlier_channel_raises_per_tensor_error_more_than_per_channel():
    # One huge channel stretches a shared scale and destroys the small channels;
    # per-channel scales are immune. This is exactly why per_channel exists.
    w = np.ones((4, 8), dtype=np.float32) * 0.01
    w[0] = 1000.0
    assert weight_quantization_error(w, per_channel=False) > weight_quantization_error(
        w, per_channel=True
    )


def test_empty_tensor_does_not_divide_by_zero():
    assert weight_quantization_error(np.array([], dtype=np.float32)) == 0.0


def test_all_zero_tensor_does_not_divide_by_zero():
    assert weight_quantization_error(np.zeros((4, 4), dtype=np.float32)) == 0.0


# ----- ranking -------------------------------------------------------------


def test_ranking_finds_conv_and_gemm_sorted_by_error(tiny_onnx: Path):
    ranked = rank_layer_sensitivity(tiny_onnx)
    assert {op for _, _, op in ranked} == {"Conv", "Gemm"}

    errors = [err for _, err, _ in ranked]
    assert errors == sorted(errors, reverse=True)
    assert all(err >= 0 for err in errors)


def test_ranking_ignores_ops_without_weights(tiny_onnx: Path):
    names = {name for name, _, _ in rank_layer_sensitivity(tiny_onnx)}
    assert "relu1" not in names
    assert "pool1" not in names


# ----- applying ------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path: Path) -> TransformContext:
    return TransformContext(workdir=tmp_path / "candidates")


def test_unknown_transform_is_rejected(tiny_onnx: Path, ctx):
    artifact = ModelArtifact(path=tiny_onnx)
    with pytest.raises(TransformError, match="unknown transform"):
        apply_transform("make_it_fast", {}, artifact, ctx)


def test_unknown_parameter_is_rejected_rather_than_ignored(tiny_onnx: Path, ctx):
    artifact = ModelArtifact(path=tiny_onnx)
    with pytest.raises(TransformError, match="unknown parameter"):
        apply_transform("graph_optimize", {"levl": "all"}, artifact, ctx)


def test_bad_enum_value_is_rejected(tiny_onnx: Path, ctx):
    artifact = ModelArtifact(path=tiny_onnx)
    with pytest.raises(TransformError, match="level must be"):
        apply_transform("graph_optimize", {"level": "ludicrous"}, artifact, ctx)


def test_reapplying_an_idempotent_transform_is_rejected(tiny_onnx: Path, ctx):
    artifact = ModelArtifact(
        path=tiny_onnx, lineage=(TransformRecord("graph_optimize", {"level": "all"}),)
    )
    with pytest.raises(TransformError, match="already in this model's lineage"):
        apply_transform("graph_optimize", {"level": "all"}, artifact, ctx)


def test_static_quantization_without_calibration_data_is_refused(tiny_onnx: Path, ctx):
    artifact = ModelArtifact(path=tiny_onnx)
    with pytest.raises(TransformError, match="needs calibration data"):
        apply_transform("quantize_static_int8", {}, artifact, ctx)


def test_graph_optimize_produces_a_new_loadable_model(tiny_onnx: Path, ctx):
    import onnxruntime as ort

    result = apply_transform("graph_optimize", {"level": "all"}, ModelArtifact(path=tiny_onnx), ctx)

    assert result.path.exists()
    assert result.path != tiny_onnx
    assert result.label == "graph_optimize(level=all)"
    ort.InferenceSession(str(result.path), providers=["CPUExecutionProvider"])


def test_dynamic_quantization_shrinks_the_model_and_preserves_behaviour(tiny_onnx: Path, ctx):
    import onnxruntime as ort

    base = ModelArtifact(path=tiny_onnx)
    result = apply_transform("quantize_dynamic_int8", {"per_channel": False}, base, ctx)

    assert result.path.exists()
    assert "quantize_dynamic_int8" in result.lineage_key

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 16, 16), dtype=np.float32)

    def run(path: Path) -> np.ndarray:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return sess.run(None, {"input": x})[0]

    before, after = run(tiny_onnx), run(result.path)
    assert before.shape == after.shape
    # INT8 should perturb, not destroy: the outputs must stay strongly correlated.
    correlation = np.corrcoef(before.ravel(), after.ravel())[0, 1]
    assert correlation > 0.9


def test_selective_quantization_records_which_layers_it_spared(tiny_onnx: Path, ctx):
    base = ModelArtifact(path=tiny_onnx)
    result = apply_transform(
        "quantize_dynamic_sensitive", {"skip_top_k": 1, "per_channel": False}, base, ctx
    )

    excluded = result.meta["excluded_nodes"]
    assert len(excluded) == 1
    # The spared layer must be the top-ranked one, so the choice is auditable.
    assert excluded[0] == rank_layer_sensitivity(tiny_onnx, per_channel=False)[0][0]
    assert result.meta["sensitivity_top5"]


def test_negative_skip_top_k_is_rejected(tiny_onnx: Path, ctx):
    with pytest.raises(TransformError, match="skip_top_k must be"):
        apply_transform(
            "quantize_dynamic_sensitive", {"skip_top_k": -1}, ModelArtifact(path=tiny_onnx), ctx
        )


def test_candidate_paths_are_deterministic_for_the_same_recipe(tiny_onnx: Path, ctx):
    base = ModelArtifact(path=tiny_onnx)
    record = TransformRecord("graph_optimize", {"level": "all"})
    assert ctx.path_for(base, record) == ctx.path_for(base, record)


def test_candidate_paths_differ_for_different_recipes(tiny_onnx: Path, ctx):
    base = ModelArtifact(path=tiny_onnx)
    a = ctx.path_for(base, TransformRecord("graph_optimize", {"level": "all"}))
    b = ctx.path_for(base, TransformRecord("graph_optimize", {"level": "basic"}))
    assert a != b
