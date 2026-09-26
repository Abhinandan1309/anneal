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


@pytest.mark.parametrize("weight_type", ["uint8", "int8"])
def test_dynamic_quantization_shrinks_the_model_and_preserves_behaviour(
    tiny_onnx: Path, ctx, weight_type: str
):
    import onnxruntime as ort

    base = ModelArtifact(path=tiny_onnx)
    result = apply_transform(
        "quantize_dynamic_int8", {"per_channel": False, "weight_type": weight_type}, base, ctx
    )

    assert result.path.exists()
    assert "quantize_dynamic_int8" in result.lineage_key

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 16, 16), dtype=np.float32)

    def run(path: Path) -> np.ndarray:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return sess.run(None, {"input": x})[0]

    try:
        after = run(result.path)
    except Exception as exc:
        # onnxruntime <= 1.23 (the last release for Python 3.10) ships no CPU ConvInteger
        # kernel for *signed* INT8 weights, so this graph does not load at all there. That
        # is a property of the runtime, not a bug in the transform — in a real run it
        # becomes a recorded failed trial. uint8 weights must load everywhere.
        if weight_type == "int8" and "ConvInteger" in str(exc):
            pytest.skip(f"onnxruntime {ort.__version__} has no signed-INT8 ConvInteger kernel")
        raise
    before = run(tiny_onnx)
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


# ----- measured ranking ----------------------------------------------------


def _sens(node, proxy, changed, error=None):
    from anneal.core.sensitivity import LayerSensitivity

    return LayerSensitivity(
        node=node, op_type="Conv", proxy_error=proxy, changed_fraction=changed, error=error
    )


def test_measured_order_puts_the_most_damaging_layer_first():
    from anneal.core.transforms import order_by_measured_damage

    order = order_by_measured_damage(
        [_sens("a", 0.02, 0.004), _sens("stem", 0.001, 0.035), _sens("b", 0.01, 0.016)]
    )
    # The stem has the *lowest* proxy error and the *highest* measured damage — the exact
    # case on ResNet-18 that the proxy got backwards.
    assert order == ["stem", "b", "a"]


def test_measured_ties_are_broken_by_the_proxy():
    from anneal.core.transforms import order_by_measured_damage

    order = order_by_measured_damage([_sens("low", 0.001, 0.016), _sens("high", 0.02, 0.016)])
    assert order == ["high", "low"]


def test_layers_that_failed_to_measure_go_last():
    from anneal.core.transforms import order_by_measured_damage

    order = order_by_measured_damage([_sens("broken", 0.9, None, "boom"), _sens("ok", 0.1, 0.01)])
    assert order == ["ok", "broken"]


def test_saved_sweep_round_trips_into_a_ranking(tmp_path: Path):
    import json

    from anneal.core.transforms import load_measured_ranking

    path = tmp_path / "sensitivity.json"
    path.write_text(
        json.dumps(
            {
                "layers": [
                    {"node": "a", "op_type": "Conv", "proxy_error": 0.02, "changed_fraction": 0.004},
                    {"node": "stem", "op_type": "Conv", "proxy_error": 0.001, "changed_fraction": 0.035},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_measured_ranking(path) == ["stem", "a"]


def test_empty_saved_sweep_is_rejected(tmp_path: Path):
    from anneal.core.transforms import load_measured_ranking

    path = tmp_path / "sensitivity.json"
    path.write_text('{"layers": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="no layer measurements"):
        load_measured_ranking(path)


def test_seeded_ranking_overrides_the_proxy(tiny_onnx: Path, ctx):
    proxy_order = [n for n, _, _ in rank_layer_sensitivity(tiny_onnx, per_channel=False)]
    seeded = list(reversed(proxy_order))
    ctx.extra["measured_ranking"] = seeded

    result = apply_transform(
        "quantize_dynamic_sensitive",
        {"skip_top_k": 1, "per_channel": False},
        ModelArtifact(path=tiny_onnx),
        ctx,
    )
    assert result.meta["excluded_nodes"] == [seeded[0]]
    assert result.meta["excluded_nodes"] != [proxy_order[0]]
    assert result.lineage[-1].params["ranking"] == "measured"


def test_a_seeded_ranking_for_a_different_model_is_refused(tiny_onnx: Path, ctx):
    ctx.extra["measured_ranking"] = ["/some/other/model/Conv"]
    with pytest.raises(TransformError, match="different model"):
        apply_transform(
            "quantize_dynamic_sensitive", {"skip_top_k": 1}, ModelArtifact(path=tiny_onnx), ctx
        )


def test_without_data_the_ranking_falls_back_to_the_proxy_and_says_so(tiny_onnx: Path, ctx):
    result = apply_transform(
        "quantize_dynamic_sensitive", {"skip_top_k": 1}, ModelArtifact(path=tiny_onnx), ctx
    )
    # Recorded resolved, so a proxy-ranked candidate never masquerades as a measured one.
    assert result.lineage[-1].params["ranking"] == "proxy"
    assert result.meta["sensitivity_ranking"] == "proxy"


def test_asking_for_measured_ranking_without_data_is_an_error(tiny_onnx: Path, ctx):
    with pytest.raises(TransformError, match="needs an eval set"):
        apply_transform(
            "quantize_dynamic_sensitive",
            {"skip_top_k": 1, "ranking": "measured"},
            ModelArtifact(path=tiny_onnx),
            ctx,
        )


def test_unknown_ranking_is_rejected(tiny_onnx: Path, ctx):
    with pytest.raises(TransformError, match="ranking must be"):
        apply_transform(
            "quantize_dynamic_sensitive",
            {"ranking": "vibes"},
            ModelArtifact(path=tiny_onnx),
            ctx,
        )


def test_measured_sweep_runs_once_and_is_cached(tiny_onnx: Path, tmp_path: Path):
    from anneal.core.dataset import SyntheticEvalSet

    ctx = TransformContext(
        workdir=tmp_path / "candidates",
        evalset=SyntheticEvalSet(shape=(3, 16, 16), n=16, batch_size=8),
    )
    base = ModelArtifact(path=tiny_onnx)

    first = apply_transform(
        "quantize_dynamic_sensitive", {"skip_top_k": 1, "per_channel": False}, base, ctx
    )
    assert first.lineage[-1].params["ranking"] == "measured"
    cache = ctx.extra["_measured_cache"]
    assert len(cache) == 1
    (order,) = cache.values()
    assert set(order) == {n for n, _, _ in rank_layer_sensitivity(tiny_onnx, per_channel=False)}

    # A second recipe on the same graph must reuse the sweep, not pay for it again.
    apply_transform("quantize_dynamic_sensitive", {"skip_top_k": 0, "per_channel": False}, base, ctx)
    assert len(ctx.extra["_measured_cache"]) == 1


# ----- calibration hygiene -------------------------------------------------


def test_static_quantization_prefers_a_separate_calibration_set(tiny_onnx: Path, tmp_path: Path):
    from anneal.core.dataset import SyntheticEvalSet

    ctx = TransformContext(
        workdir=tmp_path / "c",
        evalset=SyntheticEvalSet(shape=(3, 16, 16), n=16, batch_size=8, seed=0),
        calibset=SyntheticEvalSet(shape=(3, 16, 16), n=16, batch_size=8, seed=1),
        calib_samples=16,
    )
    result = apply_transform("quantize_static_int8", {}, ModelArtifact(path=tiny_onnx), ctx)
    assert "overlap" not in result.meta["calibration_source"]


def test_calibrating_on_the_eval_set_is_flagged_on_the_artifact(tiny_onnx: Path, tmp_path: Path):
    # Allowed as a fallback, but the artifact must say its accuracy is optimistic.
    from anneal.core.dataset import SyntheticEvalSet

    ctx = TransformContext(
        workdir=tmp_path / "c",
        evalset=SyntheticEvalSet(shape=(3, 16, 16), n=16, batch_size=8),
        calib_samples=16,
    )
    result = apply_transform("quantize_static_int8", {}, ModelArtifact(path=tiny_onnx), ctx)
    assert "overlaps evaluation images" in result.meta["calibration_source"]


def test_synthetic_calibration_data_differs_from_synthetic_eval_data(tmp_path: Path):
    import numpy as np

    from anneal.core.dataset import load_calibset, load_evalset

    ev = load_evalset("synthetic", cache_dir=tmp_path, batch_size=4, limit=4, sample_shape=(3, 8, 8))
    cal = load_calibset("synthetic", cache_dir=tmp_path, batch_size=4, limit=4, sample_shape=(3, 8, 8))
    assert not np.array_equal(next(iter(ev.batches()))[0], next(iter(cal.batches()))[0])


def test_a_directory_without_a_train_split_has_no_calibration_set(tmp_path: Path):
    from anneal.core.dataset import load_calibset

    (tmp_path / "val").mkdir()
    assert load_calibset(str(tmp_path), cache_dir=tmp_path) is None


def test_the_saturation_guard_records_what_it_excluded(tiny_onnx: Path, tmp_path: Path):
    from anneal.core.dataset import SyntheticEvalSet

    ctx = TransformContext(
        workdir=tmp_path / "g",
        calibset=SyntheticEvalSet(shape=(3, 16, 16), n=16, batch_size=8, seed=1),
        calib_samples=16,
    )
    result = apply_transform(
        "quantize_static_int8",
        {"per_channel": True, "guard_saturation": True, "saturation_tolerance": 0.0},
        ModelArtifact(path=tiny_onnx),
        ctx,
    )
    assert "saturation_excluded" in result.meta
    assert result.lineage[-1].params["guard_saturation"] is True
    # Everything it chose to exclude really did saturate above the tolerance.
    assert all(rate > 0.0 for rate in result.meta["saturation_rates"].values())


def test_the_guard_is_absent_from_the_recipe_unless_asked_for(tiny_onnx: Path, tmp_path: Path):
    from anneal.core.dataset import SyntheticEvalSet

    ctx = TransformContext(
        workdir=tmp_path / "n",
        calibset=SyntheticEvalSet(shape=(3, 16, 16), n=16, batch_size=8, seed=1),
        calib_samples=16,
    )
    result = apply_transform("quantize_static_int8", {}, ModelArtifact(path=tiny_onnx), ctx)
    assert "guard_saturation" not in result.lineage[-1].params


def test_entropy_calibration_gets_a_histogram_wider_than_its_quantized_bins():
    """onnxruntime's quantize_static cannot pass histogram sizes, and its 128/128 default makes
    entropy calibration return the min/max range. The patch must reach create_calibrator."""
    import importlib

    from anneal.core.transforms import ENTROPY_NUM_BINS, ENTROPY_NUM_QUANTIZED_BINS, _entropy_bins

    ort_quantize = importlib.import_module("onnxruntime.quantization.quantize")
    original = ort_quantize.create_calibrator
    seen = {}
    fake = lambda *a, **kw: seen.update(kw["extra_options"])  # noqa: E731
    ort_quantize.create_calibrator = fake
    try:
        with _entropy_bins(True):
            ort_quantize.create_calibrator(None, extra_options={"CalibPercentile": 99.0})
        assert ort_quantize.create_calibrator is fake  # restored on exit
        with _entropy_bins(False):
            assert ort_quantize.create_calibrator is fake  # untouched for other methods
    finally:
        ort_quantize.create_calibrator = original
    assert seen["num_bins"] == ENTROPY_NUM_BINS > ENTROPY_NUM_QUANTIZED_BINS == seen["num_quantized_bins"]
    assert seen["CalibPercentile"] == 99.0


def test_different_source_models_never_share_an_output_path(tmp_path):
    from anneal.core.artifact import ModelArtifact
    from anneal.core.transforms import TransformContext, TransformRecord

    ctx = TransformContext(workdir=tmp_path / "w")
    record = TransformRecord("quantize_static_int8", {"per_channel": True})
    a = ctx.path_for(ModelArtifact(path=tmp_path / "a.onnx"), record)
    b = ctx.path_for(ModelArtifact(path=tmp_path / "b.onnx"), record)
    assert a != b
    assert a == ctx.path_for(ModelArtifact(path=tmp_path / "a.onnx"), record)  # still deterministic
