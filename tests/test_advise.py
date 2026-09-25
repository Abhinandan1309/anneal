"""The recipe advisor: reading the family from the graph, the rules, and measuring them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from test_equalize import C_IN, _model

from anneal.core.advise import advise, profile, verification_mode, verify
from anneal.core.dataset import SyntheticEvalSet


def _transformer(path: Path) -> Path:
    """LayerNorm -> MatMul -> GELU(Erf) -> MatMul, the shape of a ViT MLP block."""
    rng = np.random.default_rng(0)
    d, h = 8, 16
    inits = [
        numpy_helper.from_array(np.ones(d, np.float32), "g"),
        numpy_helper.from_array(np.zeros(d, np.float32), "b"),
        numpy_helper.from_array(rng.standard_normal((d, h)).astype(np.float32), "w1"),
        numpy_helper.from_array(rng.standard_normal((h, d)).astype(np.float32), "w2"),
        numpy_helper.from_array(np.array(0.7071, np.float32), "k"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(0.5, np.float32), "half"),
    ]
    nodes = [
        helper.make_node("LayerNormalization", ["x", "g", "b"], ["ln"], name="ln", axis=-1),
        helper.make_node("MatMul", ["ln", "w1"], ["h"], name="fc1"),
        helper.make_node("Mul", ["h", "k"], ["hk"], name="gelu_scale"),
        helper.make_node("Erf", ["hk"], ["e"], name="erf"),
        helper.make_node("Add", ["e", "one"], ["e1"], name="erf1"),
        helper.make_node("Mul", ["h", "e1"], ["he"], name="gelu_mul"),
        helper.make_node("Mul", ["he", "half"], ["gelu"], name="gelu_half"),
        helper.make_node("MatMul", ["gelu", "w2"], ["y"], name="fc2"),
    ]
    g = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 4, d])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 4, d])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return path


def test_the_family_is_read_from_the_graph(tmp_path: Path):
    assert profile(_model(tmp_path / "s.onnx", "silu")).family == "gated-depthwise"
    assert profile(_model(tmp_path / "h.onnx", "hardswish_fused")).family == "gated-depthwise"
    assert profile(_model(tmp_path / "r.onnx", "relu")).family == "cnn"
    t = profile(_transformer(tmp_path / "t.onnx"))
    assert t.family == "transformer" and t.layernorms == 1 and t.matmuls == 2


def test_transformers_get_compute_only_quantization(tmp_path: Path):
    a = advise(_transformer(tmp_path / "t.onnx"), "arm-dotprod")
    assert a.recommended.params["quantize_ops"] == "compute"
    assert any("quantize_ops" not in c.params for c in a.alternatives)  # the default, as control


@pytest.mark.parametrize("path,expect_rr", [("x86-avx2-16bit", True), ("x86-vnni", False), ("arm-dotprod", False)])
def test_silu_nets_get_reduce_range_only_where_pair_sums_saturate(tmp_path: Path, path, expect_rr):
    a = advise(_model(tmp_path / "s.onnx", "silu"), path)
    assert a.recommended.params["equalize"] is True
    assert bool(a.recommended.params.get("reduce_range")) is expect_rr
    assert a.confidence == "high"


def test_hardswish_nets_keep_full_range_weights_even_on_saturating_x86(tmp_path: Path):
    # The lab measured reduce_range 2pp worse for MobileNetV3 on non-VNNI x86.
    a = advise(_model(tmp_path / "h.onnx", "hardswish"), "x86-avx2-16bit")
    assert not a.recommended.params.get("reduce_range")
    assert any(c.params.get("reduce_range") for c in a.alternatives)


def test_relu_cnns_get_percentile_and_float_stem(tmp_path: Path):
    a = advise(_model(tmp_path / "r.onnx", "relu"), "x86-avx2-16bit")
    assert a.recommended.params["calibrate_method"] == "percentile_asym"
    assert a.recommended.params["float_stem"] is True
    assert a.recommended.params["reduce_range"] is True
    assert "equalize" not in a.recommended.params


def test_an_unknown_path_is_flagged(tmp_path: Path):
    a = advise(_model(tmp_path / "r.onnx", "relu"), "unknown")
    assert any("unknown" in c for c in a.caveats)
    with pytest.raises(ValueError):
        advise(_model(tmp_path / "r2.onnx", "relu"), "quantum")


def test_verification_uses_real_kernels_only_when_they_match_the_target():
    assert verification_mode("x86-vnni", "x86-vnni")[0] == "fused"
    assert verification_mode("arm-dotprod", "x86-avx2-16bit")[0] == "emulated"
    mode, notes = verification_mode("x86-avx2-16bit", "arm-dotprod")
    assert mode == "emulated" and "cannot show saturation" in notes[0]


def _classifier(path: Path) -> Path:
    """The SiLU test block with a pooling head, so it produces (N, C) logits."""
    m = onnx.load(str(_model(path, "silu")))
    m.graph.node.extend([
        helper.make_node("GlobalAveragePool", ["z"], ["p"], name="gap"),
        helper.make_node("Flatten", ["p"], ["logits"], name="flat"),
    ])
    del m.graph.output[:]
    m.graph.output.extend([helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 6])])
    onnx.save(m, str(path))
    return path


def test_verify_scores_every_candidate_against_fp32(tmp_path: Path):
    model = _classifier(tmp_path / "s.onnx")
    data = SyntheticEvalSet(shape=(C_IN, 8, 8), n=32, batch_size=8, n_classes=4)
    a = advise(model, "arm-dotprod")
    result = verify(a, model, data, data, tmp_path / "w", local_path="x86-avx2-16bit")
    assert result.mode == "emulated" and result.n == 32
    labels = {r.candidate.label for r in result.rows}
    assert a.recommended.label in labels and "onnxruntime default" in labels
    assert result.best in labels
    for r in result.rows:
        assert r.error is None and 0.0 <= r.agreement <= 1.0
    json.dumps(result.to_dict())


def test_advise_command(tmp_path: Path):
    from anneal.cli import main

    out = tmp_path / "advice.json"
    code = main(["advise", str(_model(tmp_path / "s.onnx", "silu")), "--int8-path", "arm-dotprod",
                 "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["advice"]["profile"]["family"] == "gated-depthwise"
    assert main(["advise", str(tmp_path / "missing.onnx")]) == 2


def test_the_search_tries_the_advised_recipe_first():
    from anneal.agent.policy import HeuristicPolicy
    from test_policy import fresh_ledger, state_for

    state = state_for(fresh_ledger(0.90))
    state.advised_recipe = {"per_channel": True, "equalize": True}
    state.advised_rationale = "because"
    first = HeuristicPolicy()._probe(state)[0]
    assert first.transform == "quantize_static_int8" and first.params["equalize"] is True
    assert "Advised recipe" in first.rationale


def test_advice_is_contradicted_only_by_a_significant_difference():
    from anneal.core.advise import Verification

    v = Verification("emulated", 1000, 0.7, None, [], "other", [], best_vs_recommended_p=0.6)
    assert not v.advice_contradicted
    v.best_vs_recommended_p = 0.01
    assert v.advice_contradicted
    assert not Verification("emulated", 1000, 0.7, None, [], "rec", [], best_vs_recommended_p=1.0).advice_contradicted
