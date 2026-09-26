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


@pytest.mark.parametrize("path", ["x86-avx2-16bit", "arm-dotprod"])
def test_relu_cnns_get_symmetric_percentile_and_float_stem_without_reduce_range(tmp_path: Path, path):
    # ImageNet ablation: 99.99 clipped too much and reduce_range cost ~0.5pp on ResNet-50.
    a = advise(_model(tmp_path / "r.onnx", "relu"), path)
    assert a.recommended.params["calibrate_method"] == "percentile"
    assert a.recommended.params["calib_percentile"] == 99.999
    assert a.recommended.params["float_stem"] is True
    assert not a.recommended.params.get("reduce_range")
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


def _with_attention(src: Path, dst: Path, softmax: bool) -> Path:
    """The gated model plus an attention-like MatMul on its output (with or without Softmax)."""
    m = onnx.load(str(src))
    out = m.graph.output[0].name
    nodes = [helper.make_node("Transpose", [out], ["zt"], perm=[0, 1, 3, 2]),
             helper.make_node("MatMul", [out, "zt"], ["att"])]
    if softmax:
        nodes.append(helper.make_node("Softmax", ["att"], ["att_s"], axis=-1))
    m.graph.node.extend(nodes)
    del m.graph.output[:]
    m.graph.output.append(helper.make_tensor_value_info("att_s" if softmax else "att", TensorProto.FLOAT, None))
    onnx.save(m, str(dst))
    return dst


def test_gated_chains_decide_the_family_before_layernorm_and_attention(tmp_path: Path):
    src = _model(tmp_path / "s.onnx", "silu")
    # softmax attention between gated blocks (MobileViT): the gated recipe applies
    assert profile(_with_attention(src, tmp_path / "soft.onnx", softmax=True)).family == "gated-depthwise"
    # linear attention, no Softmax (EfficientViT): its own recipe, attention MatMuls in float
    lin = _with_attention(src, tmp_path / "lin.onnx", softmax=False)
    assert profile(lin).family == "gated-linear-attention"
    rec = advise(lin, "arm-dotprod").recommended.params
    assert rec["quantize_ops"] == "conv" and rec["equalize"] and rec["equalize_dense"]


# ---------------------------------------------------------------------------
# Accelerator targets (TI TIDL, AMD XINT8): per-tensor weights, pow2 symmetric feature maps
# ---------------------------------------------------------------------------

EMULATION = {"per_channel": False, "activation_type": "int8", "activation_symmetric": True,
             "pow2_activation_scales": True}


def _emulates(params: dict) -> bool:
    return all(params.get(k) == v for k, v in EMULATION.items())


@pytest.mark.parametrize("target", ["tidl", "amd-xint8"])
def test_gated_nets_get_per_tensor_equalisation_with_se_sites_and_capped_cle(tmp_path: Path, target):
    a = advise(_model(tmp_path / "s.onnx", "silu"), "arm-dotprod", target=target)
    rec = a.recommended.params
    assert a.target == target and a.to_dict()["target"] == target
    assert rec["equalize"] and rec["equalize_se"] and rec["equalize_mix"] == 0.5
    assert rec["cle"] and rec["cle_max_scale"] == 4
    assert all(_emulates(c.params) for c in a.candidates())
    assert any(c.label == "8-bit control" and not c.params.get("equalize") for c in a.alternatives)
    assert any("16" in c and "tensor_bits" in c for c in a.caveats)
    # AMD's NPU runs HardSigmoid only: SiLU nets get the surrogate there, not on TIDL.
    assert rec.get("sigmoid_surrogate") == (3 if target == "amd-xint8" else None)
    if target == "amd-xint8":
        assert a.confidence == "low" and any("35.1pp" in c for c in a.caveats)
    json.dumps(a.to_dict())


def test_hardswish_nets_get_no_sigmoid_surrogate_on_amd(tmp_path: Path):
    a = advise(_model(tmp_path / "h.onnx", "hardswish"), "unknown", target="amd-xint8")
    assert "sigmoid_surrogate" not in a.recommended.params


@pytest.mark.parametrize("target", ["tidl", "amd-xint8"])
def test_relu_cnns_get_cle_capped_at_4_on_accelerators(tmp_path: Path, target):
    a = advise(_model(tmp_path / "r.onnx", "relu"), target=target)
    rec = a.recommended.params
    assert rec["cle"] is True and rec["cle_max_scale"] == 4 and "equalize" not in rec
    assert "float_stem" not in rec  # the whole net runs on the accelerator
    assert all(_emulates(c.params) for c in a.candidates())
    assert any(c.params.get("cle") and "cle_max_scale" not in c.params for c in a.alternatives)
    assert not any("x86" in c for c in a.caveats)


def test_other_families_keep_the_cpu_recipe_with_emulation_flags_and_say_so(tmp_path: Path):
    a = advise(_transformer(tmp_path / "t.onnx"), "arm-dotprod", target="tidl")
    assert a.recommended.params["quantize_ops"] == "compute"
    assert all(_emulates(c.params) for c in a.candidates())
    assert a.confidence == "low" and any("unvalidated" in c for c in a.caveats)


def test_an_unknown_target_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        advise(_model(tmp_path / "r.onnx", "relu"), "arm-dotprod", target="hexagon")


def test_a_target_recipe_builds_and_verifies_emulated(tmp_path: Path):
    model = _classifier(tmp_path / "s.onnx")
    data = SyntheticEvalSet(shape=(C_IN, 8, 8), n=16, batch_size=8, n_classes=4)
    a = advise(model, "x86-vnni", target="tidl")
    result = verify(a, model, data, data, tmp_path / "w", local_path="x86-vnni")
    assert result.mode == "emulated"  # never this CPU's kernels, even when int8_path matches
    for r in result.rows:
        assert r.error is None, r.error
    rec = next(r for r in result.rows if r.candidate.label == a.recommended.label)
    art = onnx.load(rec.path)
    scales = [numpy_helper.to_array(i) for i in art.graph.initializer if i.name.endswith("_scale")]
    assert scales  # power-of-two activation scales were applied somewhere
    assert any(np.all(np.log2(s) == np.round(np.log2(s))) for s in scales if s.size == 1)


def test_advise_command_records_the_target(tmp_path: Path):
    from anneal.cli import main

    out = tmp_path / "advice.json"
    assert main(["advise", str(_model(tmp_path / "r.onnx", "relu")), "--target", "tidl",
                 "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["target"] == "tidl" and report["advice"]["target"] == "tidl"
    assert report["advice"]["recommended"]["params"]["cle_max_scale"] == 4
