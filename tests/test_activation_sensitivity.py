"""Ranking activation tensors by 8-bit damage, and int16_top_k in static quantization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from anneal.core.activation_sensitivity import candidate_tensors, rank_activation_tensors
from anneal.core.transforms import TransformError
from test_equalize import C, C_IN, _batches, _chain, _run, _static


def _sensitive(path: Path) -> Path:
    """input -> conv0 -> h -> conv1 -> s -> conv2 -> out, where only s is channel-imbalanced.

    One channel of s spans ~20000x the others, and conv2 reads mostly the small channels, so
    at 8 bits with one shared scale they round to a level or two and the output's argmax flips.
    """
    rng = np.random.default_rng(0)
    w0 = rng.standard_normal((C, C_IN, 1, 1)).astype(np.float32)
    gains = np.array([200.0] + [0.01] * (C - 1), dtype=np.float32)
    w1 = rng.standard_normal((C, C, 1, 1)).astype(np.float32) * gains.reshape(C, 1, 1, 1)
    w2 = rng.standard_normal((C_IN, C, 1, 1)).astype(np.float32) / gains.reshape(1, C, 1, 1)
    w2[:, 0] *= 1e-3  # the big channel hardly matters downstream
    nodes = [
        helper.make_node("Conv", ["input", "w0"], ["h"], name="conv0"),
        helper.make_node("Conv", ["h", "w1"], ["s"], name="conv1"),
        helper.make_node("Conv", ["s", "w2"], ["out"], name="conv2"),
    ]
    graph = helper.make_graph(
        nodes, "sensitive",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", C_IN, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", C_IN, 8, 8])],
        [numpy_helper.from_array(w, n) for w, n in ((w0, "w0"), (w1, "w1"), (w2, "w2"))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _uint16_quantizers(path: Path) -> list[str]:
    m = onnx.load(str(path))
    types = {i.name: i.data_type for i in m.graph.initializer}
    return [n.input[0] for n in m.graph.node if n.op_type == "QuantizeLinear" and len(n.input) > 2
            and types.get(n.input[2]) == TensorProto.UINT16]


def test_ranking_covers_every_conv_input_and_is_sorted(tmp_path: Path):
    src = _chain(tmp_path / "m.onnx")
    expected = ["input", "y0", "z0", "y1", "z1", "y2"]
    assert candidate_tensors(onnx.load(str(src))) == expected
    ranking = rank_activation_tensors(src, _batches(2), _batches(2, seed=5))
    assert sorted(r.tensor for r in ranking) == sorted(expected)
    damages = [r.damage for r in ranking]
    assert damages == sorted(damages, reverse=True)
    assert all(0.0 <= d <= 1.0 for d in damages)
    # The model file is not modified by the probing.
    assert candidate_tensors(onnx.load(str(src))) == expected


def test_a_channel_imbalanced_tensor_ranks_first(tmp_path: Path):
    src = _sensitive(tmp_path / "s.onnx")
    ranking = rank_activation_tensors(src, _batches(2), _batches(2, seed=5))
    assert [r.tensor for r in ranking][0] == "s"
    damage = dict(ranking)
    assert damage["s"] > 0.2
    assert damage["s"] > 5 * max(damage["input"], damage["h"], 0.01)


def test_explicit_tensors_are_ranked_and_unknown_ones_rejected(tmp_path: Path):
    src = _sensitive(tmp_path / "s.onnx")
    ranking = rank_activation_tensors(src, _batches(1), _batches(1, seed=5), tensors=["h", "s"])
    assert [r.tensor for r in ranking] == ["s", "h"]
    with pytest.raises(ValueError):
        rank_activation_tensors(src, _batches(1), _batches(1, seed=5), tensors=["nope"])
    with pytest.raises(ValueError):
        rank_activation_tensors(src, [], _batches(1, seed=5))


@pytest.fixture
def calib():
    from anneal.core.dataset import SyntheticEvalSet

    return SyntheticEvalSet(shape=(C_IN, 8, 8), n=32, batch_size=8, n_classes=4)


@pytest.mark.parametrize("k", [1, 2])
def test_int16_top_k_quantizes_exactly_k_tensors_to_16_bits(tmp_path: Path, calib, k: int):
    src = _chain(tmp_path / "m.onnx")
    art = _static(tmp_path, calib, src, f"k{k}", equalize=True, int16_top_k=k)
    assert art.lineage[-1].params["int16_top_k"] == k
    chosen = art.meta["int16_top_k_tensors"]
    assert len(chosen) == k
    assert chosen == [r["tensor"] for r in art.meta["activation_sensitivity"][:k]]
    assert set(art.meta["int16_top_k_damage"]) == set(chosen)
    assert art.meta["int16_probe_images"] == 32  # the whole (small) calibration set
    assert sorted(_uint16_quantizers(art.path)) == sorted(chosen)
    assert np.isfinite(_run(Path(art.path), _batches(1, seed=2)[0])).all()


def test_int16_top_k_picks_the_sensitive_tensor(tmp_path: Path, calib):
    src = _sensitive(tmp_path / "s.onnx")
    art = _static(tmp_path, calib, src, "k1", int16_top_k=1)
    assert art.meta["int16_top_k_tensors"] == ["s"]
    assert _uint16_quantizers(art.path) == ["s"]


def test_int16_top_k_zero_is_the_same_as_off(tmp_path: Path, calib):
    src = _chain(tmp_path / "m.onnx")
    x = _batches(1, seed=9)[0]
    off = _static(tmp_path, calib, src, "off", equalize=True)
    k0 = _static(tmp_path, calib, src, "k0", equalize=True, int16_top_k=0)
    assert "int16_top_k" not in off.lineage[-1].params
    assert "int16_top_k_tensors" not in k0.meta and "int16_top_k_tensors" not in off.meta
    assert _uint16_quantizers(k0.path) == []
    assert np.array_equal(_run(Path(k0.path), x), _run(Path(off.path), x))


@pytest.mark.parametrize(
    "params",
    [{"int16_top_k": -1}, {"int16_top_k": 1.5}, {"int16_top_k": True}, {"int16_top_k": "2"},
     {"int16_top_k": 1, "int16_tensors": ["y0"]}],
)
def test_invalid_int16_top_k_settings_are_rejected(tmp_path: Path, calib, params):
    with pytest.raises(TransformError):
        _static(tmp_path, calib, _chain(tmp_path / "m.onnx"), "bad", **params)
