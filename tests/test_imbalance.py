"""Predicting starved channels from the float model, and the `anneal imbalance` command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_equalize import C, _batches, _model

from anneal.core.equalize import equalise
from anneal.core.imbalance import analyse, summarise


def test_an_imbalanced_depthwise_layer_is_flagged(tmp_path: Path):
    results = analyse(_model(tmp_path / "m.onnx", "silu"), _batches())
    dw = next(r for r in results if r.node == "conv_b")
    assert dw.depthwise
    assert dw.flagged
    assert dw.levels_min < 1.0  # some channel spans less than one quantization step


def test_equalisation_clears_the_flag(tmp_path: Path):
    src = _model(tmp_path / "m.onnx", "silu")
    before = {r.node: r for r in analyse(src, _batches())}
    equalise(src, tmp_path / "eq.onnx", _batches())
    after = {r.node: r for r in analyse(tmp_path / "eq.onnx", _batches())}
    assert after["conv_b"].sqnr_p10_db > before["conv_b"].sqnr_p10_db + 10
    assert not after["conv_b"].flagged


def test_the_prediction_tracks_the_actual_error_of_the_quantized_input(tmp_path: Path):
    """Predicted noise per output channel vs the error measured by rounding the real input."""
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    src = _model(tmp_path / "m.onnx", "silu")
    dw = next(r for r in analyse(src, _batches()) if r.node == "conv_b")

    m = onnx.load(str(src))
    m.graph.output.extend([onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)])
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    ys = np.concatenate([s.run(["y"], {"input": b})[0] for b in _batches()])
    zero = min(0.0, float(ys.min()))
    step = (max(0.0, float(ys.max())) - zero) / 255
    assert step == pytest.approx(dw.step, rel=1e-5)
    wb = numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == "wb"))

    def dwconv(y):
        out = np.zeros_like(y)
        p = np.pad(y, ((0, 0), (0, 0), (1, 1), (1, 1)))
        for i in range(3):
            for j in range(3):
                out += wb[None, :, 0, i, j, None, None] * p[:, :, i:i + y.shape[2], j:j + y.shape[3]]
        return out

    yq = np.round((ys - zero) / step) * step + zero
    err = dwconv(yq) - dwconv(ys)
    sig = dwconv(ys)
    measured = 10 * np.log10(sig.var(axis=(0, 2, 3)) / (err ** 2).mean(axis=(0, 2, 3)))
    predicted = 10 * np.log10(sig.var(axis=(0, 2, 3)) / (step ** 2 / 12 * (wb ** 2).reshape(C, -1).sum(1)))
    levels = (ys.max(axis=(0, 2, 3)) - ys.min(axis=(0, 2, 3))) / step

    # Where a channel spans several steps, rounding error is uniform noise and Δ²/12 holds.
    wide = levels >= 4
    assert wide.sum() >= 3
    assert np.abs(measured[wide] - predicted[wide]).max() < 3.0
    # Below one step it is a bias, not noise: the number is off but the flag is right.
    assert np.all(measured[levels < 1] < 10.0) and np.all(predicted[levels < 1] < 10.0)
    assert dw.sqnr_p10_db == pytest.approx(np.percentile(predicted, 10), abs=0.5)


def test_summary_counts_flags(tmp_path: Path):
    s = summarise(analyse(_model(tmp_path / "m.onnx", "silu"), _batches()))
    assert s["flagged"] >= 1 and s["flagged_depthwise"] >= 1
    assert s["worst"] == "conv_b"


def test_imbalance_command_reports_and_equalises(tmp_path: Path):
    import json

    from anneal.cli import main

    src = _model(tmp_path / "m.onnx", "silu")
    out = tmp_path / "imb.json"
    code = main(["imbalance", str(src), "--eval", "synthetic", "--calib-samples", "16",
                 "--equalize", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["layers"] == 2
    assert "equalised" in report
    assert code in (0, 3)
    assert main(["imbalance", str(tmp_path / "missing.onnx")]) == 2
