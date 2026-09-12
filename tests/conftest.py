"""Shared fixtures.

Most tests build ledgers out of synthetic measurements rather than running onnxruntime —
the arithmetic of Pareto dominance should be testable without a 45MB model. The one place
real ONNX is needed (transforms) gets a tiny two-layer graph built on the fly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anneal.core.artifact import ModelArtifact, TransformRecord
from anneal.core.ledger import Ledger, Trial
from anneal.core.measure import Measurement


def make_measurement(
    *,
    latency: float = 40.0,
    size_bytes: int = 46_000_000,
    accuracy: float | None = 0.90,
    agreement: float | None = None,
    batch_size: int = 1,
) -> Measurement:
    return Measurement(
        latency_ms_p50=latency,
        latency_ms_p90=latency * 1.1,
        latency_ms_p99=latency * 1.2,
        latency_ms_mean=latency * 1.02,
        latency_ms_std=latency * 0.03,
        n_runs=50,
        n_warmup=10,
        batch_size=batch_size,
        size_bytes=size_bytes,
        providers_used=("CPUExecutionProvider",),
        load_time_ms=12.0,
        accuracy=accuracy,
        n_eval=256 if accuracy is not None else 0,
        top1_agreement=agreement,
    )


def make_trial(
    index: int,
    *,
    lineage: tuple[TransformRecord, ...] = (),
    latency: float = 40.0,
    size_bytes: int = 46_000_000,
    accuracy: float | None = 0.90,
    error: str | None = None,
    path: str = "model.onnx",
) -> Trial:
    artifact = ModelArtifact(path=Path(path), lineage=lineage)
    if error is not None:
        return Trial(index=index, artifact=artifact, error=error)
    return Trial(
        index=index,
        artifact=artifact,
        measurement=make_measurement(
            latency=latency, size_bytes=size_bytes, accuracy=accuracy
        ),
    )


@pytest.fixture
def q8() -> TransformRecord:
    return TransformRecord("quantize_dynamic_int8", {"per_channel": True})


@pytest.fixture
def fuse() -> TransformRecord:
    return TransformRecord("graph_optimize", {"level": "all"})


@pytest.fixture
def ledger(q8, fuse) -> Ledger:
    """Baseline plus three candidates spanning the latency/accuracy trade-off."""
    led = Ledger(target_fingerprint={"target": "cpu-1t", "providers": ["CPUExecutionProvider"]})
    led.add(make_trial(0, latency=40.0, size_bytes=46_000_000, accuracy=0.90))
    led.add(make_trial(1, lineage=(fuse,), latency=36.0, size_bytes=46_000_000, accuracy=0.90))
    led.add(make_trial(2, lineage=(q8,), latency=20.0, size_bytes=11_000_000, accuracy=0.87))
    led.add(make_trial(3, lineage=(fuse, q8), latency=45.0, size_bytes=12_000_000, accuracy=0.85))
    return led


@pytest.fixture
def tiny_onnx(tmp_path: Path) -> Path:
    """A minimal but genuinely quantizable graph: Conv -> Relu -> GlobalAveragePool -> Gemm."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    conv_w = rng.standard_normal((8, 3, 3, 3)).astype(np.float32)
    gemm_w = rng.standard_normal((4, 8)).astype(np.float32)
    gemm_b = np.zeros(4, dtype=np.float32)

    nodes = [
        helper.make_node("Conv", ["input", "conv_w"], ["conv_out"], name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu1"),
        helper.make_node("GlobalAveragePool", ["relu_out"], ["pool_out"], name="pool1"),
        helper.make_node("Flatten", ["pool_out"], ["flat_out"], name="flatten1", axis=1),
        helper.make_node(
            "Gemm", ["flat_out", "gemm_w", "gemm_b"], ["logits"], name="fc1", transB=1
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 16, 16])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 4])],
        [
            numpy_helper.from_array(conv_w, "conv_w"),
            numpy_helper.from_array(gemm_w, "gemm_w"),
            numpy_helper.from_array(gemm_b, "gemm_b"),
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path
