"""Artifact identity, lineage, and reading a graph's real signature."""

from __future__ import annotations

from pathlib import Path

import pytest

from anneal.core.artifact import (
    ModelArtifact,
    TransformRecord,
    concrete_input_shape,
    describe_io,
    model_batch_dim,
)


def test_bare_artifact_is_labelled_baseline():
    artifact = ModelArtifact(path=Path("m.onnx"))
    assert artifact.label == "baseline"
    assert artifact.lineage_key == "baseline"


def test_transform_record_renders_params_deterministically():
    a = TransformRecord("q", {"b": 2, "a": 1})
    b = TransformRecord("q", {"a": 1, "b": 2})
    # Sorted keys mean two equivalent recipes hash to the same lineage key.
    assert str(a) == str(b) == "q(a=1,b=2)"


def test_transform_record_without_params_is_just_its_name():
    assert str(TransformRecord("fuse")) == "fuse"


def test_derive_appends_to_lineage_without_mutating_the_parent():
    base = ModelArtifact(path=Path("m.onnx"))
    child = base.derive(TransformRecord("graph_optimize", {"level": "all"}), Path("c.onnx"))

    assert base.lineage == ()
    assert child.label == "graph_optimize(level=all)"
    assert child.path == Path("c.onnx")


def test_lineage_key_distinguishes_order_of_composition():
    fuse = TransformRecord("fuse")
    quant = TransformRecord("quant")
    m = Path("m.onnx")

    a = ModelArtifact(path=m).derive(fuse, m).derive(quant, m)
    b = ModelArtifact(path=m).derive(quant, m).derive(fuse, m)
    assert a.lineage_key != b.lineage_key


def test_derive_carries_metadata_forward():
    base = ModelArtifact(path=Path("m.onnx"), meta={"source": "torchvision:resnet18"})
    child = base.derive(TransformRecord("q"), Path("c.onnx"), excluded_nodes=["conv1"])
    assert child.meta["source"] == "torchvision:resnet18"
    assert child.meta["excluded_nodes"] == ["conv1"]


def test_round_trip_preserves_lineage():
    original = ModelArtifact(path=Path("m.onnx")).derive(
        TransformRecord("quantize_dynamic_int8", {"per_channel": True}), Path("q.onnx")
    )
    restored = ModelArtifact.from_dict(original.to_dict())
    assert restored.lineage_key == original.lineage_key
    assert restored.path == original.path


def test_content_hash_is_stable_and_distinguishes_files(tmp_path: Path):
    a, b = tmp_path / "a.onnx", tmp_path / "b.onnx"
    a.write_bytes(b"same")
    b.write_bytes(b"different")

    assert ModelArtifact(path=a).content_hash() == ModelArtifact(path=a).content_hash()
    assert ModelArtifact(path=a).content_hash() != ModelArtifact(path=b).content_hash()


def test_size_bytes_reports_the_file_size(tmp_path: Path):
    path = tmp_path / "m.onnx"
    path.write_bytes(b"x" * 1234)
    assert ModelArtifact(path=path).size_bytes == 1234


# ----- graph introspection -------------------------------------------------


def test_describe_io_reads_names_and_dynamic_dims(tiny_onnx: Path):
    io = describe_io(tiny_onnx)
    assert [i["name"] for i in io["inputs"]] == ["input"]
    assert [o["name"] for o in io["outputs"]] == ["logits"]
    # The batch axis is symbolic and must be reported as unknown, not guessed.
    assert io["inputs"][0]["shape"] == [None, 3, 16, 16]


def test_describe_io_excludes_initializers_from_inputs(tiny_onnx: Path):
    names = {i["name"] for i in describe_io(tiny_onnx)["inputs"]}
    assert "conv_w" not in names and "gemm_w" not in names


def test_dynamic_batch_dim_reports_none(tiny_onnx: Path):
    assert model_batch_dim(tiny_onnx) is None


def test_concrete_input_shape_fills_the_batch_axis():
    spec = {"name": "input", "shape": [None, 3, 224, 224]}
    assert concrete_input_shape(spec, batch_size=4) == (4, 3, 224, 224)


def test_concrete_input_shape_refuses_to_guess_spatial_dims():
    spec = {"name": "input", "shape": [1, 3, None, None]}
    with pytest.raises(ValueError, match="will not guess"):
        concrete_input_shape(spec)


def test_concrete_input_shape_leaves_a_fixed_batch_alone():
    spec = {"name": "input", "shape": [1, 3, 224, 224]}
    assert concrete_input_shape(spec, batch_size=8) == (1, 3, 224, 224)
