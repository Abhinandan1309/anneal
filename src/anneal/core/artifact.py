"""A model artifact and the lineage of transforms that produced it.

An artifact is deliberately *file-backed*. Optimisation pipelines that keep models in
memory tend to produce results nobody can reproduce six months later; here every
candidate exists as a real ``.onnx`` on disk, alongside the exact chain of transforms
that created it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TransformRecord:
    """One applied transform: its name and the exact parameters used."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.params:
            return self.name
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({inner})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TransformRecord:
        return cls(name=d["name"], params=dict(d.get("params", {})))


@dataclass(frozen=True)
class ModelArtifact:
    """An ONNX model on disk plus the provenance of how it got there."""

    path: Path
    lineage: tuple[TransformRecord, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    # ----- identity -------------------------------------------------------

    @property
    def label(self) -> str:
        """Human-readable description of the transform chain."""
        if not self.lineage:
            return "baseline"
        return " -> ".join(str(t) for t in self.lineage)

    @property
    def lineage_key(self) -> str:
        """Stable key for deduplicating candidates that are the same recipe."""
        return "|".join(str(t) for t in self.lineage) or "baseline"

    def content_hash(self) -> str:
        """SHA-256 of the model bytes. Proves two artifacts really are identical."""
        h = hashlib.sha256()
        with self.path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # ----- properties -----------------------------------------------------

    @property
    def size_bytes(self) -> int:
        total = self.path.stat().st_size
        # ONNX splits tensors >2GB into sidecar files; count them too.
        for sidecar in self.path.parent.glob(f"{self.path.name}.data*"):
            total += sidecar.stat().st_size
        return total

    def derive(self, record: TransformRecord, new_path: Path, **meta: Any) -> ModelArtifact:
        """Return a new artifact one transform further along the chain."""
        return ModelArtifact(
            path=Path(new_path),
            lineage=self.lineage + (record,),
            meta={**self.meta, **meta},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "lineage": [t.to_dict() for t in self.lineage],
            "label": self.label,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelArtifact:
        return cls(
            path=Path(d["path"]),
            lineage=tuple(TransformRecord.from_dict(t) for t in d.get("lineage", ())),
            meta=dict(d.get("meta", {})),
        )

    def __str__(self) -> str:
        return f"<ModelArtifact {self.label} @ {self.path.name}>"


def describe_io(model_path: Path) -> dict[str, Any]:
    """Read the input/output signature straight out of the ONNX graph.

    Dynamic dimensions come back as ``None`` rather than being silently guessed —
    callers decide what concrete shape to feed.
    """
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)

    def _spec(value_info) -> dict[str, Any]:
        dims: list[int | None] = []
        tt = value_info.type.tensor_type
        for d in tt.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                dims.append(int(d.dim_value))
            else:
                dims.append(None)
        return {
            "name": value_info.name,
            "dtype": int(tt.elem_type),
            "shape": dims,
        }

    initializers = {init.name for init in model.graph.initializer}
    return {
        "inputs": [_spec(i) for i in model.graph.input if i.name not in initializers],
        "outputs": [_spec(o) for o in model.graph.output],
        "opset": [{"domain": op.domain, "version": op.version} for op in model.opset_import],
        "producer": model.producer_name,
    }


def concrete_input_shape(spec: dict[str, Any], batch_size: int = 1) -> tuple[int, ...]:
    """Resolve a possibly-dynamic input spec to a concrete shape.

    The leading dynamic dim is treated as batch; any other dynamic dim is an error,
    because guessing a spatial size would silently produce meaningless benchmarks.
    """
    shape = list(spec["shape"])
    if not shape:
        raise ValueError(f"input {spec['name']!r} has no shape")
    if shape[0] is None:
        shape[0] = batch_size
    unresolved = [i for i, d in enumerate(shape) if d is None]
    if unresolved:
        raise ValueError(
            f"input {spec['name']!r} has dynamic dimensions at {unresolved} that Anneal "
            f"will not guess; export the model with a fixed shape or pass --input-shape"
        )
    return tuple(int(d) for d in shape)


def model_batch_dim(model_path: Path) -> int | None:
    """The model's fixed batch size, or ``None`` if the batch axis is dynamic.

    Callers use this to reconcile eval batching with what the graph will actually accept,
    rather than discovering the mismatch as an onnxruntime shape error mid-run.
    """
    io = describe_io(model_path)
    if not io["inputs"]:
        return None
    shape = io["inputs"][0]["shape"]
    if not shape:
        return None
    return shape[0]


def sample_shape(model_path: Path) -> tuple[int, ...] | None:
    """One input's shape without the batch axis, or None if any of it is dynamic."""
    io = describe_io(model_path)
    if not io["inputs"]:
        return None
    dims = io["inputs"][0]["shape"][1:]
    if not dims or any(d is None for d in dims):
        return None
    return tuple(int(d) for d in dims)


def write_sidecar(artifact: ModelArtifact, dest: Path) -> None:
    """Persist an artifact's provenance next to the model file."""
    dest.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
