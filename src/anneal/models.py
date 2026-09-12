"""Getting a model into Anneal.

Anneal operates on ONNX. This module is the on-ramp: export a torchvision classifier, or
point at an ``.onnx`` you already have. Torch is an *optional* dependency — the core
optimisation loop never imports it.
"""

from __future__ import annotations

from pathlib import Path

from anneal.core.artifact import ModelArtifact

#: Models that export cleanly and are worth benchmarking on edge silicon.
SUGGESTED = (
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "squeezenet1_1",
)


def export_torchvision(
    name: str,
    out_path: Path,
    *,
    batch_size: int = 1,
    image_size: int = 224,
    opset: int = 17,
    weights: str = "DEFAULT",
) -> ModelArtifact:
    """Export a pretrained torchvision classifier to ONNX.

    Spatial dimensions are deliberately static — onnxruntime cannot pick
    layout-specialised kernels without knowing them, so a model exported with dynamic
    H/W benchmarks slower and tells you nothing about the real deployment. The batch
    axis *is* dynamic, because evaluation wants to push 16 images at a time while the
    latency benchmark wants a batch of 1, and re-exporting between the two would mean
    measuring two different graphs.
    """
    try:
        import torch
        import torchvision.models as tvm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "exporting a torchvision model needs the optional torch extra: "
            "pip install 'anneal[torch]'"
        ) from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not hasattr(tvm, name):
        raise ValueError(f"torchvision has no model {name!r}; try one of {SUGGESTED}")

    builder = getattr(tvm, name)
    model = builder(weights=weights)
    model.eval()

    dummy = torch.randn(batch_size, 3, image_size, image_size)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    return ModelArtifact(
        path=out_path,
        lineage=(),
        meta={
            "source": f"torchvision:{name}",
            "weights": weights,
            "image_size": image_size,
            "batch_size": batch_size,
            "opset": opset,
        },
    )


def load_onnx(path: Path) -> ModelArtifact:
    """Wrap an existing .onnx file as a baseline artifact."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such model: {path}")
    if path.suffix.lower() != ".onnx":
        raise ValueError(f"expected a .onnx file, got {path.name}")
    return ModelArtifact(path=path, lineage=(), meta={"source": f"file:{path.name}"})


def resolve_model(spec: str, workdir: Path, *, batch_size: int = 1, image_size: int = 224) -> ModelArtifact:
    """Resolve a model spec: either ``torchvision:resnet18`` or a path to an .onnx."""
    if spec.startswith("torchvision:"):
        name = spec.split(":", 1)[1]
        out = Path(workdir) / f"{name}-fp32.onnx"
        if out.exists():
            return ModelArtifact(
                path=out, lineage=(), meta={"source": spec, "image_size": image_size}
            )
        return export_torchvision(name, out, batch_size=batch_size, image_size=image_size)
    return load_onnx(Path(spec))
