"""Export more edge classifiers built from gated activations feeding depthwise convs.

torchvision: EfficientNet-B2 (260 px), B3 (300 px), MobileNetV3-Small.
timm: EfficientViT-B0/B1 (Hardswish -> depthwise; the September 2026 TensorRT collapse report),
MobileViT-S (SiLU -> depthwise in its MV2 blocks), LCNet-100 and FBNetV3-B (Hardswish ->
depthwise). Every model takes the ImageNet-normalised input the rest of the repository uses; a
fixed affine adapter converts it to the model's own normalisation (MobileViT expects plain
[0, 1] pixels), so accuracy deltas stay paired and only absolute FP32 accuracy may shift.

    python export_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "models"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TORCHVISION = {"efficientnet_b2": 260, "efficientnet_b3": 300, "mobilenet_v3_small": 224}
TIMM = ["efficientvit_b0.r224_in1k", "efficientvit_b1.r224_in1k", "mobilevit_s.cvnets_in1k",
        "lcnet_100.ra2_in1k", "fbnetv3_b.ra2_in1k"]


class Renormalise(torch.nn.Module):
    """ImageNet-normalised input -> the model's own normalisation (identity when they agree)."""

    def __init__(self, net, mean, std) -> None:
        super().__init__()
        m0, s0 = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        m1, s1 = torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer("a", s0 / s1)
        self.register_buffer("b", (m0 - m1) / s1)
        self.identity = bool(torch.allclose(self.a, torch.ones_like(self.a)) and torch.allclose(self.b, torch.zeros_like(self.b)))
        self.net = net

    def forward(self, x):
        return self.net(x if self.identity else x * self.a + self.b)


def export(model: torch.nn.Module, size: int, dst: Path) -> None:
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, torch.randn(1, 3, size, size), str(dst), input_names=["input"],
                          output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)


def main() -> None:
    import timm
    import torchvision

    OUT.mkdir(parents=True, exist_ok=True)
    for name, size in TORCHVISION.items():
        dst = OUT / f"{name}-fp32.onnx"
        if not dst.exists():
            export(torchvision.models.get_model(name, weights="DEFAULT"), size, dst)
        print(f"{name}: {dst}", flush=True)
    for full in TIMM:
        name = full.split(".")[0]
        dst = OUT / f"{name}-fp32.onnx"
        if not dst.exists():
            net = timm.create_model(full, pretrained=True)
            cfg = net.pretrained_cfg
            export(Renormalise(net, cfg["mean"], cfg["std"]), cfg["input_size"][-1], dst)
        print(f"{name}: {dst}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
