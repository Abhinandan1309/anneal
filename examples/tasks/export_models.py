"""Export the popular edge segmentation and detection networks to ONNX, fixed input size.

* ``lraspp_mobilenet_v3_large``    torchvision, COCO with the 21 VOC classes; 512x512 in,
                                   per-pixel logits out
* ``ssdlite320_mobilenet_v3_large`` torchvision, COCO; 320x320 in, raw head outputs out
                                   (box regression + class logits per anchor). Anchor decoding
                                   and NMS stay in PyTorch, outside the quantized graph, as they
                                   would in a deployed pipeline.
* ``yolov8n``                      Ultralytics (AGPL-3.0; used for benchmarking only, no code
                                   copied), COCO; 640x640 in, raw box and class logits out (see
                                   ``yolo_raw``). SiLU everywhere and no depthwise convs: the
                                   stress test for an equalisation that needs a depthwise consumer.

    python export_models.py            # all three
    python export_models.py --models lraspp_mobilenet_v3_large
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "models"


class LRASPP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large

        self.net = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT).eval()

    def forward(self, x):
        return self.net(x)["out"]


class SSDLiteHead(torch.nn.Module):
    """Backbone + head only; torchvision's own normalisation is applied by the caller."""

    def __init__(self, net) -> None:
        super().__init__()
        self.backbone, self.head = net.backbone, net.head

    def forward(self, x):
        features = list(self.backbone(x).values())
        out = self.head(features)
        return out["bbox_regression"], out["cls_logits"]


def ssdlite():
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large

    return ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).eval()


def yolo_raw():
    """YOLOv8n with the Detect head's decode cut off: raw DFL box logits (1, 64, 8400) and
    pre-sigmoid class logits (1, 80, 8400). Ultralytics' own ONNX output concatenates decoded
    pixel boxes (0-700) with sigmoid scores (0-1), and one INT8 scale on that tensor rounds every
    score to zero; the decode runs in PyTorch instead, as SSDLite's does."""
    import os

    from ultralytics import YOLO

    weights = Path(__file__).resolve().parents[2] / "scratch" / "tasks"
    weights.mkdir(parents=True, exist_ok=True)
    here = os.getcwd()
    os.chdir(weights)  # Ultralytics downloads yolov8n.pt into the working directory
    try:
        model = YOLO("yolov8n.pt").model.fuse().eval()
    finally:
        os.chdir(here)
    head = model.model[-1]
    head.export = True
    head._inference = lambda x: (x["boxes"], x["scores"])
    return model


def export(name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{name}-fp32.onnx"
    if name == "lraspp_mobilenet_v3_large":
        model, x, outputs = LRASPP().eval(), torch.randn(1, 3, 512, 512), ["logits"]
    elif name == "ssdlite320_mobilenet_v3_large":
        model, x, outputs = SSDLiteHead(ssdlite()).eval(), torch.randn(1, 3, 320, 320), ["bbox_regression", "cls_logits"]
    elif name == "yolov8n":
        model, x, outputs = yolo_raw(), torch.randn(1, 3, 640, 640), ["boxes", "scores"]
    else:
        raise ValueError(name)
    with torch.no_grad():
        torch.onnx.export(model, x, str(dst), input_names=["input"], output_names=outputs,
                          opset_version=17, dynamo=False)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lraspp_mobilenet_v3_large,ssdlite320_mobilenet_v3_large,yolov8n")
    args = ap.parse_args()
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"{name}: {export(name)}", flush=True)


if __name__ == "__main__":
    main()
