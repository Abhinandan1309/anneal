"""The classic U-Net (conv-BN-ReLU double convs, max-pool down, transposed-conv up, encoder
features concatenated into the decoder), pretrained on Carvana car masks by milesial/Pytorch-UNet.

The architecture is written out here (module names match the checkpoint) rather than loaded
with torch.hub, which would execute the repository's code; only the weights are downloaded, and
they are read with ``torch.load(weights_only=True)``, which cannot execute code.

    python unet_carvana.py            # writes examples/models/unet_carvana-fp32.onnx
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_URL = "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth"
CACHE = Path.home() / ".anneal_cache" / "unet"
#: Input size: Carvana at scale 0.5 is 959x640; 192x288 keeps the aspect, is divisible by 16, and keeps
#: calibration (every activation of every image held in RAM) within this laptop's memory.
SIZE = (192, 288)


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int, mid: int | None = None) -> None:
        super().__init__()
        mid = mid or cout
        self.double_conv = nn.Sequential(
            nn.Conv2d(cin, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(cin, cout))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(cin, cout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 2) -> None:
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1, self.down2, self.down3, self.down4 = Down(64, 128), Down(128, 256), Down(256, 512), Down(512, 1024)
        self.up1, self.up2, self.up3, self.up4 = Up(1024, 512), Up(512, 256), Up(256, 128), Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def pretrained() -> UNet:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / Path(WEIGHTS_URL).name
    if not path.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state.pop("mask_values", None)
    net = UNet()
    net.load_state_dict(state)
    return net.eval()


def export(dst: Path | None = None) -> Path:
    dst = dst or ROOT / "examples" / "models" / "unet_carvana-fp32.onnx"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(pretrained(), torch.rand(1, 3, *SIZE), str(dst), input_names=["input"],
                          output_names=["logits"], opset_version=17, dynamo=False)
    return dst


if __name__ == "__main__":
    print(export())
    sys.exit(0)
