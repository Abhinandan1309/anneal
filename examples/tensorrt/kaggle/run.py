"""Kaggle kernel: run examples/tensorrt/run_trt.py on a T4 and leave the JSON in /kaggle/working.

Pushed with `kaggle kernels push -p examples/tensorrt/kaggle`; MODELS/IMAGES/VARIANTS below are
edited per run. The repository is public, so the kernel clones it.
"""

import subprocess
import sys

MODELS = "efficientnet_b0,mobilenet_v3_large,efficientnet_b1,efficientvit_b0"
IMAGES = "1000"
VARIANTS = ""  # empty = all
REF = "main"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("nvidia-smi")
sh(f"git clone --depth 1 --branch {REF} https://github.com/Abhinandan1309/anneal.git /kaggle/temp/anneal")
sh(f"{sys.executable} -m pip install -q tensorrt 'nvidia-modelopt[onnx]' timm onnx onnxruntime rich")
sh(f"{sys.executable} -m pip install -q --no-deps -e /kaggle/temp/anneal")
sh(f"{sys.executable} -m pip list 2>/dev/null | grep -i -E 'tensorrt|modelopt|onnx|torch'")
extra = f'--variants "{VARIANTS}"' if VARIANTS else ""
sh(f"cd /kaggle/temp/anneal && {sys.executable} examples/tensorrt/run_trt.py --models {MODELS} --images {IMAGES} "
   f"{extra} --out /kaggle/working/trt-result.json")
