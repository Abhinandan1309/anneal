"""Kaggle kernel: run examples/tensorrt/run_trt.py on a T4 and leave the JSON in /kaggle/working.

Pushed with `kaggle kernels push -p examples/tensorrt/kaggle`; MODELS/IMAGES/VARIANTS below are
edited per run. The repository is public, so the kernel clones it.
"""

import subprocess
import sys

MODELS = "efficientvit_b0,efficientnet_b0"
IMAGES = "1000"
VARIANTS = ("trt int8 (conv-only int8),trt int8 + eq (conv-only int8),"
            "modelopt int8 (modelopt convs only),modelopt int8 + eq (modelopt convs only),"
            "trt int8 + eq pos #2,trt int8 + eq pos #3")  # empty = all
REF = "main"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("nvidia-smi")
sh(f"git clone --depth 1 --branch {REF} https://github.com/Abhinandan1309/anneal.git /kaggle/temp/anneal")
# TensorRT 10: what JetPack 6 ships, and the last with implicit INT8 calibration (11 removed the
# FP16/INT8 builder flags and the calibrator)
# no separate 'onnxruntime': ModelOpt brings onnxruntime-gpu, and the two installed together break
# each other (ModelOpt's calibration failed with "CopyTensorAsync is not implemented")
sh(f"{sys.executable} -m pip uninstall -y -q onnxruntime onnxruntime-gpu")
sh(f"{sys.executable} -m pip install -q 'tensorrt>=10,<11' 'nvidia-modelopt[onnx]' timm onnx rich")
sh(f"{sys.executable} -m pip install -q --no-deps -e /kaggle/temp/anneal")
sh(f"{sys.executable} -m pip list 2>/dev/null | grep -i -E 'tensorrt|modelopt|onnx|torch'")
extra = f'--variants "{VARIANTS}"' if VARIANTS else ""
sh(f"cd /kaggle/temp/anneal && {sys.executable} examples/tensorrt/run_trt.py --models {MODELS} --images {IMAGES} "
   f"{extra} --out /kaggle/working/trt-result.json")

