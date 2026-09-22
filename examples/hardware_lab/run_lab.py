"""One hardware-lab job: identify this machine's CPU, then run the static INT8 recipe study.

Runs on GitHub Actions runners of different architectures (see
.github/workflows/hardware-lab.yml) and on any laptop. The question it answers: does the
full-range per-channel static INT8 failure found on a Zen 2 laptop — and the reduce_range
fix — depend on the CPU's INT8 instructions?

* x86 with VNNI (avx512_vnni / avx_vnni) has a native u8 x s8 dot product that accumulates
  in 32 bits. onnxruntime documents saturation problems for x86 *without* it.
* ARM with the dot-product extension (asimddp) has its own INT8 path.

    python examples/hardware_lab/run_lab.py --eval-limit 1024 --out lab-result.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
MODEL = EXAMPLES / "resnet18-cpu1t" / "models" / "resnet18-fp32.onnx"

from anneal.core.environment import cpu_features as cpu_info  # noqa: E402


def ensure_model() -> None:
    if MODEL.exists():
        return
    from anneal.models import export_torchvision

    export_torchvision("resnet18", MODEL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-limit", type=int, default=1024)
    parser.add_argument("--target", default="cpu-1t")
    parser.add_argument("--out", default="lab-result.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    info = cpu_info()
    print(json.dumps(info, indent=2))
    ensure_model()

    study_out = Path(args.out).with_suffix(".study.json")
    subprocess.run(
        [sys.executable, str(EXAMPLES / "static_recipe_ab.py"),
         "--eval-limit", str(args.eval_limit), "--target", args.target, "--out", str(study_out)],
        check=True,
    )
    study = json.loads(study_out.read_text(encoding="utf-8"))
    import onnxruntime

    Path(args.out).write_text(json.dumps({
        "cpu": info,
        "onnxruntime": onnxruntime.__version__,
        "python": platform.python_version(),
        "study": study,
    }, indent=2), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
