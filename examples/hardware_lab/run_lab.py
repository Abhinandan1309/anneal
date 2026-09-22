"""One hardware-lab job: identify this machine's CPU, then run the static INT8 recipe study.

Runs on GitHub Actions runners of different architectures (see
.github/workflows/hardware-lab.yml) and on any laptop. The question it answers: does the
full-range per-channel static INT8 failure found on a Zen 2 laptop — and the reduce_range
fix — depend on the CPU's INT8 instructions?

* x86 with VNNI (avx512_vnni / avx_vnni) has a native u8 x s8 dot product that accumulates
  in 32 bits. onnxruntime documents saturation problems for x86 *without* it.
* ARM with the dot-product extension (asimddp) has its own INT8 path.

    python examples/hardware_lab/run_lab.py --models resnet18,mobilenet_v3_large --out lab.json
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
from anneal.core.environment import cpu_features as cpu_info  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-limit", type=int, default=1024)
    parser.add_argument("--target", default="cpu-1t")
    parser.add_argument("--out", default="lab-result.json")
    parser.add_argument("--models", default="resnet18",
                        help="comma-separated torchvision model names")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    info = cpu_info()
    print(json.dumps(info, indent=2))
    studies = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        study_out = Path(args.out).with_suffix(f".{name}.study.json")
        subprocess.run(
            [sys.executable, str(EXAMPLES / "static_recipe_ab.py"), "--model", name,
             "--eval-limit", str(args.eval_limit), "--target", args.target,
             "--out", str(study_out)],
            check=True,
        )
        studies[name] = json.loads(study_out.read_text(encoding="utf-8"))
    import onnxruntime

    Path(args.out).write_text(json.dumps({
        "cpu": info,
        "onnxruntime": onnxruntime.__version__,
        "python": platform.python_version(),
        "studies": studies,
        # Kept for results made before multi-model runs, which have a single "study".
        **({"study": studies["resnet18"]} if "resnet18" in studies else {}),
    }, indent=2), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
