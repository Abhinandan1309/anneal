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
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
MODEL = EXAMPLES / "resnet18-cpu1t" / "models" / "resnet18-fp32.onnx"

#: Instruction-set features that decide which INT8 kernels onnxruntime can use.
INT8_FEATURES = ("avx2", "avx512f", "avx512_vnni", "avx512vnni", "avx_vnni", "avxvnni",
                 "amx_int8", "asimddp", "dotprod", "i8mm", "sve")


def cpu_info() -> dict:
    """Best-effort CPU identity and INT8-relevant feature flags."""
    info: dict = {
        "machine": platform.machine(),
        "system": platform.system(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "runner": os.environ.get("RUNNER_NAME"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
    }
    flags: set[str] = set()
    try:
        import cpuinfo  # py-cpuinfo

        ci = cpuinfo.get_cpu_info()
        info["brand"] = ci.get("brand_raw")
        flags |= set(ci.get("flags", []))
    except Exception:  # noqa: BLE001
        pass
    if sys.platform.startswith("linux") and Path("/proc/cpuinfo").exists():
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.lower().startswith(("flags", "features")):
                flags |= set(line.split(":", 1)[1].split())
            if line.lower().startswith("model name") and not info.get("brand"):
                info["brand"] = line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        try:
            info["brand"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip() or info.get("brand")
            dot = subprocess.run(["sysctl", "-n", "hw.optional.arm.FEAT_DotProd"],
                                 capture_output=True, text=True).stdout.strip()
            if dot == "1":
                flags.add("dotprod")
        except OSError:
            pass
    info["int8_features"] = sorted(f for f in flags if f.lower() in INT8_FEATURES)
    lower = {f.lower() for f in flags}
    info["has_vnni"] = bool(lower & {"avx512_vnni", "avx512vnni", "avx_vnni", "avxvnni"})
    info["has_arm_dotprod"] = bool(lower & {"asimddp", "dotprod"})
    return info


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
