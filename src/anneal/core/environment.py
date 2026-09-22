"""Is this machine in a fit state to be benchmarked?

Found the hard way: a ResNet-18 search on a laptop reported static INT8 with reduce_range
as 0.36x the FP32 speed, contradicting an earlier 1.5x. The model was fine. The battery had
dropped to 21% mid-run, the CPU was capped at 1.7 of its 2.9 GHz, and every trial measured
after that point was slow for reasons that had nothing to do with the model.

Two defences:

* :func:`snapshot` records AC/battery state and CPU clock limits, and :func:`warnings_for`
  turns a bad state into a loud warning before any number is produced.
* The optimisation loop re-measures the baseline at the end of a run; if it has drifted,
  every latency in the run is flagged (see ``anneal.agent.loop``).

Everything here is best-effort: on a platform where the facts cannot be read, they are
reported as unknown rather than guessed.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

#: A CPU whose clock ceiling is below this fraction of its rated maximum is throttled.
THROTTLE_FRACTION = 0.9
#: Baseline latency drift across a run above which its latencies are not comparable.
DRIFT_TOLERANCE = 0.10


def _windows_power() -> dict[str, Any]:
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return {}
    has_battery = status.BatteryFlag != 128  # 128 = no system battery
    return {
        "on_ac": {0: False, 1: True}.get(status.ACLineStatus),
        "battery_percent": status.BatteryLifePercent if status.BatteryLifePercent != 255 else None,
        "has_battery": has_battery,
        "battery_saver": bool(status.SystemStatusFlag & 1),
    }


def _windows_clock() -> dict[str, Any]:
    import ctypes

    class ProcessorPowerInformation(ctypes.Structure):
        _fields_ = [
            ("Number", ctypes.c_ulong),
            ("MaxMhz", ctypes.c_ulong),
            ("CurrentMhz", ctypes.c_ulong),
            ("MhzLimit", ctypes.c_ulong),
            ("MaxIdleState", ctypes.c_ulong),
            ("CurrentIdleState", ctypes.c_ulong),
        ]

    n = os.cpu_count() or 1
    buf = (ProcessorPowerInformation * n)()
    # 11 = ProcessorInformation. Returns NTSTATUS; 0 is success.
    if ctypes.windll.powrprof.CallNtPowerInformation(11, None, 0, buf, ctypes.sizeof(buf)) != 0:
        return {}
    return {
        "max_mhz": max(p.MaxMhz for p in buf),
        "limit_mhz": min(p.MhzLimit for p in buf),
        "current_mhz": round(sum(p.CurrentMhz for p in buf) / n),
    }


def _linux_power() -> dict[str, Any]:
    supplies = Path("/sys/class/power_supply")
    if not supplies.is_dir():
        return {}
    on_ac = None
    percent = None
    for dev in supplies.iterdir():
        kind = (dev / "type").read_text().strip() if (dev / "type").exists() else ""
        if kind == "Mains" and (dev / "online").exists():
            on_ac = (dev / "online").read_text().strip() == "1"
        if kind == "Battery" and (dev / "capacity").exists():
            percent = int((dev / "capacity").read_text().strip())
    return {"on_ac": on_ac, "battery_percent": percent, "has_battery": percent is not None}


def _linux_clock() -> dict[str, Any]:
    base = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    try:
        rated = int((base / "cpuinfo_max_freq").read_text()) // 1000
        limit = int((base / "scaling_max_freq").read_text()) // 1000
        current = int((base / "scaling_cur_freq").read_text()) // 1000
    except (OSError, ValueError):
        return {}
    return {"max_mhz": rated, "limit_mhz": limit, "current_mhz": current}


def snapshot() -> dict[str, Any]:
    """Power and clock facts right now. Unknown values are None, never guessed."""
    facts: dict[str, Any] = {
        "on_ac": None,
        "battery_percent": None,
        "has_battery": None,
        "battery_saver": None,
        "max_mhz": None,
        "limit_mhz": None,
        "current_mhz": None,
    }
    try:
        if sys.platform == "win32":
            facts.update(_windows_power())
            facts.update(_windows_clock())
        elif sys.platform.startswith("linux"):
            facts.update(_linux_power())
            facts.update(_linux_clock())
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass
    facts["platform"] = platform.platform()
    return facts


def warnings_for(facts: dict[str, Any]) -> list[str]:
    """Reasons this machine's latency numbers should not be trusted right now."""
    out: list[str] = []
    if facts.get("on_ac") is False and facts.get("has_battery"):
        pct = facts.get("battery_percent")
        out.append(
            f"running on battery{f' ({pct}%)' if pct is not None else ''}: the OS may cut CPU "
            f"clocks mid-run, which distorts every latency after it. Plug in."
        )
    if facts.get("battery_saver"):
        out.append("battery saver is on, which caps CPU performance")
    rated, limit = facts.get("max_mhz"), facts.get("limit_mhz")
    if rated and limit and limit < THROTTLE_FRACTION * rated:
        out.append(
            f"CPU clock is capped at {limit} MHz of a rated {rated} MHz "
            f"({limit / rated * 100:.0f}%): power or thermal throttling"
        )
    return out


def drift(start_ms: float, end_ms: float) -> float:
    """Relative change in a latency measured at the start and end of a run."""
    return abs(end_ms - start_ms) / start_ms if start_ms > 0 else float("nan")


#: Instruction-set features that decide which INT8 kernels onnxruntime can use.
INT8_FEATURES = ("avx2", "avx512f", "avx512_vnni", "avx512vnni", "avx_vnni", "avxvnni",
                 "amx_int8", "asimddp", "dotprod", "i8mm", "sve")


def cpu_features() -> dict[str, Any]:
    """CPU identity and the INT8-relevant instruction-set features, best effort.

    ``int8_path`` summarises what matters for accumulator saturation:

    * ``"x86-avx2-16bit"`` — x86 without VNNI: u8×s8 pairs are summed in saturating 16-bit
    * ``"x86-vnni"`` — x86 with VNNI: 32-bit accumulation
    * ``"arm-dotprod"`` — ARM with the dot-product extension: 32-bit accumulation
    * ``"unknown"`` — could not tell
    """
    import subprocess

    info: dict[str, Any] = {
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
        import cpuinfo  # py-cpuinfo, optional

        ci = cpuinfo.get_cpu_info()
        info["brand"] = ci.get("brand_raw")
        flags |= set(ci.get("flags", []))
    except Exception:  # noqa: BLE001
        pass
    if sys.platform.startswith("linux") and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("flags", "features")):
                flags |= set(line.split(":", 1)[1].split())
            if line.lower().startswith("model name") and not info.get("brand"):
                info["brand"] = line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        try:
            info["brand"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip() or info.get("brand")
            if subprocess.run(["sysctl", "-n", "hw.optional.arm.FEAT_DotProd"],
                              capture_output=True, text=True).stdout.strip() == "1":
                flags.add("dotprod")
        except OSError:
            pass
    lower = {f.lower() for f in flags}
    info["int8_features"] = sorted(f for f in lower if f in INT8_FEATURES)
    info["has_vnni"] = bool(lower & {"avx512_vnni", "avx512vnni", "avx_vnni", "avxvnni"})
    info["has_arm_dotprod"] = bool(lower & {"asimddp", "dotprod"})
    machine = info["machine"].lower()
    if info["has_arm_dotprod"]:
        info["int8_path"] = "arm-dotprod"
    elif info["has_vnni"]:
        info["int8_path"] = "x86-vnni"
    elif machine in ("amd64", "x86_64", "x64") and "avx2" in lower:
        info["int8_path"] = "x86-avx2-16bit"
    else:
        info["int8_path"] = "unknown"
    return info
