"""Hardware targets.

A *target* is the thing you are optimising **for**. The whole premise of Anneal is that
"make the model faster" is meaningless without naming the silicon — an INT8 graph that
flies on a Jetson can be slower than FP32 on a desktop CPU that lacks VNNI, and a
transform that helps a TDA4VM's fixed-function accelerator can be a no-op elsewhere.

Targets that need vendor toolchains (TIDL, TensorRT, Edge TPU) are *declared* here but
raise a precise, actionable error until an adapter is supplied. Declaring them keeps the
extension point honest and visible instead of pretending CPU is the whole world.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any


class TargetUnavailable(RuntimeError):
    """Raised when a target is declared but its runtime is not present on this machine."""


@dataclass(frozen=True)
class Target:
    """An execution target: a runtime, its providers, and its threading policy."""

    name: str
    providers: tuple[str, ...]
    description: str
    intra_op_threads: int = 1
    inter_op_threads: int = 1
    graph_opt_level: str = "all"
    #: Providers that must be present in onnxruntime for this target to run here.
    requires_providers: tuple[str, ...] = ()
    #: Set for targets that need a vendor SDK Anneal does not bundle.
    adapter_hint: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def ensure_available(self) -> None:
        """Raise a precise error if this target cannot actually execute here."""
        if self.adapter_hint is not None:
            raise TargetUnavailable(
                f"target {self.name!r} requires a vendor adapter that Anneal does not "
                f"bundle.\n  {self.adapter_hint}\n"
                f"Implement anneal.core.targets.Target for it and register via "
                f"register_target(), or run on a CPU target to develop the recipe first."
            )
        if not self.requires_providers:
            return
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        missing = [p for p in self.requires_providers if p not in available]
        if missing:
            raise TargetUnavailable(
                f"target {self.name!r} needs onnxruntime provider(s) {missing}, which are "
                f"not installed. Available here: {sorted(available)}"
            )

    def is_available(self) -> bool:
        try:
            self.ensure_available()
        except (TargetUnavailable, ImportError):
            return False
        return True

    def session_options(self) -> Any:
        """Build an onnxruntime SessionOptions matching this target's policy."""
        import onnxruntime as ort

        levels = {
            "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        opts = ort.SessionOptions()
        opts.graph_optimization_level = levels[self.graph_opt_level]
        opts.intra_op_num_threads = self.intra_op_threads
        opts.inter_op_num_threads = self.inter_op_threads
        # Deterministic scheduling makes latency percentiles comparable between trials.
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return opts

    def fingerprint(self) -> dict[str, Any]:
        """Everything a reader needs to judge whether our numbers transfer to their box."""
        fp: dict[str, Any] = {
            "target": self.name,
            "providers": list(self.providers),
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
            "graph_opt_level": self.graph_opt_level,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        try:
            import onnxruntime as ort

            fp["onnxruntime"] = ort.__version__
        except ImportError:  # pragma: no cover - onnxruntime is a hard dep in practice
            fp["onnxruntime"] = None
        return fp


_BUILTIN: dict[str, Target] = {
    "cpu-1t": Target(
        name="cpu-1t",
        providers=("CPUExecutionProvider",),
        description="Single-threaded CPU. The cleanest signal for per-op cost.",
        intra_op_threads=1,
        tags=("cpu", "deterministic"),
    ),
    "cpu-4t": Target(
        name="cpu-4t",
        providers=("CPUExecutionProvider",),
        description="4-thread CPU. Closer to how a desktop service is actually served.",
        intra_op_threads=4,
        tags=("cpu",),
    ),
    "cuda": Target(
        name="cuda",
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        description="NVIDIA GPU via the CUDA execution provider.",
        intra_op_threads=1,
        requires_providers=("CUDAExecutionProvider",),
        tags=("gpu", "nvidia"),
    ),
    "tensorrt": Target(
        name="tensorrt",
        providers=("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"),
        description="NVIDIA GPU/Jetson via TensorRT.",
        requires_providers=("TensorrtExecutionProvider",),
        tags=("gpu", "nvidia", "edge"),
    ),
    "jetson-orin": Target(
        name="jetson-orin",
        providers=("TensorrtExecutionProvider", "CUDAExecutionProvider"),
        description="NVIDIA Jetson Orin, TensorRT engine with INT8/FP16 tactics.",
        adapter_hint=(
            "Run Anneal on the Orin itself with onnxruntime-gpu built against TensorRT, "
            "then use the 'tensorrt' target."
        ),
        tags=("edge", "nvidia"),
    ),
    "tda4vm": Target(
        name="tda4vm",
        providers=("TIDLExecutionProvider",),
        description="TI TDA4VM (J721E) C7x/MMA via the TIDL runtime.",
        adapter_hint=(
            "Needs TI's TIDL tools and the TIDLExecutionProvider build of onnxruntime, "
            "plus on-device measurement. Develop the quantization recipe on cpu-1t first; "
            "the sensitivity ranking transfers, the absolute latencies do not."
        ),
        tags=("edge", "ti", "npu"),
    ),
    "coral-edgetpu": Target(
        name="coral-edgetpu",
        providers=("EdgeTPUExecutionProvider",),
        description="Google Coral Edge TPU. INT8-only, requires the edgetpu compiler.",
        adapter_hint=(
            "The Edge TPU is INT8-only and runs TFLite, not ONNX. Convert after Anneal "
            "has chosen a quantization recipe, then measure with pycoral on-device."
        ),
        tags=("edge", "google", "npu", "int8-only"),
    ),
}


def register_target(target: Target) -> None:
    """Register a custom target (e.g. an in-house board with its own runtime)."""
    _BUILTIN[target.name] = target


def get_target(name: str) -> Target:
    try:
        return _BUILTIN[name]
    except KeyError:
        raise KeyError(
            f"unknown target {name!r}. Known targets: {sorted(_BUILTIN)}"
        ) from None


def list_targets() -> list[Target]:
    return list(_BUILTIN.values())


def default_target() -> Target:
    """Pick the best target that actually works on this machine."""
    for name in ("cuda", "cpu-4t", "cpu-1t"):
        t = _BUILTIN[name]
        if t.is_available():
            return t
    return _BUILTIN["cpu-1t"]
