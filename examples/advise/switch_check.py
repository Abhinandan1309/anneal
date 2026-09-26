"""Which model-level predictor says in advance whether equalisation is needed?

The summed at-the-site gain (rank_sites) predicted no collapse for MobileNetV3-Small (gain 5.2);
it collapsed (-66pp on ImageNet) and equalisation fixed it (-0.9pp). This compares that predictor
with a measured one on the models whose answer is known: all site tensors rounded to 8 bits at once
(joint_damage), in the plain and the equalised model, on the 64 calibration images only.

    python switch_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = Path.home() / ".anneal_cache"

MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
          "efficientvit_b0", "mobilevit_s", "lcnet_100", "fbnetv3_b", "efficientnet_b3", "efficientvit_b1",
          "ssdlite320_mobilenet_v3_large", "lraspp_mobilenet_v3_large"]


def main() -> None:
    import onnx

    from anneal.core.activation_sensitivity import joint_damage
    from anneal.core.artifact import sample_shape
    from anneal.core.dataset import load_calibset
    from anneal.core.equalize import _name_unnamed_nodes, equalise, find_sites

    sys.stdout.reconfigure(errors="replace")
    gains = json.loads((ROOT / "examples/tasks/results/predicted_gain_totals.json").read_text(encoding="utf-8"))
    out_path = HERE / "switch_check.json"
    rows = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    work = ROOT / "scratch" / "switch_check"
    work.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        if name in rows:
            continue
        src = ROOT / "examples" / "models" / f"{name}-fp32.onnx"
        if not src.exists():
            continue
        if name.startswith(("ssdlite", "lraspp")):
            sys.path.insert(0, str(ROOT / "examples" / "tasks"))
            import run_tasks as rt

            api = rt.coco()
            task = rt.SSDLite() if name.startswith("ssdlite") else rt.Segmentation()
            xs = [task.preprocess(api, i)[0] for i in task.image_ids(api)[-32:]]
        else:
            cal = load_calibset("imagenette", cache_dir=CACHE, batch_size=1, limit=64, sample_shape=sample_shape(src))
            xs = list(cal.calibration_batches(64))
        m = onnx.load(str(src))
        _name_unnamed_nodes(m)
        named = work / f"{name}-named.onnx"
        onnx.save(m, str(named))
        tensors = sorted({t for s in find_sites(m) for t in (s.x, s.y)})
        eq = work / f"{name}-eq.onnx"
        equalise(named, eq, xs[:32])
        plain_d = joint_damage(named, tensors, xs, xs)
        eq_d = joint_damage(eq, tensors, xs, xs)
        rows[name] = {"sites": len(find_sites(m)), "summed_gain": gains.get(name, {}).get("total_gain"),
                      "joint_damage_plain": plain_d, "joint_damage_equalised": eq_d, "measured_benefit": plain_d - eq_d,
                      "probe_images": len(xs)}
        print(f"{name:32s} gain {rows[name]['summed_gain']!s:>8.8s}  joint damage {100 * plain_d:5.1f}% -> {100 * eq_d:5.1f}%", flush=True)
        out_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
