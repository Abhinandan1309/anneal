"""Predicted equalisation gain per model (rank_sites, summed over sites), the model-level switch.

8 calibration images per model (Imagenette for the classifiers, the 8 highest-id COCO val2017
images for SSDLite). Writes results/predicted_gain_totals.json.

    python examples/tasks/predicted_gain.py
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anneal.core.equalize import rank_sites
from anneal.core.dataset import load_calibset
C = Path.home() / ".anneal_cache"
out = {}
cal = load_calibset("imagenette", cache_dir=C, batch_size=1, limit=8)
imgs = [x for x in cal.calibration_batches(8)]
for m in ["efficientnet_b0", "mobilenet_v3_large", "efficientnet_b1", "efficientnet_v2_s", "mobilenet_v2"]:
    p = Path(f"examples/models/{m}-fp32.onnx")
    if p.exists():
        g = [gain for _, gain in rank_sites(p, [np.concatenate(imgs)])]
        out[m] = g
import run_tasks as rt
api = rt.coco(); t = rt.SSDLite(); ids = t.image_ids(api)[-8:]
xs = [t.preprocess(api, i)[0] for i in ids]
out["ssdlite320_mobilenet_v3_large"] = [g for _, g in rank_sites(Path("examples/models/ssdlite320_mobilenet_v3_large-fp32.onnx"), xs)]
rows = {m: {"sites": len(g), "total_gain": float(np.sum(g)), "per_site_desc": sorted(map(float, g), reverse=True)}
        for m, g in out.items()}
(Path(__file__).resolve().parent / "results" / "predicted_gain_totals.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
for m, g in out.items():
    g = np.array(g)
    print(f"{m:32s} sites {len(g):3d}  total gain {g.sum():7.2f}  max {g.max() if len(g) else 0:6.2f}  top3 {np.sort(g)[::-1][:3].round(2)}")
