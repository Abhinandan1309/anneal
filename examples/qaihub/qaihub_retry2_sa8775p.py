"""Second SA8775P retry: equalised inference in 4 chunks of 256; FP32 / hub-int8 profiles once more.
Scores the equalised variant against the device's own FP32 predictions (job jpe76xz05)."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import qai_hub as hub
from anneal.core.audit import mcnemar_exact, paired_delta_ci
from anneal.core.dataset import load_evalset

C = Path.home() / ".anneal_cache"
device = hub.Device("SA8775P ADP")
compiled = {"fp32": "jgoldwrdg", "hub int8": "jpvl2mdm5", "hub int8 + equalised": "jgjr3y78p"}
targets = {k: hub.get_job(v).get_target_model() for k, v in compiled.items()}
ev = load_evalset("imagenette", cache_dir=C, batch_size=16, limit=1024)
imgs, labels = [], []
for x, y in ev.batches():
    imgs += [x[i:i + 1] for i in range(len(x))]; labels += list(y)
labels = np.array(labels)

def preds_of(job):
    data = job.download_output_data()
    return np.concatenate([np.asarray(a).reshape(1, -1) for a in next(iter(data.values()))]).argmax(1)

profs = {k: hub.submit_profile_job(targets[k], device=device, name=f"anneal-efficientnet_b0-{k}-prof-retry2") for k in ("fp32", "hub int8")}
chunks = []
for i in range(0, 1024, 256):
    ds = hub.upload_dataset({"input": imgs[i:i + 256]}, name=f"anneal-imagenette-1024-part{i // 256}")
    chunks.append(hub.submit_inference_job(targets["hub int8 + equalised"], device=device, inputs=ds,
                                           name=f"anneal-efficientnet_b0-hub int8 + equalised-inf-part{i // 256}"))
out = {"jobs": {}}
parts = []
for j in chunks:
    st = j.wait(); out["jobs"][j.name] = {"job": j.job_id, "status": st.code, "message": st.message}
    parts.append(preds_of(j) if st.success else None)
fp32 = preds_of(hub.get_job("jpe76xz05")); int8 = preds_of(hub.get_job("jgzlzym65"))
out["fp32_accuracy"] = float((fp32 == labels).mean()); out["hub_int8_accuracy"] = float((int8 == labels).mean())
if all(p is not None for p in parts):
    eq = np.concatenate(parts); ref = fp32 == labels; right = eq == labels
    b, c = int(np.sum(ref & ~right)), int(np.sum(~ref & right)); d, lo, hi = paired_delta_ci(b, c, len(labels))
    out["hub int8 + equalised"] = {"accuracy": float(right.mean()), "delta_pp": d, "ci95_pp": [lo, hi],
                                   "mcnemar_p": mcnemar_exact(b, c), "agreement": float(np.mean(eq == fp32))}
s24 = {}
for k, jid in {"fp32": "jp8ejr8zp"}.items():
    pass
for k, j in profs.items():
    st = j.wait(); row = {"job": j.job_id, "status": st.code, "message": st.message}
    if st.success:
        prof = j.download_profile()
        row["latency_ms"] = prof["execution_summary"]["estimated_inference_time"] / 1000.0
        row["compute_units"] = dict(Counter(u.get("compute_unit") for u in prof.get("execution_detail", [])))
    out["jobs"][j.name] = row
np.savez("scratch/qaihub/sa8775p_preds.npz", labels=labels, fp32=fp32, int8=int8)
Path("scratch/qaihub/sa8775p_retry2.json").write_text(json.dumps(out, indent=2))
print("RETRY2 DONE"); print(json.dumps(out, indent=1))
