"""Retry the SA8775P jobs that failed with 'failed after compiling' (FP32 included), once, on the same compiled models."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import qai_hub as hub

compiled = {"fp32": "jgoldwrdg", "hub int8": "jpvl2mdm5", "hub int8 + equalised": "jgjr3y78p"}
ok_inf = hub.get_job("jgzlzym65")
dataset = ok_inf.inputs
device = hub.Device("SA8775P ADP")
targets = {k: hub.get_job(v).get_target_model() for k, v in compiled.items()}
inf = hub.submit_inference_job(targets["hub int8 + equalised"], device=device, inputs=dataset,
                               name="anneal-efficientnet_b0-hub int8 + equalised-inf-retry")
profs = {k: hub.submit_profile_job(t, device=device, name=f"anneal-efficientnet_b0-{k}-prof-retry") for k, t in targets.items()}
out = {"retry_of": "examples/qaihub/results/efficientnet_b0-sa8775p-adp-tflite-n1024.json", "jobs": {}}
st = inf.wait()
out["jobs"]["hub int8 + equalised inference"] = {"job": inf.job_id, "status": st.code, "message": st.message}
if st.success:
    data = inf.download_output_data()
    logits = np.concatenate([np.asarray(a).reshape(1, -1) for a in next(iter(data.values()))])
    np.save("scratch/qaihub/sa8775p_eq_preds.npy", logits.argmax(1))
for k, j in profs.items():
    st = j.wait()
    row = {"job": j.job_id, "status": st.code, "message": st.message}
    if st.success:
        prof = j.download_profile()
        row["latency_ms"] = prof["execution_summary"]["estimated_inference_time"] / 1000.0
        row["compute_units"] = dict(Counter(u.get("compute_unit") for u in prof.get("execution_detail", [])))
    out["jobs"][f"{k} profile"] = row
Path("scratch/qaihub/sa8775p_retry.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
