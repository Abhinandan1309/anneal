"""Make equalisation's gate rescale free: fold 1/s into a per-channel dequantization.

In the QDQ model Qualcomm's quantizer produces for an equalised network, each gate reads

    x'_q --DQ(S)--> Mul(DQ(q(1/s))) --Q(S_u)--> DQ --> Sigmoid

one elementwise Mul over the full feature map (the S24's +29% latency), with 1/s itself rounded
to 8 bits on one scale although it spans orders of magnitude. But x'/s is the same integer tensor
read with a per-channel scale S/s_c:

    x'_q --DQ(scale = S / s_c per channel, axis 1)--> Sigmoid

exact, no Mul, no second rounding. This rewrites Qualcomm's own QDQ model that way and compiles,
profiles and scores both on the device, so the only difference is the gate path.

    python run_perchannel_gate.py --images 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = Path.home() / ".anneal_cache"
sys.path.insert(0, str(HERE))
from run_qaihub import target_of  # noqa: E402


def fold_gate_rescale(qdq: Path, eq_float: Path, dst: Path) -> int:
    """Rewrite every anneal_eq_gate_mul_* path of a QDQ model into a per-channel DQ. Returns count."""
    import onnx
    from onnx import helper, numpy_helper

    from anneal.core.equalize import GATE_MUL_PREFIX

    # exact 1/s from the float equalised model (the QDQ model holds a rounded copy)
    fm = onnx.load(str(eq_float))
    finits = {i.name: i for i in fm.graph.initializer}
    inv_of = {}
    for n in fm.graph.node:
        if n.name.startswith(GATE_MUL_PREFIX):
            inv_of[n.name] = numpy_helper.to_array(finits[n.input[1]]).reshape(-1).astype(np.float64)

    m = onnx.load(str(qdq))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    remove, count = [], 0
    for mul in [n for n in g.node if n.name.startswith(GATE_MUL_PREFIX)]:
        if mul.name not in inv_of:
            continue
        dq_x = prod.get(mul.input[0])
        if dq_x is None or dq_x.op_type != "DequantizeLinear":
            continue
        q_u = [c for c in cons[mul.output[0]] if c.op_type == "QuantizeLinear"]
        if len(q_u) != 1 or len(cons[mul.output[0]]) != 1:
            continue
        dq_u = cons[q_u[0].output[0]]
        if len(dq_u) != 1 or dq_u[0].op_type != "DequantizeLinear":
            continue
        gate_inputs = cons[dq_u[0].output[0]]
        s_x = float(numpy_helper.to_array(inits[dq_x.input[1]]).reshape(-1)[0])
        zp_x = numpy_helper.to_array(inits[dq_x.input[2]]).reshape(-1)[0] if len(dq_x.input) > 2 else None
        inv = inv_of[mul.name]
        base = f"anneal_pcgate_{count}"
        scale = numpy_helper.from_array((s_x * inv).astype(np.float32), f"{base}_scale")
        g.initializer.append(scale)
        ins = [dq_x.input[0], scale.name]
        if zp_x is not None:
            g.initializer.append(numpy_helper.from_array(np.full(inv.shape, zp_x, dtype=zp_x.dtype), f"{base}_zp"))
            ins.append(f"{base}_zp")
        new_dq = helper.make_node("DequantizeLinear", ins, [f"{base}_out"], name=f"{base}_dq", axis=1)
        for gi in gate_inputs:
            for k, name in enumerate(gi.input):
                if name == dq_u[0].output[0]:
                    gi.input[k] = new_dq.output[0]
        g.node.insert(list(g.node).index(dq_u[0]), new_dq)
        remove += [mul, q_u[0], dq_u[0]]
        dq_inv = prod.get(mul.input[1])
        if dq_inv is not None and len(cons[dq_inv.output[0]]) == 1:
            remove.append(dq_inv)
            q_inv = prod.get(dq_inv.input[0])
            if q_inv is not None and len(cons[q_inv.output[0]]) == 1:
                remove.append(q_inv)
        count += 1
    for n in remove:
        g.node.remove(n)
    used = {i for n in g.node for i in n.input}
    for init in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(init)
    onnx.save(m, str(dst))
    return count


def main() -> None:
    import onnxruntime as ort
    import qai_hub as hub

    from anneal.core.artifact import sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_evalset

    ap = argparse.ArgumentParser()
    ap.add_argument("--qdq", required=True, help="Qualcomm's QDQ model of the equalised network")
    ap.add_argument("--eq", required=True, help="the float equalised model it was quantized from")
    ap.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    ap.add_argument("--images", type=int, default=1024)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    qdq, eq = Path(args.qdq), Path(args.eq)
    work = qdq.parent
    folded = work / "eq-qdq-pcgate.onnx"
    n = fold_gate_rescale(qdq, eq, folded)
    print(f"folded {n} gate rescales into per-channel dequantization", flush=True)

    src = ROOT / "examples" / "models" / "efficientnet_b0-fp32.onnx"
    shape = sample_shape(src)
    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=16, limit=args.images, sample_shape=shape)
    imgs, labels = [], []
    for xb, yb in ev.batches():
        imgs += [xb[i:i + 1] for i in range(len(xb))]
        labels += list(yb)
    labels = np.array(labels)
    # onnxruntime check of the rewrite before spending device time
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    a = ort.InferenceSession(str(qdq), so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(folded), so, providers=["CPUExecutionProvider"])
    agree = np.mean([a.run(None, {"input": x})[0].argmax() == b.run(None, {"input": x})[0].argmax() for x in imgs[:64]])
    print(f"onnxruntime: top-1 agreement QDQ vs folded on 64 images {agree:.3f}", flush=True)

    device = hub.Device(args.device)
    fp32 = ROOT / "scratch" / "qaihub_speed" / "efficientnet_b0" / "efficientnet_b0-fp32-b1.onnx"
    models = {"fp32": str(fp32), "eq (hub qdq)": str(qdq), "eq per-channel gate": str(folded)}
    specs = {"input": (1, *shape)}
    c_jobs = {k: hub.submit_compile_job(p, device=device, input_specs=specs, options="--target_runtime qnn_dlc",
                                        name=f"anneal-pcgate-{k}") for k, p in models.items()}
    targets = {k: t for k, j in c_jobs.items() if (t := target_of(j, k)) is not None}
    dataset = hub.upload_dataset({"input": imgs}, name=f"anneal-imagenette-{args.images}")
    i_jobs = {k: hub.submit_inference_job(t, device=device, inputs=dataset, name=f"anneal-pcgate-{k}-inf") for k, t in targets.items()}
    p_jobs = {k: hub.submit_profile_job(t, device=device, name=f"anneal-pcgate-{k}-prof") for k, t in targets.items()}
    result, preds = {"device": args.device, "n": int(len(labels)), "folded": n, "variants": {}}, {}
    for k in targets:
        row = {}
        try:
            out = i_jobs[k].download_output_data()
            preds[k] = np.concatenate([np.asarray(v).reshape(1, -1) for v in next(iter(out.values()))]).argmax(1)
        except Exception as exc:  # noqa: BLE001
            row["inference_error"] = str(exc)[:300]
        try:
            prof = p_jobs[k].download_profile()
            row["latency_ms"] = prof["execution_summary"]["estimated_inference_time"] / 1000.0
            units = defaultdict(int)
            for d in prof.get("execution_detail", []):
                units[d.get("compute_unit", "?")] += 1
            row["compute_units"] = dict(units)
        except Exception as exc:  # noqa: BLE001
            row["profile_error"] = str(exc)[:300]
        result["variants"][k] = row
    for k in list(targets) + [k for k in models if k not in targets]:
        if k not in targets:
            result["variants"][k] = {"compile_failed": True}
    if "fp32" in preds:
        ref = preds["fp32"] == labels
        for k, p in preds.items():
            if k == "fp32":
                continue
            r = p == labels
            b_, c_ = int(np.sum(ref & ~r)), int(np.sum(~ref & r))
            d, lo, hi = paired_delta_ci(b_, c_, len(labels))
            result["variants"][k].update({"delta_pp": d, "ci95_pp": [lo, hi], "mcnemar_p": mcnemar_exact(b_, c_)})
    for k, row in result["variants"].items():
        print(f"  {k:22s} {row.get('latency_ms', float('nan')):.3f} ms  {row.get('delta_pp', float('nan')):+.2f}pp  "
              f"{row.get('compute_units', row)}", flush=True)
    (HERE / "results" / f"efficientnet_b0-pcgate-s24-n{len(labels)}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
