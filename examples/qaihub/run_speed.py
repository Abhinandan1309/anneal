"""Where does equalisation's latency go on the Galaxy S24 NPU, and can an exact rewrite remove it?

Equalised INT8 EfficientNet-B0 takes 0.544 ms on the S24's NPU against 0.420 ms plain (-0.7pp vs
-12.0pp). Leaving sites out does not help (run_site_value.py: accuracy is spread over all 16
sites, the cost sits in the early high-resolution ones). The cost is structural: every site puts
a per-channel Mul(x', 1/s) between the conv and its gate, which also breaks the Conv+SiLU pattern
the NPU fuses. Three exact forms of the same function, y' = s * SiLU(x):

* ``eq``            Conv(sW) -> x' ; y' = x' * sigmoid(x' * (1/s))   (Anneal's default)
* ``eq gate-conv``  the Mul(1/s) as a depthwise 1x1 Conv               (equalise(gate_conv=True))
* ``eq post-scale`` Conv(W) -> x ; y' = SiLU(x) * s                    (the SiLU stays fusable; one
                    Mul by a per-channel constant after it, which an NPU may fold into requantization)

Each is quantized by Qualcomm's own quantizer, compiled for the device, profiled per op, and
scored on Imagenette against FP32 (paired).

    python run_speed.py --images 1024
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
from run_qaihub import static_copy, target_of  # noqa: E402


def post_scale(eq_path: Path, dst: Path) -> int:
    """Rewrite each gated site of an equalised model into Conv(W) -> SiLU -> Mul(s): exact."""
    import onnx
    from onnx import helper, numpy_helper

    from anneal.core.equalize import GATE_MUL_PREFIX

    m = onnx.load(str(eq_path))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    done = 0
    new_nodes = []
    for n in list(g.node):
        if not n.name.startswith(GATE_MUL_PREFIX):
            continue
        x, inv_name = n.input
        conv = prod.get(x)
        users = cons[x]
        if conv is None or conv.op_type != "Conv" or len(users) != 2:
            continue
        sig = next((c for c in cons[n.output[0]] if c.op_type in ("Sigmoid", "HardSigmoid")), None)
        mul = next((c for c in users if c is not n and c.op_type == "Mul"), None)
        if sig is None or mul is None or sig.output[0] not in mul.input:
            continue
        inv = numpy_helper.to_array(inits[inv_name]).reshape(-1).astype(np.float64)
        w = numpy_helper.to_array(inits[conv.input[1]]).astype(np.float64)
        shape = (-1,) + (1,) * (w.ndim - 1)
        inits[conv.input[1]].CopyFrom(numpy_helper.from_array((w * inv.reshape(shape)).astype(np.float32), conv.input[1]))
        if len(conv.input) > 2 and conv.input[2]:
            b = numpy_helper.to_array(inits[conv.input[2]]).astype(np.float64)
            inits[conv.input[2]].CopyFrom(numpy_helper.from_array((b * inv).astype(np.float32), conv.input[2]))
        sig.input[0] = x  # the gate now sees the unscaled x directly: plain SiLU
        g.node.remove(n)
        y = mul.output[0]
        pre = f"{y}_unscaled"
        mul.output[0] = pre
        s_name = f"anneal_post_s_{done}"
        g.initializer.append(numpy_helper.from_array((1.0 / inv).reshape(1, -1, 1, 1).astype(np.float32), s_name))
        new_nodes.append((mul, helper.make_node("Mul", [pre, s_name], [y], name=f"anneal_post_scale_{done}")))
        done += 1
    for after, node in new_nodes:
        g.node.insert(list(g.node).index(after) + 1, node)
    used = {i for nd in g.node for i in nd.input}
    for init in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(init)
    onnx.save(m, str(dst))
    return done


def main() -> None:
    import onnx
    import onnxruntime as ort
    import qai_hub as hub

    from anneal.core.artifact import sample_shape
    from anneal.core.audit import mcnemar_exact, paired_delta_ci
    from anneal.core.dataset import load_calibset, load_evalset
    from anneal.core.equalize import equalise

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    ap.add_argument("--runtime", default="qnn_dlc")
    ap.add_argument("--images", type=int, default=1024)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    src = ROOT / "examples" / "models" / f"{args.model}-fp32.onnx"
    work = ROOT / "scratch" / "qaihub_speed" / args.model
    work.mkdir(parents=True, exist_ok=True)
    shape = sample_shape(src)
    calib = load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64, sample_shape=shape)
    calib_imgs = [x[i:i + 1] for x in calib.calibration_batches(64) for i in range(len(x))][:64]
    batches = [np.concatenate(calib_imgs[i:i + 8]) for i in range(0, 64, 8)]
    eq, gc, ps = work / "eq.onnx", work / "eq-gate-conv.onnx", work / "eq-post-scale.onnx"
    equalise(src, eq, batches)
    equalise(src, gc, batches, gate_conv=True)
    n_ps = post_scale(eq, ps)
    print(f"post-scale rewrote {n_ps} sites", flush=True)
    # exactness in float before spending device time
    x = calib_imgs[0]
    ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    for p in (eq, gc, ps):
        out = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
        print(f"  {p.name}: max |logit diff| vs FP32 {np.abs(out - ref).max():.2e}", flush=True)

    ev = load_evalset("imagenette", cache_dir=CACHE, batch_size=16, limit=args.images, sample_shape=shape)
    eval_imgs, labels = [], []
    for xb, yb in ev.batches():
        eval_imgs += [xb[i:i + 1] for i in range(len(xb))]
        labels += list(yb)
    labels = np.array(labels)

    device = hub.Device(args.device)
    paths = {"fp32": src, "int8": src, "eq": eq, "eq gate-conv": gc, "eq post-scale": ps}
    statics = {k: static_copy(p, work) if k != "fp32" else static_copy(p, work) for k, p in paths.items()}
    q_jobs = {k: hub.submit_quantize_job(str(p), {"input": calib_imgs}, name=f"anneal-speed-{k}-q")
              for k, p in statics.items() if k != "fp32"}
    models = {"fp32": str(statics["fp32"])}
    for k, j in q_jobs.items():
        mdl = target_of(j, k)
        if mdl is not None:
            models[k] = mdl
    specs = {"input": (1, *shape)}
    c_jobs = {k: hub.submit_compile_job(mdl, device=device, input_specs=specs,
                                        options=f"--target_runtime {args.runtime}", name=f"anneal-speed-{k}")
              for k, mdl in models.items()}
    targets = {k: t for k, j in c_jobs.items() if (t := target_of(j, k)) is not None}
    dataset = hub.upload_dataset({"input": eval_imgs}, name=f"anneal-imagenette-{args.images}")
    i_jobs = {k: hub.submit_inference_job(t, device=device, inputs=dataset, name=f"anneal-speed-{k}-inf")
              for k, t in targets.items()}
    p_jobs = {k: hub.submit_profile_job(t, device=device, name=f"anneal-speed-{k}-prof") for k, t in targets.items()}

    result = {"model": args.model, "device": args.device, "runtime": args.runtime, "n": int(len(labels)),
              "post_scale_sites": n_ps, "variants": {}}
    preds = {}
    for k in targets:
        row = {}
        try:
            out = i_jobs[k].download_output_data()
            logits = np.concatenate([np.asarray(a).reshape(1, -1) for a in next(iter(out.values()))])
            preds[k] = logits.argmax(1)
        except Exception as exc:  # noqa: BLE001
            row["inference_error"] = str(exc)[:300]
        try:
            prof = p_jobs[k].download_profile()
            row["latency_ms"] = prof["execution_summary"]["estimated_inference_time"] / 1000.0
            by_type = defaultdict(lambda: [0, 0.0])
            for d in prof.get("execution_detail", []):
                t = d.get("type", "?")
                by_type[t][0] += 1
                by_type[t][1] += float(d.get("execution_time", 0)) / 1000.0
            row["ops"] = {t: {"count": c, "ms": round(ms, 4)} for t, (c, ms) in sorted(by_type.items(), key=lambda kv: -kv[1][1])}
        except Exception as exc:  # noqa: BLE001
            row["profile_error"] = str(exc)[:300]
        result["variants"][k] = row
    ref_ok = preds.get("fp32") == labels if "fp32" in preds else None
    for k, p in preds.items():
        if k == "fp32" or ref_ok is None:
            continue
        right = p == labels
        b, c = int(np.sum(ref_ok & ~right)), int(np.sum(~ref_ok & right))
        d, lo, hi = paired_delta_ci(b, c, len(labels))
        result["variants"][k].update({"delta_pp": d, "ci95_pp": [lo, hi], "mcnemar_p": mcnemar_exact(b, c)})
    for k, row in result["variants"].items():
        ops = row.get("ops", {})
        top = ", ".join(f"{t} {v['count']}x {v['ms']:.3f}" for t, v in list(ops.items())[:6])
        print(f"  {k:16s} {row.get('latency_ms', float('nan')):.3f} ms  "
              f"{row.get('delta_pp', float('nan')):+.2f}pp  | {top}", flush=True)
    out = HERE / "results" / f"{args.model}-speed-{args.device.split('(')[0].strip().lower().replace(' ', '-')}-{args.runtime}-n{len(labels)}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
