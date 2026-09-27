"""Per-layer diff of a real TIDL run against the float model, from `run_tidl_tasks.py --trace`.

TIDL's fixed-point traces (tidl_trace_subgraph_<sg>_<dataId>_..._<C>_<H>x<W>.y) carry no scale,
so each is compared after the least-squares scale (and, for unsigned data, zero point 0): the
SQNR left is what 8-bit TIDL lost on that tensor, independent of its scale. Tensors are named via
each subgraph's layer_info.txt and computed in float by onnxruntime on the same input.

    python trace_diff.py <trace dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    import onnx
    import onnxruntime as ort

    d = Path(sys.argv[1])
    names: dict[tuple[int, int], str] = {}
    for f in d.glob("art_subgraph_*_tidl_net.bin.layer_info.txt"):
        sg = int(re.search(r"subgraph_(\d+)_", f.name).group(1))
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                names[(sg, int(parts[1]))] = parts[2]
    m = onnx.load(str(d / "model_float.onnx"))
    have = {o for n in m.graph.node for o in n.output}
    traces = []
    for f in sorted(d.glob("tidl_trace_subgraph_*.y")):
        g = re.match(r"tidl_trace_subgraph_(\d+)_(\d+)_\d+_\d+_\d+_(\d+)_(\d+)x(\d+)\.y", f.name)
        if not g:
            continue
        sg, did, c, h, w = map(int, g.groups())
        name = names.get((sg, did))
        if name and name in have:
            traces.append((sg, did, name, c, h, w, f))
    wanted = sorted({t[2] for t in traces})
    for nm in wanted:
        m.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    x = np.load(d / "input.npy")
    outs = dict(zip([o.name for o in s.get_outputs()], s.run(None, {s.get_inputs()[0].name: x})))
    order = {o: i for i, o in enumerate(o for n in m.graph.node for o in n.output)}
    rows = []
    for sg, did, name, c, h, w, f in traces:
        ref = np.asarray(outs[name], np.float64).reshape(-1)
        raw = f.read_bytes()
        best = None
        for dt in (np.int8, np.uint8, np.int16, np.uint16):
            q = np.frombuffer(raw, dtype=dt).astype(np.float64)
            if q.size != ref.size:
                continue
            a = float(q @ ref / max(q @ q, 1e-12))
            err = ref - a * q
            sq = 10 * np.log10(max(ref @ ref, 1e-20) / max(err @ err, 1e-20))
            if best is None or sq > best[0]:
                best = (sq, dt.__name__)
        if best is None:
            continue
        # per-channel imbalance of the float tensor: how many channels a per-tensor scale starves
        t = np.asarray(outs[name], np.float64)[0]
        rng = np.abs(t).reshape(t.shape[0], -1).max(1)
        starved = float((rng < rng.max() / 64).mean())
        rows.append((order.get(name, 1e9), sg, name, best[0], best[1], c, h, w, starved))
    rows.sort()
    print(f"{'sg':>2} {'SQNR dB':>8} {'type':6} {'C':>4} {'HxW':>9} {'<1/64 ch':>8}  tensor")
    for _, sg, name, sq, dt, c, h, w, starved in rows:
        flag = "  <==" if sq < 15 else ""
        print(f"{sg:>2} {sq:8.1f} {dt:6} {c:>4} {h:>4}x{w:<4} {starved:8.2f}  {name[-70:]}{flag}")


if __name__ == "__main__":
    main()
