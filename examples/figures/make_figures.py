"""Figures for the write-up, from committed results (and one local measurement for figure 1).

    python examples/figures/make_figures.py            # figures 1-4 -> docs/figures/
    python examples/figures/make_figures.py --masks tidl-tasks-result-masks.npz   # + figure 5

1. channel_ranges.png   why per-tensor INT8 breaks: per-channel ranges of one EfficientNet-B0 tensor,
                        before and after equalisation (64 Imagenette calibration images)
2. method.png           the exact rewrite through the gate
3. grid.png             8 models x 4 toolchains: vendor default INT8 vs Anneal
4. speed_s24.png        latency and accuracy on the Galaxy S24 NPU
5. segmentation.png     LRASPP masks on TI TDA4VM: FP32, TIDL 8-bit, TIDL + Anneal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
CACHE = Path.home() / ".anneal_cache"
INK, MUTED, BAD, GOOD, ACCENT = "#1f2328", "#6e7781", "#cf222e", "#1a7f37", "#0969da"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})

# Top-1 change vs FP32, pp (docs/benchmark_grid.md; every number is in a committed result JSON)
MODELS = ["EfficientNet-B0", "EfficientNet-B1", "MobileNetV3-S", "MobileNetV3-L", "MobileNetV2",
          "LCNet-100", "MobileViT-S", "ResNet-50"]
TARGETS = ["AMD XINT8\n(emulated)", "Galaxy S24 NPU\n(device)", "TI TDA4VM\n(emulated)",
           "NVIDIA T4\n(device)"]
DEFAULT = np.array([[-75.1, -12.8, -73.0, -52.5], [-75.8, -75.2, -76.7, -75.7],
                    [-64.7, -58.0, -65.8, -64.1], [-44.8, -2.4, -14.8, -9.9],
                    [-7.9, -0.4, -10.7, -0.8], [-67.0, -34.4, -69.3, -64.0],
                    [-76.7, -17.3, -71.7, -57.4], [-4.6, 1.0, -1.9, -1.1]])
ANNEAL = np.array([[-0.1, -0.7, -1.6, -0.1], [-4.1, -1.7, -8.9, -3.5],
                   [-1.7, -1.4, -2.1, -1.8], [-1.0, -1.2, 0.0, -2.2],
                   [-0.8, -0.4, -1.2, -0.8], [-2.9, -1.8, -3.7, -1.2],
                   [-0.9, -0.3, -4.2, 0.6], [-0.2, 1.0, -1.8, -0.5]])


def channel_ranges() -> None:
    """Per-channel max |x| of the depthwise input at EfficientNet-B0's most imbalanced gated site."""
    import onnx
    import onnxruntime as ort

    from anneal.core.dataset import load_calibset
    from anneal.core.equalize import equalise

    src = ROOT / "examples" / "models" / "efficientnet_b0-fp32.onnx"
    work = ROOT / "scratch" / "figures"
    work.mkdir(parents=True, exist_ok=True)
    batches = list(load_calibset("imagenette", cache_dir=CACHE, batch_size=8, limit=64).calibration_batches(64))
    eq = work / "b0-eq.onnx"
    res = equalise(src, eq, batches)

    def ranges(path: Path, consumers: list[str]) -> dict[str, np.ndarray]:
        m = onnx.load(str(path))
        ins = {n.name: n.input[0] for n in m.graph.node if n.name in consumers}
        known = {o.name for o in m.graph.output}
        m.graph.output.extend([onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
                               for t in ins.values() if t not in known])
        tmp = work / f"{path.stem}-probe.onnx"
        onnx.save(m, str(tmp))
        s = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        names = [o.name for o in s.get_outputs()]
        acc = {}
        for x in batches:
            outs = dict(zip(names, s.run(None, {s.get_inputs()[0].name: x})))
            for c, t in ins.items():
                r = np.abs(outs[t]).max(axis=(0, 2, 3))
                acc[c] = np.maximum(acc.get(c, 0), r)
        return acc

    consumers = [site.consumer for site in res.sites]
    before, after = ranges(src, consumers), ranges(eq, consumers)

    def starved(r: np.ndarray) -> int:  # channels given less than 1 of int8's 127 levels by one shared scale
        return int(np.sum(127 * r / r.max() < 1))

    site = max(consumers, key=lambda c: starved(before[c]))
    b, a = np.sort(before[site])[::-1], np.sort(after[site])[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, r, title, colour in ((axes[0], b, "Standard INT8 input", BAD),
                                 (axes[1], a, "After Anneal's equalisation", GOOD)):
        levels = 127 * r / r.max()
        ax.bar(np.arange(len(levels)), levels, width=1.0, color=colour)
        ax.set_yscale("log")
        ax.set_ylim(0.2, 200)
        ax.axhline(1, color=INK, lw=0.8, ls="--")
        ax.text(len(levels) * 0.98, 1.25, "1 level", ha="right", va="bottom", color=INK, fontsize=8)
        ax.set_title(f"{title}\n{starved(r)} of {len(r)} channels get less than 1 level", fontsize=10)
        ax.set_xlabel("channel (sorted by range)")
    axes[0].set_ylabel("INT8 levels available\n(127 × channel range / tensor range)")
    fig.suptitle(f"EfficientNet-B0: one INT8 scale shared by {len(b)} channels "
                 f"(input of {site.split('/')[2]}'s depthwise conv)", fontsize=10.5, y=1.08)
    fig.savefig(OUT / "channel_ranges.png")
    plt.close(fig)
    print(f"channel_ranges.png: site {site}, starved {starved(b)} -> {starved(a)} of {len(b)}")


def method() -> None:
    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)

    def box(x, y, w, text, fc="#f6f8fa", ec=MUTED, color=INK):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, 0.62, boxstyle="round,pad=0.04",
                                                       fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + 0.31, text, ha="center", va="center", fontsize=9.5, color=color)

    def arrow(x0, x1, y):
        ax.annotate("", (x1, y), (x0, y), arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})

    y1, y0 = 2.45, 0.75
    ax.text(0, 3.4, "Before", fontsize=10.5, weight="bold", color=INK, va="top")
    box(0.2, y1, 1.9, "conv A")
    arrow(2.15, 2.75, y1 + 0.31)
    box(2.8, y1, 2.7, "y = x · sigmoid(x)")
    arrow(5.55, 6.15, y1 + 0.31)
    box(6.2, y1, 2.1, "depthwise B")
    ax.text(8.5, y1 + 0.31, "y: channel ranges\ndiffer up to 360×", fontsize=8.5, color=BAD, va="center")

    ax.text(0, 1.7, "After (exact)", fontsize=10.5, weight="bold", color=INK, va="top")
    box(0.2, y0, 1.9, "conv A · s", fc="#dafbe1", ec=GOOD)
    arrow(2.15, 2.75, y0 + 0.31)
    box(2.8, y0, 2.7, "y' = x' · sigmoid(x' / s)", fc="#dafbe1", ec=GOOD)
    arrow(5.55, 6.15, y0 + 0.31)
    box(6.2, y0, 2.1, "depthwise B / s", fc="#dafbe1", ec=GOOD)
    ax.text(8.5, y0 + 0.31, "y' = s · y:\nranges balanced", fontsize=8.5, color=GOOD, va="center")
    ax.text(0.2, 0.18, "x' = s·x per channel. The gate sees x'/s = x, so y' = s·y exactly, and B divides s "
            "back out: the float model's output is unchanged. No retraining.", fontsize=8.5, color=MUTED)
    fig.savefig(OUT / "method.png")
    plt.close(fig)
    print("method.png")


def workflow() -> None:
    """Where Anneal sits: between the trained float model and the vendor's toolchain."""
    fig, ax = plt.subplots(figsize=(10, 4.1))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)

    def box(x, y, w, h, title, body="", fc="#f6f8fa", ec=MUTED):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                                       fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.5, weight="bold", color=INK)
        if body:
            ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8, color=INK, linespacing=1.4)

    def arrow(x0, y0, x1, y1, colour=MUTED):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "->", "color": colour, "lw": 1.2})

    box(0.05, 1.85, 1.55, 1.5, "Trained model", "FP32\nPyTorch / timm\n→ ONNX")
    box(2.05, 1.15, 2.6, 2.9, "Anneal", "1. equalise: exact rewrite\n   (same float output)\n"
        "2. recipe for the target:\n   calibration, 16-bit layers\n3. or quantize itself (QDQ)\n"
        "4. measure vs FP32:\n   paired, with CIs", fc="#dafbe1", ec=GOOD)
    box(5.35, 2.7, 2.2, 1.55, "Vendor quantizer", "Qualcomm AI Hub\nTI TIDL, AMD Quark\nNVIDIA ModelOpt")
    box(5.35, 0.75, 2.2, 1.55, "Vendor compiler", "Qualcomm QNN\nNVIDIA TensorRT")
    box(8.05, 1.75, 1.9, 1.55, "Target", "S24 NPU, T4 GPU\n(TDA4VM, Ryzen AI:\nvendor emulators)")
    arrow(1.62, 2.6, 2.03, 2.6)
    arrow(4.67, 3.35, 5.33, 3.45, GOOD)
    ax.text(5.0, 3.52, "float model", ha="center", va="bottom", fontsize=7.5, color=GOOD)
    arrow(4.67, 1.8, 5.33, 1.55, GOOD)
    ax.text(5.0, 1.38, "INT8 model", ha="center", va="top", fontsize=7.5, color=GOOD)
    arrow(7.57, 3.45, 8.03, 2.75)
    arrow(7.57, 1.55, 8.03, 2.3)
    # feedback: target -> below everything -> Anneal
    ax.plot([9.0, 9.0, 3.35], [1.73, 0.3, 0.3], color=ACCENT, lw=1, ls="--")
    ax.annotate("", (3.35, 1.13), (3.35, 0.3), arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1,
                                                            "linestyle": "--"})
    ax.text(6.2, 0.38, "accuracy and latency measured on the target, fed back", ha="center", va="bottom",
            fontsize=7.5, color=ACCENT)
    ax.text(2.05, 4.35, "No retraining. No change to the vendor's tools.", fontsize=8.5, color=MUTED)
    fig.savefig(OUT / "workflow.png")
    plt.close(fig)
    print("workflow.png")


def grid() -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("loss", [BAD, "#fff8c5", "#dafbe1", GOOD], N=256)
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-80, vcenter=-5, vmax=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, data, title in ((axes[0], DEFAULT, "Vendor's default INT8"), (axes[1], ANNEAL, "With Anneal")):
        ax.imshow(np.clip(data, -80, 1.5), cmap=cmap, norm=norm, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                ax.text(j, i, "0.0" if v == 0 else f"{v:+.1f}".replace("-", "−"), ha="center", va="center",
                        fontsize=9, color="white" if v < -50 else INK, weight="bold" if data is ANNEAL and v >= -2 else None)
        ax.set_xticks(range(4), TARGETS, fontsize=8)
        ax.set_yticks(range(len(MODELS)), MODELS)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=11, pad=8)
        ax.xaxis.tick_top()
    fig.suptitle("Top-1 accuracy change vs FP32, percentage points (Imagenette, 1,000–1,500 images)",
                 fontsize=10, y=0.02, color=MUTED)
    fig.savefig(OUT / "grid.png")
    plt.close(fig)
    print("grid.png")


def speed_s24() -> None:
    # Galaxy S24, QNN, on-device profiler; accuracy vs true FP32 (onnxruntime CPU), 1,024 images
    rows = {"EfficientNet-B0": [(0.838, -0.8), (0.422, -12.8), (0.538, -0.7)],
            "EfficientNet-B1": [(1.196, -6.8), (0.559, -75.2), (0.893, -1.7)],
            "MobileNetV3-S": [(0.291, -0.1), (0.201, -58.0), (0.260, -1.4)]}
    labels = ["FP16", "Qualcomm INT8", "INT8 + Anneal"]
    colours = [MUTED, BAD, GOOD]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.26
    for i, (model, vals) in enumerate(rows.items()):
        for k, (ms, d) in enumerate(vals):
            x = i + (k - 1) * w
            ax.bar(x, ms, w * 0.92, color=colours[k], label=labels[k] if i == 0 else None)
            ax.text(x, ms + 0.02, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", fontsize=8.5,
                    color=colours[k], weight="bold")
    ax.set_xticks(range(len(rows)), list(rows))
    ax.set_ylabel("latency per image, ms")
    ax.set_ylim(0, 1.45)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    ax.set_title("Galaxy S24 NPU: latency (bars) and top-1 change vs FP32 (labels, pp)", fontsize=10)
    fig.savefig(OUT / "speed_s24.png")
    plt.close(fig)
    print("speed_s24.png")


# VOC palette (torchvision's LRASPP classes)
def _palette() -> np.ndarray:
    p = np.zeros((256, 3), np.uint8)
    for i in range(256):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        p[i] = (r, g, b)
    p[255] = (255, 255, 255)
    return p


def segmentation(mask_file: Path, ids: list[int], scores: dict[str, float]) -> None:
    sys.path.insert(0, str(ROOT / "examples" / "tasks"))
    import run_tasks as rt

    z = np.load(mask_file)
    api = rt.coco()
    pal = _palette()
    cols = [("image", None), ("FP32", "fp32"), ("TIDL 8-bit", "tidl 8-bit"),
            ("TIDL 8-bit + Anneal", "tidl 8-bit + equalised + 16-bit backbone 0-1")]
    fig, axes = plt.subplots(len(ids), 4, figsize=(9.2, 2.35 * len(ids)), constrained_layout=True)
    credits = []
    for r, img_id in enumerate(ids):
        img = rt.pil_resize(rt.load_rgb(api, img_id), (512, 512))
        info = api.loadImgs(img_id)[0]
        credits.append(f"COCO val2017 {img_id}: {info['flickr_url']}, CC BY 2.0 (https://creativecommons.org/licenses/by/2.0/); resized to 512x512, masks overlaid")
        for c, (title, key) in enumerate(cols):
            ax = axes[r, c]
            ax.set_axis_off()
            if key is None:
                ax.imshow(img)
            else:
                m = z[f"{key}|{img_id}"]
                ax.imshow((0.45 * img + 0.55 * pal[m]).astype(np.uint8))
            if r == 0:
                sub = f"\n{scores[key]:.1f} mIoU" if key in scores else ""
                ax.set_title(title + sub, fontsize=9.5)
    fig.suptitle("LRASPP-MobileNetV3 on TI TDA4VM (TIDL emulation), COCO val2017; mIoU over 300 images", fontsize=10.5)
    fig.savefig(OUT / "segmentation.png")
    plt.close(fig)
    (OUT / "segmentation_credits.txt").write_text("\n".join(credits) + "\n", encoding="utf-8")
    print("segmentation.png")


def segmentation_linkedin(mask_file: Path, scores: dict[str, float], ids=(249643, 283412)) -> None:
    """Two rows, three columns, large type: readable on a phone."""
    sys.path.insert(0, str(ROOT / "examples" / "tasks"))
    import run_tasks as rt

    z = np.load(mask_file)
    api = rt.coco()
    pal = _palette()
    cols = [("Full precision", "fp32", INK), ("TI's 8-bit", "tidl 8-bit", BAD),
            ("8-bit + Anneal", "tidl 8-bit + equalised + 16-bit backbone 0-1", GOOD)]
    fig, axes = plt.subplots(len(ids), 3, figsize=(12, 8.9), constrained_layout=True)
    for r, img_id in enumerate(ids):
        img = rt.pil_resize(rt.load_rgb(api, img_id), (512, 512))
        for c, (title, key, colour) in enumerate(cols):
            ax = axes[r, c]
            ax.set_axis_off()
            ax.imshow((0.45 * img + 0.55 * pal[z[f"{key}|{img_id}"]]).astype(np.uint8))
            if r == 0:
                ax.set_title(f"{title}\n{scores[key]:.1f} mIoU", fontsize=19, color=colour, weight="bold")
    fig.suptitle("Segmentation on TI's TDA4VM car chip: same model, INT8", fontsize=21, color=INK)
    fig.text(0.5, -0.03, "LRASPP-MobileNetV3, TIDL emulation; mIoU over 300 COCO images. "
             "Photos: Flickr, CC BY 2.0, resized, masks overlaid.", ha="center", fontsize=11, color=MUTED)
    fig.savefig(OUT / "segmentation_linkedin.png", dpi=150)
    plt.close(fig)
    print("segmentation_linkedin.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", type=Path, help="tidl-tasks-result-masks.npz from the tidl-tasks workflow")
    ap.add_argument("--mask-result", type=Path, help="the matching tidl-tasks-result.json (mIoU per variant)")
    ap.add_argument("--ids", default="110449,143998,249643,283412")
    ap.add_argument("--only", default="", help="comma-separated figure names")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    for name, fn in (("channel_ranges", channel_ranges), ("method", method), ("workflow", workflow), ("grid", grid), ("speed", speed_s24)):
        if not only or name in only:
            fn()
    if args.masks:
        import json

        scores = {}
        if args.mask_result:
            res = json.loads(args.mask_result.read_text(encoding="utf-8"))
            scores["fp32"] = 100 * res["fp32"]
            for k, v in res["variants"].items():
                if "metric" in v:
                    scores[k] = 100 * v["metric"]
        segmentation(args.masks, [int(i) for i in args.ids.split(",")], scores)
        segmentation_linkedin(args.masks, scores)


if __name__ == "__main__":
    main()
