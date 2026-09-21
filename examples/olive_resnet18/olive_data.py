"""Olive user script: Imagenette data for calibrating and evaluating ResNet-18.

Uses Anneal's own preprocessing and image ordering, so the images Olive evaluates on are
byte-identical to the ones Anneal's search used. Calibration draws from the *train* split
so it never touches evaluation images.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from olive.data.registry import Registry

# Reuse Anneal's preprocessing without installing Anneal into Olive's environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anneal.core.dataset import (  # noqa: E402
    IMAGENETTE_SYNSET_TO_IMAGENET_IDX,
    ImagenetteEvalSet,
    preprocess_image,
)

ROOT = Path.home() / ".anneal_cache" / "imagenette2-160"


class _Images:
    def __init__(self, items):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        # Olive's calibration reader only accepts dict inputs, keyed by graph input name.
        return {"input": preprocess_image(path).astype(np.float32)}, int(label)


@Registry.register_dataset()
def imagenette_eval(limit: int = 256, **kwargs):
    """The same deterministic shuffle and limit as `anneal run --eval-limit`."""
    return _Images(ImagenetteEvalSet(ROOT, split="val", limit=limit).items)


@Registry.register_dataset()
def imagenette_calib(limit: int = 64, seed: int = 0, **kwargs):
    """Calibration images from the train split, balanced across the ten classes."""
    rng = np.random.default_rng(seed)
    items = []
    for synset, label in sorted(IMAGENETTE_SYNSET_TO_IMAGENET_IDX.items()):
        files = sorted((ROOT / "train" / synset).glob("*.JPEG"))
        items += [(files[i], label) for i in rng.permutation(len(files))]
    items = [items[i] for i in rng.permutation(len(items))][:limit]
    return _Images(items)


@Registry.register_post_process()
def top1(output, **kwargs):
    """Model logits -> predicted ImageNet class index."""
    logits = output.logits if hasattr(output, "logits") else output
    if hasattr(logits, "argmax"):
        return logits.argmax(axis=-1)
    return np.asarray(logits).argmax(axis=-1)
