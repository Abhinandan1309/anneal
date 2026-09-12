"""Evaluation data.

Accuracy claims are only worth anything if they come from real images. The default
eval set is Imagenette (a 10-class subset of ImageNet, ~100MB) scored with a *full
1000-way* argmax — no restricting the softmax to the ten classes present, which would
inflate top-1 by several points and quietly make every quantization result look safer
than it is.

A synthetic fallback exists so the test suite and a laptop on a plane still work, but it
is explicitly labelled and never used to justify an accuracy number in the README.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from anneal.core.measure import EvalSet

IMAGENETTE_URLS = {
    "160": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    "320": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
}

#: Imagenette's ten WordNet synsets and their index in the full ImageNet-1k label space.
IMAGENETTE_SYNSET_TO_IMAGENET_IDX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def download_imagenette(cache_dir: Path, variant: str = "160") -> Path:
    """Download and extract Imagenette once; return the extracted root."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = cache_dir / f"imagenette2-{variant}"
    if (root / "val").is_dir():
        return root

    url = IMAGENETTE_URLS[variant]
    archive = cache_dir / f"imagenette2-{variant}.tgz"
    if not archive.exists():
        tmp = archive.with_suffix(".tgz.part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(archive)

    with tarfile.open(archive, "r:gz") as tf:
        # Reject absolute paths and traversal before extracting a downloaded archive.
        members = []
        for m in tf.getmembers():
            name = Path(m.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"refusing unsafe archive member: {m.name}")
            members.append(m)
        tf.extractall(cache_dir, members=members)

    if not (root / "val").is_dir():
        raise RuntimeError(f"extracted archive but {root / 'val'} is missing")
    return root


def preprocess_image(path: Path, image_size: int = 224, resize: int = 256) -> np.ndarray:
    """Standard ImageNet eval preprocessing: resize shortest side, centre crop, normalise."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = resize / min(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        w, h = im.size
        left = (w - image_size) // 2
        top = (h - image_size) // 2
        im = im.crop((left, top, left + image_size, top + image_size))
        arr = np.asarray(im, dtype=np.float32) / 255.0

    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW


class ImagenetteEvalSet(EvalSet):
    """Imagenette validation images, scored against the full ImageNet-1k label space."""

    name = "imagenette"
    synthetic = False

    #: Above this, decoded tensors are streamed rather than held in RAM. A validation
    #: pass over the full ImageNet-derived val set would otherwise want several GB, and
    #: swapping makes every latency number in the run meaningless.
    CACHE_BUDGET_BYTES = 768 * 1024 * 1024

    def __init__(
        self,
        root: Path,
        *,
        split: str = "val",
        batch_size: int = 16,
        limit: int | None = None,
        image_size: int = 224,
        resize: int = 256,
        seed: int = 0,
        cache: bool | None = None,
    ) -> None:
        self.root = Path(root)
        self.batch_size = batch_size
        self.image_size = image_size
        self.resize = resize
        self._cache_requested = cache

        items: list[tuple[Path, int]] = []
        split_dir = self.root / split
        for synset, label in sorted(IMAGENETTE_SYNSET_TO_IMAGENET_IDX.items()):
            class_dir = split_dir / synset
            if not class_dir.is_dir():
                continue
            for img in sorted(class_dir.glob("*.JPEG")):
                items.append((img, label))

        if not items:
            raise FileNotFoundError(f"no images found under {split_dir}")

        # Shuffle deterministically so a limited subset stays class-balanced.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(items))
        items = [items[i] for i in order]
        if limit is not None:
            items = items[:limit]

        self.items: Sequence[tuple[Path, int]] = items
        self._cache: list[tuple[np.ndarray, np.ndarray]] | None = None

        per_image = 3 * image_size * image_size * 4
        self.caching = (
            cache
            if cache is not None
            else (len(items) * per_image) <= self.CACHE_BUDGET_BYTES
        )
        self.estimated_bytes = len(items) * per_image

    def __len__(self) -> int:
        return len(self.items)

    def _decode_batch(self, chunk: Sequence[tuple[Path, int]]) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack(
            [preprocess_image(p, self.image_size, self.resize) for p, _ in chunk]
        ).astype(np.float32)
        y = np.array([label for _, label in chunk], dtype=np.int64)
        return x, y

    def _iter_batches(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Decode on the fly, holding one batch at a time."""
        for start in range(0, len(self.items), self.batch_size):
            yield self._decode_batch(self.items[start : start + self.batch_size])

    def _materialise(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Decode once and keep it, so every trial sees byte-identical inputs.

        Re-decoding per candidate costs wall time, but the decode is deterministic, so
        streaming gives the same pixels — only slower. Caching is the default because a
        search runs the eval set ~20 times; it is abandoned above ``CACHE_BUDGET_BYTES``
        because swapping would corrupt the very latency numbers this tool exists to
        measure.
        """
        if self._cache is None:
            self._cache = list(self._iter_batches())
        return self._cache

    def batches(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.caching:
            yield from self._materialise()
        else:
            yield from self._iter_batches()

    def calibration_batches(self, limit: int) -> Iterator[np.ndarray]:
        seen = 0
        for x, _ in self.batches():
            if seen >= limit:
                return
            yield x
            seen += x.shape[0]


class SyntheticEvalSet(EvalSet):
    """Deterministic noise with arbitrary labels. For plumbing tests only.

    Accuracy from this set is meaningless by construction; it exists so the pipeline can
    be exercised without a dataset download. Anything that consumes an EvalSet should
    check ``.synthetic`` before publishing an accuracy number.
    """

    name = "synthetic"
    synthetic = True

    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (3, 224, 224),
        n: int = 32,
        batch_size: int = 8,
        n_classes: int = 1000,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.shape = shape
        self.batch_size = batch_size
        self._batches: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, n, batch_size):
            bs = min(batch_size, n - start)
            x = rng.standard_normal((bs, *shape), dtype=np.float32)
            y = rng.integers(0, n_classes, size=bs).astype(np.int64)
            self._batches.append((x, y))
        self._n = n

    def __len__(self) -> int:
        return self._n

    def batches(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from self._batches

    def calibration_batches(self, limit: int) -> Iterator[np.ndarray]:
        seen = 0
        for x, _ in self._batches:
            if seen >= limit:
                return
            yield x
            seen += x.shape[0]


def load_evalset(
    spec: str,
    *,
    cache_dir: Path,
    batch_size: int = 16,
    limit: int | None = None,
) -> EvalSet:
    """Resolve an eval-set spec string to a concrete EvalSet."""
    if spec == "synthetic":
        return SyntheticEvalSet(batch_size=batch_size, n=limit or 32)
    if spec.startswith("imagenette"):
        variant = spec.split(":", 1)[1] if ":" in spec else "160"
        root = download_imagenette(Path(cache_dir), variant)
        return ImagenetteEvalSet(root, batch_size=batch_size, limit=limit)
    path = Path(spec)
    if path.is_dir():
        return ImagenetteEvalSet(path, batch_size=batch_size, limit=limit)
    raise ValueError(
        f"unrecognised eval set {spec!r}; expected 'synthetic', 'imagenette[:160|:320]', "
        f"or a path to an Imagenette-layout directory"
    )
