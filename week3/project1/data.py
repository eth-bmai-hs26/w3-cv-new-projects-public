"""
In-memory data pipeline for the tile inspection project.

Each image shows ONE reclaimed tile on the conveyor belt of an inspection
station. Every pixel belongs to one of three classes:

    0  background   the belt
    1  usable       tile surface that can be sold (after cutting, if needed)
    2  damaged      the part of the tile a cutter would remove: cracks and
                    chips plus a small safety margin around them

The decision for a tile follows from its usable share:

    usable fraction = usable pixels / (usable + damaged pixels)
    APPROVE if usable fraction >= USABLE_THRESHOLD, otherwise REJECT

Decoding 10 000 PNG/JPEG files every epoch is the slowest part of training,
so we decode every image and its class map ONCE and keep compact uint8
tensors in RAM (~650 MB for 5000 images at 256x256). Labels are matched to
images by FILENAME and recomputed at the resolution the model sees.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

USABLE_THRESHOLD = 0.85            # usable share of the tile needed to APPROVE
BACKGROUND, USABLE, DAMAGED = 0, 1, 2
CLASS_NAMES = ("background", "usable", "damaged")
N_CLASSES = 3
VALID_EXT = (".jpg", ".jpeg", ".png")


def mask_path(root, img_name, kind="segmented"):
    """'tile_0001.jpg' → root/segmented/tile_0001-segmented.png (or tiles/…-tiles.png)."""
    stem, _ = os.path.splitext(img_name)
    return os.path.join(root, kind, f"{stem}-{kind}.png")


def class_map(tile_mask, usable_mask):
    """Combine the two binary masks into one class map (0 background, 1 usable, 2 damaged)."""
    target = np.zeros(tile_mask.shape, np.uint8)
    target[tile_mask] = DAMAGED
    target[usable_mask & tile_mask] = USABLE
    return target


def usable_fraction(target):
    """Usable share of the tile for class maps (..., H, W); works on numpy arrays and tensors."""
    usable = (target == USABLE).sum(axis=(-2, -1)) if isinstance(target, np.ndarray) \
        else (target == USABLE).sum(dim=(-2, -1)).float()
    tile = (target != BACKGROUND).sum(axis=(-2, -1)) if isinstance(target, np.ndarray) \
        else (target != BACKGROUND).sum(dim=(-2, -1)).float()
    return usable / np.maximum(tile, 1) if isinstance(target, np.ndarray) else usable / tile.clamp(min=1)


def _load(root, fname, size):
    img = Image.open(os.path.join(root, "original", fname)).convert("L")
    tiles = Image.open(mask_path(root, fname, "tiles")).convert("L")
    usable = Image.open(mask_path(root, fname, "segmented")).convert("L")
    # Same ops as T.Resize: bilinear for the photo, NEAREST for masks so they stay binary.
    img = TF.resize(img, [size, size], interpolation=TF.InterpolationMode.BILINEAR)
    tiles = TF.resize(tiles, [size, size], interpolation=TF.InterpolationMode.NEAREST)
    usable = TF.resize(usable, [size, size], interpolation=TF.InterpolationMode.NEAREST)
    return np.asarray(img, dtype=np.uint8), class_map(np.asarray(tiles) > 127, np.asarray(usable) > 127)


@dataclass
class TileCache:
    """All images and class maps decoded into memory."""
    images: torch.Tensor            # (N, 1, H, W) uint8 grayscale photos
    targets: torch.Tensor           # (N, H, W) uint8 class maps (0 background, 1 usable, 2 damaged)
    files: list                     # filenames, sorted
    usable: np.ndarray              # usable fraction at model resolution
    usable_csv: np.ndarray          # usable fraction from ground_truth.csv (full resolution)
    labels: np.ndarray = field(init=False)  # 1 = APPROVE, 0 = REJECT

    def __post_init__(self):
        self.labels = (self.usable >= USABLE_THRESHOLD).astype(np.int64)

    def __len__(self):
        return len(self.files)

    @property
    def size(self):
        return self.images.shape[-1]

    @property
    def label_flips(self) -> int:
        """How many labels differ between the CSV (full resolution) and the model resolution."""
        return int(((self.usable_csv >= USABLE_THRESHOLD) != (self.usable >= USABLE_THRESHOLD)).sum())

    def image(self, idx) -> np.ndarray:
        return self.images[idx, 0].numpy()

    def target(self, idx) -> np.ndarray:
        return self.targets[idx].numpy()


def build_cache(root, size=256, workers=8, cache_path=None, verbose=True) -> TileCache:
    """
    Decode every image and its class map under `root` once and keep them in memory.

    Args:
        root:        folder with original/, tiles/, segmented/ and ground_truth.csv
        size:        side length the model sees (images are square)
        workers:     decoding threads
        cache_path:  optional .pt file; loaded if it exists, written otherwise
    """
    if cache_path and os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        if blob.get("size") == size and "targets" in blob:
            if verbose:
                print(f"Loaded cached dataset from {cache_path}")
            return TileCache(blob["images"], blob["targets"], blob["files"], blob["usable"], blob["usable_csv"])

    files = sorted(
        f for f in os.listdir(os.path.join(root, "original"))
        if not f.startswith(".") and f.lower().endswith(VALID_EXT)
    )
    gt = pd.read_csv(os.path.join(root, "ground_truth.csv")).set_index("filename")
    missing = set(files) - set(gt.index)
    assert not missing, f"{len(missing)} images have no row in ground_truth.csv, e.g. {sorted(missing)[:3]}"

    n = len(files)
    images = torch.empty((n, 1, size, size), dtype=torch.uint8)
    targets = torch.empty((n, size, size), dtype=torch.uint8)

    def work(i):
        img, t = _load(root, files[i], size)
        images[i, 0] = torch.from_numpy(img.copy())
        targets[i] = torch.from_numpy(t)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, _ in enumerate(pool.map(work, range(n))):
            if verbose and (k + 1) % 1000 == 0:
                print(f"  decoded {k + 1}/{n}")

    usable = usable_fraction(targets).numpy()
    usable_csv = gt.loc[files, "usable_fraction"].to_numpy(dtype=np.float64)
    cache = TileCache(images, targets, files, usable, usable_csv)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save({"size": size, "images": images, "targets": targets, "files": files,
                    "usable": usable, "usable_csv": usable_csv}, cache_path)
    if verbose:
        mb = (images.numel() + targets.numel()) / 1e6
        print(f"Cached {n} images at {size}x{size} ({mb:.0f} MB in RAM)")
    return cache


def make_splits(n, fracs=(0.70, 0.15, 0.15), seed=42):
    """Reproducible train/val/test index lists (independent of global RNG state)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def subsample(indices, k, seed=0):
    """First k indices after a seeded shuffle (used by FAST_DEV_RUN)."""
    if k is None or k >= len(indices):
        return list(indices)
    g = torch.Generator().manual_seed(seed)
    return [indices[i] for i in torch.randperm(len(indices), generator=g)[:k].tolist()]


class CachedTileDataset(Dataset):
    """
    Dataset view over a TileCache. Yields (image, target, label):

        image:  float (1, H, W) in [0, 1]
        target: long  (H, W) class map — 0 background, 1 usable, 2 damaged
        label:  int   1 = APPROVE, 0 = REJECT

    With augment=True, random flips / 90° rotations are applied to image and
    target together (they don't change the usable fraction, so labels stay valid).
    """

    def __init__(self, cache: TileCache, indices=None, augment=False):
        self.cache = cache
        self.indices = list(range(len(cache))) if indices is None else list(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.cache.images[idx].float() / 255.0
        target = self.cache.targets[idx].long()
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img, target = img.flip(-1), target.flip(-1)
            if torch.rand(1).item() < 0.5:
                img, target = img.flip(-2), target.flip(-2)
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img, target = torch.rot90(img, k, (-2, -1)), torch.rot90(target, k, (-2, -1))
        return img, target, int(self.cache.labels[idx])
