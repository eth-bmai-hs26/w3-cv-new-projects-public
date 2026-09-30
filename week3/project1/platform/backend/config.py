"""
Paths and defaults. Everything can be overridden with environment variables so
the same code runs on a staff laptop, a student laptop and in the test suite.

    TILE_PLATFORM_DATA        folder for the SQLite DB + stored photos/overlays (default platform/var)
    TILE_PLATFORM_CHECKPOINT  model file to load at start-up (default platform/models/default_model.pt)
    TILE_PLATFORM_IMAGES      folder of photos the seed script inspects
    TILE_PLATFORM_DEVICE      auto | cpu | cuda | mps
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent.parent          # .../week3/project1/platform
REPO_ROOT = PLATFORM_DIR.parent.parent.parent                   # repo root (contains week3/) when run from the repo
FRONTEND_DIST = PLATFORM_DIR / "frontend" / "dist"
MODELS_DIR = PLATFORM_DIR / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "default_model.pt"
UPLOADED_MODELS_DIR = MODELS_DIR / "uploads"

# The course's single-tile dataset (week3/project1/dataset_new). The first
# folder that exists and holds photos wins; otherwise the seed script
# generates placeholder tiles into <data>/sample_tiles. (The older
# week3/project1/dataset holds multi-tile batch photos and is not used.)
DATASET_CANDIDATES = [
    PLATFORM_DIR.parent / "dataset_new" / "original",
]


def relocate(path: str | None) -> str | None:
    """
    A stored path that no longer exists but pointed inside the project folder (the
    course folder was renamed week2 -> week3, or the repository was moved or cloned
    elsewhere) is re-pointed at the same file in the current project folder.
    """
    if not path or Path(path).expanduser().exists():
        return path
    m = re.search(r"[\\/]week\d+[\\/]project1[\\/](.+)$", str(path))
    if m:
        moved = PLATFORM_DIR.parent.joinpath(*re.split(r"[\\/]", m.group(1)))
        if moved.exists():
            return str(moved)
    return path


def data_dir() -> Path:
    d = Path(os.environ.get("TILE_PLATFORM_DATA", PLATFORM_DIR / "var"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "platform.db"


def uploads_dir() -> Path:
    d = data_dir() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def env_checkpoint() -> str | None:
    return os.environ.get("TILE_PLATFORM_CHECKPOINT")


def env_images() -> str | None:
    return os.environ.get("TILE_PLATFORM_IMAGES")


def env_device() -> str:
    return os.environ.get("TILE_PLATFORM_DEVICE", "auto")


VALID_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def has_photos(folder) -> bool:
    try:
        return any(f.lower().endswith(VALID_EXT) for f in os.listdir(folder))
    except OSError:
        return False


def is_single_tile_dataset(folder: Path) -> bool:
    """The new dataset names photos tile_XXXX.jpg (old multi-tile one: mosaic_XXXX.jpg)."""
    try:
        return any(f.startswith("tile_") and f.lower().endswith(VALID_EXT) for f in os.listdir(folder))
    except OSError:
        return False


def default_image_dir() -> Path:
    if env_images():
        return Path(env_images())
    for c in DATASET_CANDIDATES:
        if is_single_tile_dataset(c):
            return c
    return data_dir() / "sample_tiles" / "original"
