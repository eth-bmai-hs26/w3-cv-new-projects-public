"""Photo storage and the overlay shown to the inspector (usable = green, damaged zone = red)."""

from __future__ import annotations

import io
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 768
GREEN = np.array([38, 170, 96], np.float32)
RED = np.array([226, 58, 42], np.float32)


try:                                            # iPhone photos (HEIC/HEIF), if pillow-heif is installed
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def open_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)          # phone photos
    return img.convert("RGB")


def _fit(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(1.0, MAX_SIDE / max(w, h))
    return img if s == 1.0 else img.resize((round(w * s), round(h * s)), Image.LANCZOS)


def overlay(img: Image.Image, classes: np.ndarray) -> Image.Image:
    """Tint usable pixels green and the damaged zone red; the belt stays untouched."""
    rgb = np.asarray(img, dtype=np.float32)
    h, w = rgb.shape[:2]
    # upsample per-class masks smoothly, then re-threshold (no blocky 256px staircase)
    use = cv2.resize((classes == 1).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
    dmg = cv2.resize((classes == 2).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
    out = rgb.copy()
    out[use] = out[use] * 0.62 + GREEN * 0.38
    out[dmg] = out[dmg] * 0.42 + RED * 0.58
    out = out.clip(0, 255).astype(np.uint8)
    t = max(1, round(max(w, h) / 320))
    for m, col in ((dmg, (255, 96, 80)), (use | dmg, (240, 255, 244))):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, t, cv2.LINE_AA)
    return Image.fromarray(out)


def store(root: Path, img: Image.Image, classes: np.ndarray, when: datetime | None = None) -> dict:
    """Save photo, overlay and class mask; returns paths relative to `root`."""
    when = when or datetime.now()
    day = when.strftime("%Y-%m-%d")
    (root / day).mkdir(parents=True, exist_ok=True)
    key = uuid.uuid4().hex[:12]
    photo = _fit(img)
    rel = {
        "photo_path": f"{day}/{key}.jpg",
        "overlay_path": f"{day}/{key}-overlay.jpg",
        "mask_path": f"{day}/{key}-classes.png",
    }
    photo.save(root / rel["photo_path"], quality=88)
    overlay(photo, classes).save(root / rel["overlay_path"], quality=88)
    Image.fromarray(classes.astype(np.uint8)).save(root / rel["mask_path"])   # values 0/1/2, for retraining
    return rel
