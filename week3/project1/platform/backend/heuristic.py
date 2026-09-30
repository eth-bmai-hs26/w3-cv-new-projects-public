"""
DEMO MODE: a hand-written heuristic that imitates the 3-class segmentation
(0 = belt, 1 = usable tile area, 2 = damaged zone) so the platform works before
any model is trained. It is NOT a trained model and it is easily fooled by
patterned tiles, stains or unusual belts; the UI labels its results as demo.

    tile   = pixels that differ from the belt colours seen along the image border,
             largest blob, convex hull (so chipped-off corners count as damaged)
    cracks = thin dark lines inside the tile (black top-hat)
    damage = cracks + chipped area, grown by a cutting margin
"""

from __future__ import annotations

import cv2
import numpy as np


def _kmeans(samples: np.ndarray, k=6, iters=8, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = samples[rng.choice(len(samples), size=min(k, len(samples)), replace=False)].copy()
    for _ in range(iters):
        d = ((samples[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(len(c)):
            sel = samples[lab == j]
            if len(sel):
                c[j] = sel.mean(0)
    return c


def tile_mask(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(hull mask, blob mask), both bool, for an RGB uint8 image at working size."""
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring = max(4, h // 32)
    # belt samples: left/right strips (the belt runs through them), skipping the
    # top/bottom 20% where side guards sit; plus the full top/bottom ring as a fallback
    y0, y1 = int(0.2 * h), int(0.8 * h)
    border = np.concatenate([lab[y0:y1, :ring].reshape(-1, 3), lab[y0:y1, -ring:].reshape(-1, 3)])
    border = border[:: max(1, len(border) // 2000)]
    centers = _kmeans(border)
    dist = np.sqrt(((lab[:, :, None, :] - centers[None, None]) ** 2).sum(-1)).min(-1)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.5)
    d8 = np.clip(dist * 2, 0, 255).astype(np.uint8)
    thr, fg = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr < 12:                                       # nothing stands out from the belt
        return np.zeros((h, w), bool), np.zeros((h, w), bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return np.zeros((h, w), bool), np.zeros((h, w), bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < 0.01 * h * w:
        return np.zeros((h, w), bool), np.zeros((h, w), bool)
    blob = (labels == best).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(blob)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    hull = np.zeros_like(blob)
    cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(cnts))], 1)
    return hull.astype(bool), filled.astype(bool)


def segment(rgb: np.ndarray):
    """
    rgb: uint8 (H, W, 3) at the model resolution.
    Returns (class_map uint8 {0,1,2}, prob_usable float32 in [0,1] inside the tile).
    """
    h, w = rgb.shape[:2]
    hull, blob = tile_mask(rgb)
    cls = np.zeros((h, w), np.uint8)
    q = np.zeros((h, w), np.float32)
    if not hull.any():
        return cls, q

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k7).astype(np.float32)
    inner = cv2.erode(blob.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    vals = tophat[inner]
    thr = max(14.0, float(vals.mean() + 3.5 * vals.std())) if vals.size else 255.0
    cracks = ((tophat > thr) & inner).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cracks, 8)
    keep = np.zeros_like(cracks)
    for j in range(1, n):
        x, y, bw, bh, area = stats[j]
        if area >= 12 and max(bw, bh) >= 10:              # line-like, not a speck
            keep[labels == j] = 1
    chips = (hull & ~blob).astype(np.uint8)
    chips = cv2.morphologyEx(chips, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    side = np.sqrt(hull.sum())
    r = max(3, int(side * 0.055))
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    damage = cv2.dilate(keep | chips, grow).astype(bool) & hull

    cls[hull] = 1
    cls[damage] = 2
    soft = cv2.GaussianBlur(damage.astype(np.float32), (0, 0), 2.0)
    q = np.where(hull, 1.0 - soft, 0.0).astype(np.float32)
    return cls, q
