"""
Placeholder photos: ONE reclaimed tile lying on a conveyor belt.

The course dataset is being regenerated in this format; until it is available
the platform can generate its own stand-in images so the demo, the seed script
and the tests always have something realistic to inspect.

Layout written by `write_folder(out, n)` (same as the course dataset):
    out/original/tile_0001.jpg            photo (512x512 RGB)
    out/segmented/tile_0001-segmented.png usable-area mask (255 = sellable)
    out/tiles/tile_0001-tiles.png         whole-tile footprint mask
    out/damage/tile_0001-damage.png       damaged / removed zone (tile - usable)
    out/ground_truth.csv                  filename, usable_fraction, damage_fraction, ...

usable_fraction = usable pixels / tile footprint pixels.

CLI:  python -m backend.sample_tiles OUT_DIR --n 60      (run inside platform/)
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import cv2
import numpy as np

SIZE = 512

MATERIALS = {
    # name: (base RGB, variation, speckle)
    "terracotta": ((176, 92, 58), 18, 0.05),
    "cream glaze": ((226, 216, 192), 8, 0.02),
    "grey porcelain": ((150, 150, 146), 10, 0.10),
    "blue glaze": ((52, 92, 140), 10, 0.02),
    "green glaze": ((78, 118, 92), 10, 0.02),
    "encaustic": ((214, 204, 186), 6, 0.02),
    "slate": ((74, 78, 82), 10, 0.08),
}


def _noise(rng, h, w, cell):
    """Smooth value noise in [0, 1]."""
    gh, gw = max(2, h // cell + 2), max(2, w // cell + 2)
    g = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def _belt(rng):
    h = w = SIZE
    base = rng.uniform(34, 46)
    tone = np.array([base, base * rng.uniform(1.0, 1.06), base * rng.uniform(0.98, 1.04)], np.float32)
    img = np.ones((h, w, 3), np.float32) * tone
    # rubber grain + lengthwise wear streaks (belt runs left -> right)
    grain = rng.normal(0, 4.5, (h, w)).astype(np.float32)
    streak = cv2.resize(rng.normal(0, 7, (h, 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=40, sigmaY=1.2)
    img += (grain + streak + 10 * (_noise(rng, h, w, 96) - 0.5))[..., None]
    # transverse cleats every ~ 120 px
    period = int(rng.uniform(110, 150))
    offset = int(rng.uniform(0, period))
    for x in range(-period + offset, w, period):
        cv2.rectangle(img, (x, 0), (x + 7, h), (tone * 0.62).tolist(), -1)
        cv2.line(img, (x + 8, 0), (x + 8, h), (tone * 1.35).tolist(), 1)
    # steel side guards top and bottom
    gh = int(rng.uniform(26, 38))
    for y0, y1 in ((0, gh), (h - gh, h)):
        grad = np.linspace(150, 110, y1 - y0, dtype=np.float32)[:, None]
        if y0 > 0:
            grad = grad[::-1]
        img[y0:y1] = np.stack([grad * 0.97, grad, grad * 1.03], -1) + rng.normal(0, 3, (y1 - y0, w, 1))
        for bx in range(20, w, 64):
            cy = (y0 + y1) // 2
            cv2.circle(img, (bx, cy), 4, (80, 82, 86), -1, cv2.LINE_AA)
            cv2.circle(img, (bx - 1, cy - 1), 2, (190, 192, 196), -1, cv2.LINE_AA)
    return img, gh


def _tile_texture(rng, mat, th, tw):
    base, var, speck = MATERIALS[mat]
    base = np.array(base, np.float32) * rng.uniform(0.9, 1.08)
    tex = np.ones((th, tw, 3), np.float32) * base
    tex += (var * (_noise(rng, th, tw, 40) - 0.5) * 2)[..., None]
    tex += (0.5 * var * (_noise(rng, th, tw, 9) - 0.5) * 2)[..., None]
    sp = rng.random((th, tw)) < speck
    tex[sp] *= rng.uniform(0.7, 1.2)
    if mat == "encaustic":
        ink = np.array(rng.choice([(40, 60, 110), (150, 50, 40), (40, 90, 70), (40, 40, 44)]), np.float32)
        yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
        cx, cy = tw / 2, th / 2
        r = np.hypot(xx - cx, yy - cy) / (min(th, tw) / 2)
        diamond = (np.abs(xx - cx) + np.abs(yy - cy)) / (min(th, tw) / 2)
        pat = ((diamond > 0.55) & (diamond < 0.72)) | (r < 0.22) | ((np.minimum(xx, tw - xx) < tw * 0.06) | (np.minimum(yy, th - yy) < th * 0.06))
        tex[pat] = tex[pat] * 0.25 + ink * 0.75
    if "glaze" in mat:
        # soft specular sheen
        g = np.linspace(-1, 1, tw, dtype=np.float32)[None, :] * rng.uniform(-1, 1) + np.linspace(-1, 1, th, dtype=np.float32)[:, None] * rng.uniform(-1, 1)
        tex += (18 * np.exp(-(g ** 2) * 3))[..., None]
    # bevel: darker rim
    rim = np.ones((th, tw), np.float32)
    b = max(3, int(min(th, tw) * 0.025))
    for i in range(b):
        f = 0.78 + 0.22 * i / b
        rim[i, :] = np.minimum(rim[i, :], f); rim[-1 - i, :] = np.minimum(rim[-1 - i, :], f)
        rim[:, i] = np.minimum(rim[:, i], f); rim[:, -1 - i] = np.minimum(rim[:, -1 - i], f)
    tex *= rim[..., None]
    return tex


def _crack_walk(rng, th, tw):
    side = rng.integers(4)
    if side == 0:
        p = np.array([rng.uniform(0.1, 0.9) * tw, 0.0])
    elif side == 1:
        p = np.array([tw - 1.0, rng.uniform(0.1, 0.9) * th])
    elif side == 2:
        p = np.array([rng.uniform(0.1, 0.9) * tw, th - 1.0])
    else:
        p = np.array([0.0, rng.uniform(0.1, 0.9) * th])
    target = np.array([rng.uniform(0.25, 0.75) * tw, rng.uniform(0.25, 0.75) * th])
    ang = math.atan2(*(target - p)[::-1])
    pts = [p.copy()]
    length = rng.uniform(0.45, 1.3) * max(th, tw)
    walked = 0.0
    while walked < length:
        ang += rng.normal(0, 0.35)
        step = rng.uniform(5, 11)
        p = p + step * np.array([math.cos(ang), math.sin(ang)])
        walked += step
        pts.append(p.copy())
        if not (0 <= p[0] < tw and 0 <= p[1] < th):
            break
    return np.array(pts, np.int32)


def render(seed: int, severity: float | None = None):
    """
    One photo + masks. `severity` in [0, 1] steers how damaged the tile is
    (None = random). Returns (rgb uint8, usable bool, tile bool, info dict).
    """
    rng = np.random.default_rng(seed)
    if severity is None:
        severity = float(rng.beta(0.8, 1.6))
    img, guard = _belt(rng)

    mat = str(rng.choice(list(MATERIALS)))
    side = rng.uniform(250, 330)
    aspect = rng.choice([1.0, 1.0, 1.0, 0.5, 0.66])
    tw, th = int(side), int(side * aspect) if aspect != 1.0 else int(side)
    if aspect != 1.0 and rng.random() < 0.5:
        tw, th = th, tw
    tex = _tile_texture(rng, mat, th, tw)
    tile_local = np.ones((th, tw), np.uint8) * 255      # footprint in tile coords
    damage_local = np.zeros((th, tw), np.uint8)
    crack_local = np.zeros((th, tw), np.uint8)
    chip_local = np.zeros((th, tw), np.uint8)

    n_cracks = 0 if severity < 0.22 else int(1 + severity * 3.2 * rng.uniform(0.5, 1.2))
    n_chips = 0 if severity < 0.35 else int(rng.integers(0, 2 + int(severity * 2)))
    zone = max(6, int(min(th, tw) * rng.uniform(0.045, 0.07)))       # cut-away margin around damage
    for _ in range(n_cracks):
        pts = _crack_walk(rng, th, tw)
        cv2.polylines(crack_local, [pts], False, 255, int(rng.integers(2, 4)), cv2.LINE_AA)
        if rng.random() < 0.5 and len(pts) > 6:        # a branch
            k = int(rng.integers(2, len(pts) - 2))
            q = pts[k].astype(np.float64)
            a = rng.uniform(0, 2 * math.pi)
            br = [q.copy()]
            for _ in range(int(rng.integers(5, 14))):
                a += rng.normal(0, 0.4)
                q = q + 8 * np.array([math.cos(a), math.sin(a)])
                br.append(q.copy())
            cv2.polylines(crack_local, [np.array(br, np.int32)], False, 255, 2, cv2.LINE_AA)
    for _ in range(n_chips):
        cx = rng.choice([0, tw - 1]) if rng.random() < 0.6 else rng.uniform(0, tw)
        cy = rng.choice([0, th - 1]) if rng.random() < 0.6 else rng.choice([0, th - 1])
        r = rng.uniform(0.07, 0.16) * min(th, tw)
        k = int(rng.integers(7, 12))
        poly = [(cx + r * rng.uniform(0.6, 1.3) * math.cos(2 * math.pi * i / k),
                 cy + r * rng.uniform(0.6, 1.3) * math.sin(2 * math.pi * i / k)) for i in range(k)]
        cv2.fillPoly(chip_local, [np.array(poly, np.int32)], 255, cv2.LINE_AA)

    crack_bin = crack_local > 60
    chip_bin = chip_local > 127
    dmg = (crack_bin | chip_bin).astype(np.uint8) * 255
    if dmg.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * zone + 1, 2 * zone + 1))
        damage_local = cv2.dilate(dmg, k)
    # crack appearance: dark groove with a thin highlight
    c = crack_local.astype(np.float32)[..., None] / 255.0
    tex = tex * (1 - 0.72 * c)
    hl = np.roll(crack_local, 1, axis=0).astype(np.float32)[..., None] / 255.0
    tex = tex + 25 * np.clip(hl - c, 0, 1)
    # distractors that are NOT damage: stains, grout residue
    for _ in range(int(rng.integers(0, 3))):
        m = np.zeros((th, tw), np.float32)
        cv2.circle(m, (int(rng.uniform(0, tw)), int(rng.uniform(0, th))), int(rng.uniform(10, 40)), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 9) * rng.uniform(0.08, 0.2)
        tex = tex * (1 - m[..., None]) + np.array([90, 70, 50], np.float32) * m[..., None]
    alpha_local = (tile_local > 0) & ~chip_bin

    # place the tile on the belt: rotation + translation
    ang = rng.uniform(-14, 14)
    cx = SIZE / 2 + rng.uniform(-40, 40)
    cy = SIZE / 2 + rng.uniform(-22, 22)
    M = cv2.getRotationMatrix2D((tw / 2, th / 2), ang, 1.0)
    M[0, 2] += cx - tw / 2
    M[1, 2] += cy - th / 2

    def warp(a, interp=cv2.INTER_LINEAR):
        return cv2.warpAffine(a, M, (SIZE, SIZE), flags=interp, borderValue=0)

    alpha = warp(alpha_local.astype(np.float32))
    foot = warp(tile_local, cv2.INTER_NEAREST) > 127
    damage = warp(damage_local, cv2.INTER_NEAREST) > 127
    tex_w = warp(tex)

    # drop shadow
    sh = cv2.GaussianBlur(alpha, (0, 0), 7)
    sh = np.roll(np.roll(sh, 6, axis=0), 5, axis=1)
    img *= (1 - 0.55 * sh)[..., None]
    img = img * (1 - alpha[..., None]) + tex_w * alpha[..., None]

    # lighting + camera
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    light = 1.0 + 0.12 * (np.cos((xx - rng.uniform(0.3, 0.7)) * 3) - 0.5) - 0.18 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    img *= light[..., None] * rng.uniform(0.9, 1.1)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.0))
    img += rng.normal(0, 2.5, img.shape)
    rgb = img.clip(0, 255).astype(np.uint8)

    usable = foot & ~damage
    info = {
        "material": mat,
        "severity": round(float(severity), 3),
        "n_cracks": n_cracks,
        "n_chips": n_chips,
        "usable_fraction": round(float(usable.sum() / max(1, foot.sum())), 4),
        "tile_pixels": int(foot.sum()),
        "usable_pixels": int(usable.sum()),
    }
    return rgb, usable, foot, info


def write_folder(out, n=60, start=1, seed=0, prefix="tile"):
    """Write n samples in the course dataset layout. Returns the list of image paths."""
    for sub in ("original", "segmented", "tiles", "damage"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    rows, paths = [], []
    for i in range(start, start + n):
        rgb, usable, foot, info = render(seed * 100_003 + i)
        stem = f"{prefix}_{i:04d}"
        p = os.path.join(out, "original", f"{stem}.jpg")
        cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(out, "segmented", f"{stem}-segmented.png"), usable.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(out, "tiles", f"{stem}-tiles.png"), foot.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(out, "damage", f"{stem}-damage.png"), (foot & ~usable).astype(np.uint8) * 255)
        rows.append({"filename": f"{stem}.jpg", "usable_fraction": info["usable_fraction"],
                     "damage_fraction": round(1 - info["usable_fraction"], 4),
                     "tile_pixels": info["tile_pixels"], "usable_pixels": info["usable_pixels"],
                     "material": info["material"], "n_cracks": info["n_cracks"], "n_chips": info["n_chips"]})
        paths.append(p)
    csv_path = os.path.join(out, "ground_truth.csv")
    new = not os.path.exists(csv_path) or start == 1
    with open(csv_path, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    write_folder(a.out, a.n, seed=a.seed)
    print(f"wrote {a.n} samples to {a.out}")


if __name__ == "__main__":
    main()
