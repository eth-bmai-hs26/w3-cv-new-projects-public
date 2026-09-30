"""Shared rendering primitives for the tile-inspection generator (materials, damage and distractors).

Noise, CC0 photo-texture sampling, crack / chip geometry, tile materials (glazed, terracotta, marble,
stone, concrete, encaustic), distractors (dirt, stains, grout residue, faint scratches, edge wear) and
`render_tile`, which renders ONE tile patch (colour, alpha, damage map) with bevel + gloss shading.
Round-3 label-visibility fixes are kept: chip body contrast, light cracks on dark tiles, min chip size.
Tile-inspection additions: continuous damage severity tile["sev"] and a soft-box specular reflection.
"""
import argparse
import colorsys
import glob
import json
import math
import os
import time
import cv2
import numpy as np

F32 = np.float32

DEFAULT_CFG = {
    # --- rendering
    "supersample": 2,
    "canvas_margin": 1.14,          # canvas side / (2x output side); room for rotation+perspective
    # --- layout
    "px_per_cm": (8.0, 14.0),       # at 2x supersampling
    "tile_sizes": [(10, 10), (15, 15), (20, 20), (20, 20), (30, 30), (20, 10), (30, 15), (40, 20), (15, 7.5)],
    "n_lots": (1, 4),
    "tile_rot_deg": 3.0,
    "pos_jitter_cm": 0.8,
    "gap_cm": (0.4, 3.0),
    "big_gap_p": 0.07,
    "skip_p": (0.0, 0.25),          # empty slot probability (per image range)
    "region_p": 0.35,               # prob. that tiles occupy only a sub-rectangle of the scene
    "transpose_p": 0.5,             # columns instead of rows
    # --- materials
    "materials": {"glazed": 0.20, "terracotta": 0.20, "marble": 0.15, "stone": 0.12, "concrete": 0.13, "encaustic": 0.20},
    "use_textures": True,           # CC0 photo textures from textures/<category>/*.jpg (procedural fallback if missing)
    "texture_dir": None,            # default: <this dir>/textures
    "edge_wear": True,
    "bevel": True,
    "specular": True,
    # --- damage (a damaged tile is fully 0 in the mask)
    "damage_rate": (0.0, 0.5),      # per image range of P(tile damaged) for "normal" batches
    "batch_modes": {"normal": 0.845, "heavy": 0.09, "pristine": 0.065},   # widen coverage tails
    "crack_px": (6, 9),             # main crack width in 2x px at size 512 (>= 3 px at 512)
    "chip_min_px": 28,              # min chip radius in 2x px at size 512 (visible at 256 px)
    "chip_frac": (0.15, 0.3),       # chip radius as fraction of the tile's short side
    "min_body_contrast": 0.3,       # |luminance(exposed body) - luminance(tile)|: chips visible in grayscale
    "adaptive_crack": True,         # light (dust-filled) cracks on dark tiles, where dark cracks are invisible
    "min_visible_damage": 0.6,      # >= this fraction of a damaged tile's defect pixels must be inside the frame
    "damage_types": {"crack": 0.45, "chip": 0.30, "crack+chip": 0.13, "shatter": 0.12},
    "crack_displacement_p": 0.35,
    # --- non-damage distractors (tile stays 1)
    "distractors": True,
    "p_dirt": 0.35, "p_stain": 0.22, "p_grout": 0.16, "p_scratch": 0.30,
    # --- scene
    "backgrounds": {"pallet": 0.25, "planks": 0.15, "cardboard": 0.25, "concrete": 0.35},
    "shadows": True,                # height-field shadows: floor AND lower neighbouring tiles
    "uneven_light": True,
    "occluder_shadow_p": 0.2,
    "perspective": True,
    "persp_jitter": 0.035,          # corner jitter as fraction of side
    "global_rot_deg": 4.0,
    # --- camera
    "blur_sigma": (0.0, 0.8),
    "noise_sigma": (0.004, 0.018),
    "wb_jitter": 0.06,
    "exposure": (0.85, 1.12),
    "vignette": (0.0, 0.35),
    "jpeg_quality": (70, 95),
}


# ----------------------------------------------------------------------------- utils
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def vnoise(rng, h, w, cy, cx=None):
    """Value noise in [0,1] with feature size ~cy (rows) x cx (cols) pixels."""
    cx = cy if cx is None else cx
    gh = max(2, int(h / max(cy, 1e-3)) + 3)
    gw = max(2, int(w / max(cx, 1e-3)) + 3)
    g = rng.random((gh, gw), dtype=F32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def fbm(rng, h, w, cell, octaves=4, pers=0.5, aniso=1.0):
    """Fractal noise, approx zero mean / unit std. aniso>1 stretches features along x."""
    out = np.zeros((h, w), F32)
    amp, tot = 1.0, 0.0
    c = float(cell)
    for _ in range(octaves):
        if c < 1.2:
            n = rng.random((h, w), dtype=F32)
        else:
            n = vnoise(rng, h, w, c, c * aniso)
        out += amp * (n - 0.5)
        tot += amp * amp
        amp *= pers
        c /= 2.0
    return out / (0.24 * math.sqrt(tot))


def hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, np.clip(s, 0, 1), np.clip(v, 0, 1)), F32)


def pick(rng, d):
    keys = list(d.keys())
    p = np.array([d[k] for k in keys], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------- photo textures
TEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")
_TEX_CACHE = {}


def tex_list(cfg, cat):
    """List of (name, uint8 RGB array, mean RGB float) for a category; [] if disabled/missing."""
    if not cfg.get("use_textures", True):
        return []
    d = os.path.join(cfg.get("texture_dir") or TEX_DIR, cat)
    if d not in _TEX_CACHE:
        out = []
        for f in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            if im is not None:
                im = np.ascontiguousarray(im[..., ::-1])
                out.append((os.path.basename(f)[:-4], im, im.reshape(-1, 3).mean(0).astype(F32) / 255))
        _TEX_CACHE[d] = out
    return _TEX_CACHE[d]


def pick_tex(rng, cfg, cat):
    lst = tex_list(cfg, cat)
    return lst[rng.integers(len(lst))] if lst else None


def photo_canvas(rng, tex, C, crop=(0.55, 0.95), rot90=True, rot_jit=4.0):
    """Random crop (scale, rotation, wrap-around offset) of a seamless texture, resampled to CxC float RGB."""
    T = tex.shape[0]
    s = rng.uniform(*crop) * T / C
    a = math.radians((90 * rng.integers(4) if rot90 else 180 * rng.integers(2)) + rng.uniform(-rot_jit, rot_jit))
    ca, sa = math.cos(a) * s, math.sin(a) * s
    ox, oy = rng.uniform(0, T, 2)
    M = np.array([[ca, -sa, ox - ca * C / 2 + sa * C / 2], [sa, ca, oy - sa * C / 2 - ca * C / 2]], np.float32)
    img = cv2.warpAffine(tex, M, (C, C), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_WRAP)
    return img.astype(F32) / 255


def color_jitter(rng, img, sat=(0.8, 1.1), gain=0.05, bright=(0.85, 1.1), target=None):
    """Saturation / per-channel gain / brightness as one 3x3 colour matrix (cv2.transform, fast).
    target: optionally rescale so that the mean colour becomes `target` (RGB)."""
    sv = rng.uniform(*sat)
    Msat = sv * np.eye(3) + (1 - sv) / 3.0
    g = (1 + rng.uniform(-gain, gain, 3)) * rng.uniform(*bright)
    M = g[:, None] * Msat
    if target is not None:
        mean = np.array(cv2.mean(img)[:3])
        M = (np.asarray(target, float) / np.clip(M @ mean, 1e-3, None))[:, None] * M
    return cv2.transform(img, M.astype(np.float32))


def sample_tex(rng, tex, u, v, w, h, frac, rot90):
    """Texture crop that moves/rotates with the tile (local coords u,v)."""
    T = tex.shape[0]
    k = frac * T / max(w, h)
    for _ in range(rot90):
        u, v = v, -u
    ox, oy = rng.uniform(0, T, 2)
    mx = (u * k + ox).astype(np.float32)
    my = (v * k + oy).astype(np.float32)
    return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).astype(F32) / 255



# ----------------------------------------------------------------------------- crack geometry
def crack_walk(rng, start, ang, w, h, max_len, step=(3, 7), jag=0.35, pull=0.18, drift=0.05):
    """Jagged random walk in tile-local coords (origin at tile centre). Returns Nx2 points."""
    pts = [start]
    x, y = start
    L = 0.0
    a = heading = ang
    while L < max_len:
        s = rng.uniform(*step)
        heading += rng.normal(0, drift)
        a += rng.normal(0, jag) + pull * wrap(heading - a)
        x += s * math.cos(a)
        y += s * math.sin(a)
        L += s
        pts.append((x, y))
        if abs(x) > w / 2 + 3 or abs(y) > h / 2 + 3:
            break
    return np.array(pts, F32)


def edge_point(rng, w, h):
    """Random point on the rectangle border and inward normal angle."""
    side = rng.choice(4, p=np.array([w, w, h, h]) / (2 * (w + h)))
    t = rng.uniform(-0.4, 0.4)
    if side == 0:
        return (t * w, -h / 2), math.pi / 2
    if side == 1:
        return (t * w, h / 2), -math.pi / 2
    if side == 2:
        return (-w / 2, t * h), 0.0
    return (w / 2, t * h), math.pi


def draw_poly(img, pts, thick, val=255):
    p = np.round(pts * 8).astype(np.int32)
    for i in range(len(p) - 1):
        t = float(thick[i])
        vv = int(val * min(1.0, 0.45 + 0.55 * t / 2)) if t < 2 else val
        cv2.line(img, tuple(p[i]), tuple(p[i + 1]), vv, int(max(1, round(t))), cv2.LINE_AA, 3)


def make_cracks(rng, w, h, kind, wr=(6, 9), sc=1.0, ncr=None, through_p=0.55, nrays=None,
                branch_rate=0.04, branch_max=3):
    """List of (points Nx2 local, thickness per segment). wr = main width range (2x px @512), sc = size scale.
    ncr / through_p / nrays / branch_* let the caller scale the damage with a severity value."""
    out = []
    diag = math.hypot(w, h)
    t0 = rng.uniform(*wr) * sc
    tmin = 0.75 * wr[0] * sc
    if kind == "star":
        cpt = (rng.uniform(-0.3, 0.3) * w, rng.uniform(-0.3, 0.3) * h)
        n = nrays if nrays is not None else rng.integers(3, 7)
        a0 = rng.uniform(0, 2 * np.pi)
        for i in range(n):
            a = a0 + 2 * np.pi * i / n + rng.normal(0, 0.3)
            pts = crack_walk(rng, cpt, a, w, h, 3 * diag, (5 * sc, 11 * sc), 0.28, 0.35, 0.04)
            out.append((pts, np.full(len(pts), t0, F32) + rng.normal(0, 0.4, len(pts)).astype(F32)))
    else:
        ncr = ncr if ncr is not None else rng.choice([1, 1, 1, 2, 2, 3])
        for _ in range(ncr):
            s, a = edge_point(rng, w, h)
            a += rng.normal(0, 0.45)
            through = kind == "through" or rng.random() < through_p
            ml = 3 * diag if through else rng.uniform(0.3, 0.75) * diag
            for _try in range(8):   # reject corner-clipping stubs: a crack must be clearly visible
                pts = crack_walk(rng, s, a, w, h, ml, (5 * sc, 11 * sc), rng.uniform(0.22, 0.38), 0.3, 0.04)
                if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() >= 0.4 * min(w, h):
                    break
                s, a = edge_point(rng, w, h)
                a += rng.normal(0, 0.3)
            th = np.full(len(pts), t0, F32) + rng.normal(0, 0.6 * sc, len(pts)).astype(F32)
            if not through:   # taper to hairline
                th *= np.linspace(1.0, 0.6, len(pts)).astype(F32)
            out.append((pts, np.maximum(th, tmin)))
    # branches
    branches = []
    for pts, th in out:
        nb = rng.poisson(len(pts) * branch_rate)
        for _ in range(min(nb, branch_max)):
            i = rng.integers(1, max(2, len(pts) - 1))
            d = pts[min(i + 1, len(pts) - 1)] - pts[i - 1]
            a = math.atan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.35, 1.0)
            bp = crack_walk(rng, tuple(pts[i]), a, w, h, rng.uniform(0.1, 0.45) * diag, (4 * sc, 9 * sc), 0.35, 0.3)
            bt = np.maximum(np.linspace(0.85 * th[i], tmin, len(bp)), tmin).astype(F32)
            branches.append((bp, bt))
    return out + branches


def jagged_poly(rng, center, r, n=7, levels=2, rough=0.2):
    angs = np.sort((np.arange(n) + rng.uniform(-0.3, 0.3, n)) * 2 * np.pi / n + rng.uniform(0, 2 * np.pi))
    rad = r * rng.uniform(0.45, 1.1, n)
    pts = np.stack([center[0] + rad * np.cos(angs), center[1] + rad * np.sin(angs)], 1)
    for _ in range(levels):
        q = np.roll(pts, -1, 0)
        seg = q - pts
        nrm = np.stack([-seg[:, 1], seg[:, 0]], 1)
        mid = (pts + q) / 2 + nrm * rng.normal(0, rough, (len(pts), 1))
        new = np.empty((2 * len(pts), 2))
        new[0::2] = pts
        new[1::2] = mid
        pts = new
    return pts.astype(F32)


def chip_center(rng, w, h):
    if rng.random() < 0.6:   # corner
        return (rng.choice([-1, 1]) * w / 2, rng.choice([-1, 1]) * h / 2)
    (p, _) = edge_point(rng, w, h)
    return p


# ----------------------------------------------------------------------------- materials
ENC_PALETTE = [(0.90, 0.86, 0.76), (0.12, 0.12, 0.13), (0.62, 0.20, 0.14), (0.78, 0.58, 0.22),
               (0.30, 0.42, 0.55), (0.30, 0.45, 0.35), (0.55, 0.55, 0.55), (0.80, 0.78, 0.72)]


TEX_CAT = {"terracotta": "terracotta", "marble": "marble", "stone": "stone", "concrete": "plain",
           "glazed": "plain", "encaustic": "plain"}


def make_lot(rng, cfg):
    mat = pick(rng, cfg["materials"])
    lot = {"material": mat}
    tex = pick_tex(rng, cfg, TEX_CAT[mat])
    if mat in ("glazed", "encaustic") and tex is not None and tex[0].startswith("terrazzo"):
        plain = [t for t in tex_list(cfg, "plain") if not t[0].startswith("terrazzo")]
        tex = plain[rng.integers(len(plain))] if plain else None
    if mat == "stone" and tex is None:
        mat = lot["material"] = "concrete"
    if mat == "encaustic":
        lot["size"] = [(15, 15), (20, 20), (20, 20)][rng.integers(3)]
        idx = rng.choice(len(ENC_PALETTE), 3, replace=False)
        lot["colors"] = [np.array(ENC_PALETTE[i], F32) for i in idx]
        lot["motif"] = int(rng.integers(6))
        lot["base"] = lot["colors"][0]
    else:
        s = cfg["tile_sizes"][rng.integers(len(cfg["tile_sizes"]))]
        lot["size"] = s if rng.random() < 0.5 else (s[1], s[0])
        if mat == "glazed":
            if rng.random() < 0.4:
                lot["base"] = hsv(rng.uniform(0.05, 0.15), rng.uniform(0.0, 0.12), rng.uniform(0.7, 0.92))
            else:   # muted, earthy glazes (less plasticky than saturated primaries)
                lot["base"] = hsv(rng.random(), rng.uniform(0.08, 0.32), rng.uniform(0.28, 0.68))
            lot["body"] = [np.array([0.88, 0.82, 0.72], F32), np.array([0.80, 0.55, 0.40], F32),
                           np.array([0.92, 0.9, 0.86], F32)][rng.integers(3)]
            lot["gloss"] = rng.uniform(0.15, 0.5)
        elif mat == "terracotta":
            lot["base"] = hsv(rng.uniform(0.02, 0.07), rng.uniform(0.45, 0.72), rng.uniform(0.42, 0.72))
            lot["body"] = np.clip(lot["base"] * np.array([1.3, 1.25, 1.2], F32) + 0.06, 0, 1)
            lot["gloss"] = rng.uniform(0.0, 0.06)
        elif mat == "marble":
            c = rng.random()
            if c < 0.5:
                lot["base"] = hsv(rng.uniform(0.05, 0.15), rng.uniform(0.0, 0.12), rng.uniform(0.78, 0.93))
            elif c < 0.75:
                lot["base"] = hsv(rng.uniform(0.02, 0.1), rng.uniform(0.15, 0.3), rng.uniform(0.6, 0.8))
            else:
                lot["base"] = hsv(rng.uniform(0.3, 0.6), rng.uniform(0.0, 0.3), rng.uniform(0.15, 0.35))
            lot["vein"] = lot["base"] * rng.uniform(0.35, 0.7) if lot["base"].mean() > 0.4 else np.array([0.85, 0.83, 0.8], F32)
            lot["body"] = np.clip(lot["base"] * 1.1 + 0.08, 0, 1)
            lot["gloss"] = rng.uniform(0.05, 0.5)
            lot["vein_dir"] = rng.uniform(0, np.pi)
        elif mat == "stone":
            lot["base"] = tex[2].copy()
            lot["body"] = np.clip(lot["base"] * 1.15 + 0.1, 0, 1)
            lot["gloss"] = rng.uniform(0.0, 0.15)
        else:  # concrete / cement tile
            lot["base"] = hsv(rng.uniform(0.0, 0.6), rng.uniform(0.0, 0.3), rng.uniform(0.35, 0.72))
            lot["body"] = np.clip(lot["base"] * 1.15 + 0.06, 0, 1)
            lot["gloss"] = rng.uniform(0.0, 0.05)
    if mat == "encaustic":
        lot["body"] = np.array([0.62, 0.61, 0.58], F32)
        lot["gloss"] = rng.uniform(0.03, 0.12)
        g = 0.55
        lot["colors"] = [g + 0.75 * (c - g) for c in lot["colors"]]   # pigment cement: muted colours
    # photo texture: how it is used per material
    if tex is not None:
        lot["tex"], lot["tex_name"], lot["tex_mean"] = tex[1], tex[0], tex[2]
        if mat in ("marble", "stone") or (mat == "concrete" and tex[0].startswith("terrazzo")):
            lot["tex_mode"] = "keep"
            lot["tex_gain"] = ((1 + rng.normal(0, 0.04, 3)) * rng.uniform(0.85, 1.1)).astype(F32)
            if mat == "marble":
                lot["base"] = tex[2].copy()
                lot["body"] = np.clip(lot["base"] * 1.1 + 0.12, 0, 1)
            lot["tex_frac"] = rng.uniform(0.3, 0.75) if mat == "marble" else rng.uniform(0.2, 0.5)
            lot["tex_contrast"] = 1.0
        elif mat in ("terracotta", "concrete"):
            lot["tex_mode"] = "tint"
            lot["tex_frac"] = rng.uniform(0.3, 0.7)
            lot["tex_contrast"] = rng.uniform(1.0, 2.0) if mat == "terracotta" else rng.uniform(0.8, 1.4)
        else:   # glazed / encaustic: only a subtle luminance modulation from a plaster-like photo
            lot["tex_mode"] = "lum"
            lot["tex_frac"] = rng.uniform(0.4, 0.9)
            lot["tex_contrast"] = rng.uniform(0.15, 0.3) if mat == "glazed" else rng.uniform(0.3, 0.6)
    # chips must stay visible on dark tiles: light broken body
    if float(np.mean(lot["base"])) < 0.38:
        lot["body"] = np.array([0.80, 0.77, 0.70], F32) * rng.uniform(0.9, 1.05)
    lot["thick_cm"] = rng.uniform(0.8, 1.2) if mat in ("glazed", "marble", "stone") else rng.uniform(1.0, 2.0)
    lot["bevel_cm"] = rng.uniform(0.2, 0.5)
    lot["corner_cm"] = rng.uniform(0.05, 0.5)
    return lot


def encaustic_pattern(lot, s, t, px, rot):
    """s,t in [-0.5,0.5] tile coords. Returns RGB field."""
    for _ in range(rot):
        s, t = t, -s
    c0, c1, c2 = lot["colors"]
    aa = 1.5 / px
    m = lot["motif"]
    layers = []

    def fill(f, thr):   # alpha = f < thr, anti-aliased
        return np.clip((thr - f) / aa + 0.5, 0, 1)
    if m == 0:    # border + diamond
        layers += [(1 - fill(np.maximum(np.abs(s), np.abs(t)), 0.42), c1), (fill(np.abs(s) + np.abs(t), 0.3), c2)]
    elif m == 1:  # quarter circles at corners + centre dot
        dc = np.hypot(0.5 - np.abs(s), 0.5 - np.abs(t))
        layers += [(fill(dc, 0.36), c1), (fill(dc, 0.22), c2), (fill(np.hypot(s, t), 0.1), c1)]
    elif m == 2:  # floral rosette
        r = np.hypot(s, t)
        th = np.arctan2(t, s)
        layers += [(fill(r - 0.12 * np.cos(4 * th), 0.26), c1), (fill(r, 0.08), c2),
                   (fill(np.hypot(0.5 - np.abs(s), 0.5 - np.abs(t)), 0.16), c2)]
    elif m == 3:  # checker 4x4
        n = 4
        a = (np.floor((s + 0.5) * n) + np.floor((t + 0.5) * n)) % 2
        layers += [(a.astype(F32), c1)]
    elif m == 4:  # cross + diamond ring
        layers += [(np.maximum(fill(np.abs(s), 0.08), fill(np.abs(t), 0.08)), c1),
                   (fill(np.abs(np.abs(s) + np.abs(t) - 0.33), 0.05), c2)]
    else:         # diagonal half (triangles)
        layers += [(fill(s - t, 0.0), c1), (fill(np.abs(s + t), 0.06), c2)]
    col = np.broadcast_to(c0, s.shape + (3,)).astype(F32)
    for a, c in layers:
        col = col * (1 - a[..., None]) + a[..., None] * c
    return col


def material_color(rng, lot, tile, u, v, H, W, pc):
    mat = lot["material"]
    w, h = tile["w"], tile["h"]
    base = lot["base"] * tile["tint"]
    grain = rng.standard_normal((H, W), dtype=F32)
    if "tex" in lot:
        t = sample_tex(rng, lot["tex"], u, v, w, h, lot["tex_frac"], tile["rot90"])
        mode, k = lot["tex_mode"], lot["tex_contrast"]
        if mode == "keep":
            col = t * (lot["tex_gain"] * tile["tint"])
        elif mode == "tint":
            col = base * np.clip(1 + k * (t / lot["tex_mean"].clip(1e-3) - 1), 0.2, 2.0)
        else:
            L = t.mean(2) / max(1e-3, float(lot["tex_mean"].mean()))
            mod = np.clip(1 + k * (L - 1), 0.5, 1.5)[..., None]
            if mat == "glazed":
                g = 1 + tile["grad"] * (u * math.cos(tile["gdir"]) + v * math.sin(tile["gdir"])) / max(w, h)
                col = (g + 0.006 * grain)[..., None] * base * mod
            else:
                col = encaustic_pattern(lot, u / w, v / h, min(w, h), tile["rot90"]) * tile["tint"] * mod
                wear = fbm(rng, H, W, 5 * pc, 3)
                col = col * (1 + 0.03 * grain)[..., None] + 0.05 * smoothstep(0.5, 1.8, wear)[..., None]
        if mat == "terracotta":
            spk = rng.random((H, W), dtype=F32)
            col = col * (1 + 0.035 * grain - 0.3 * (spk < 0.004))[..., None] + (0.1 * (spk > 0.997))[..., None]
        elif mat in ("concrete", "stone"):
            col = col * (1 + 0.03 * grain)[..., None]
        return col
    if mat == "glazed":
        g = 1 + tile["grad"] * (u * math.cos(tile["gdir"]) + v * math.sin(tile["gdir"])) / max(w, h)
        lum = g + 0.012 * fbm(rng, H, W, 6 * pc, 2) + 0.006 * grain
        col = lum[..., None] * base
    elif mat == "terracotta":
        mott = fbm(rng, H, W, 3 * pc, 3)
        burn = fbm(rng, H, W, 10 * pc, 2)
        lum = 1 + 0.07 * mott - 0.06 * smoothstep(0.5, 1.5, burn) + 0.045 * grain
        col = lum[..., None] * base
        spk = rng.random((H, W), dtype=F32)
        col *= (1 - 0.35 * (spk < 0.004))[..., None]
        col += (0.12 * (spk > 0.996))[..., None]
    elif mat == "marble":
        a = lot["vein_dir"] + tile["rot90"] * np.pi / 2
        turb = fbm(rng, H, W, 18 * pc, 5, 0.5)
        f = rng.uniform(0.02, 0.05) / pc
        x = (u * math.cos(a) + v * math.sin(a)) * f * np.pi + 1.4 * turb + tile["phase"]
        vein = np.exp(-np.abs(np.sin(x)) * rng.uniform(10, 22))
        cloud = fbm(rng, H, W, 8 * pc, 3)
        # sparse secondary veins: ridges of a second noise, only where a mask noise is high
        fine = np.exp(-np.abs(fbm(rng, H, W, 10 * pc, 4)) * rng.uniform(14, 24)) * smoothstep(1.3, 2.3, cloud) * 0.6
        vein = vein * (0.55 + 0.45 * smoothstep(-1.0, 1.0, fbm(rng, H, W, 5 * pc, 2)))   # veins fade in/out
        k = np.clip(0.75 * vein + 0.35 * fine, 0, 1)[..., None] * rng.uniform(0.35, 0.8)
        col = (1 + 0.04 * cloud + 0.01 * grain)[..., None] * base
        col = col * (1 - k) + k * lot["vein"]
    elif mat == "concrete":
        lum = 1 + 0.05 * fbm(rng, H, W, 2 * pc, 3) + 0.05 * grain
        spk = vnoise(rng, H, W, 1.8)
        lum += 0.12 * smoothstep(0.82, 0.95, spk) - 0.25 * smoothstep(0.12, 0.03, spk)
        col = lum[..., None] * base
    else:  # encaustic
        col = encaustic_pattern(lot, u / w, v / h, min(w, h), tile["rot90"]) * tile["tint"]
        wear = fbm(rng, H, W, 5 * pc, 3)
        col = col * (1 + 0.035 * grain + 0.04 * wear)[..., None] + 0.05 * smoothstep(0.5, 1.8, wear)[..., None]
    return col


def body_color(rng, lot, H, W):
    grain = rng.standard_normal((H, W), dtype=F32)
    spk = rng.random((H, W), dtype=F32)
    lum = 1 + 0.06 * grain - 0.25 * (spk < 0.03)
    return lum[..., None] * lot["body"]


# ----------------------------------------------------------------------------- tile rendering
def render_tile(rng, tile, lot, pc, light, cfg, C, visible):
    cx, cy, w, h, ang = tile["cx"], tile["cy"], tile["w"], tile["h"], tile["ang"]
    ca, sa = math.cos(ang), math.sin(ang)
    ex = abs(w / 2 * ca) + abs(h / 2 * sa) + 3
    ey = abs(w / 2 * sa) + abs(h / 2 * ca) + 3
    x0, x1 = max(0, int(cx - ex)), min(C, int(cx + ex) + 1)
    y0, y1 = max(0, int(cy - ey)), min(C, int(cy + ey) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    vis = visible[y0:y1, x0:x1]
    if not vis.any():
        return None
    H, W = y1 - y0, x1 - x0
    dx = np.arange(x0, x1, dtype=F32)[None, :] - cx
    dy = np.arange(y0, y1, dtype=F32)[:, None] - cy
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    r = min(lot["corner_cm"] * pc, 0.3 * min(w, h))
    qx = np.abs(u) - (w / 2 - r)
    qy = np.abs(v) - (h / 2 - r)
    sd = -(np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r)
    alpha = np.clip(sd + 0.5, 0, 1)
    alpha0 = alpha

    def to_patch(pts):   # local -> patch pixel coords
        return np.stack([cx + pts[:, 0] * ca - pts[:, 1] * sa - x0, cy + pts[:, 0] * sa + pts[:, 1] * ca - y0], 1)

    # ---------------- damage geometry (decided before texturing so we can check visibility)
    dmg_kind = tile["damage"]
    crack = flake = notch = missing = None
    pieces = None
    for attempt in range(4):
        if dmg_kind is None:
            break
        crack_u8 = np.zeros((H, W), np.uint8)
        flake_u8 = np.zeros((H, W), np.uint8)
        notch_u8 = np.zeros((H, W), np.uint8)
        flake2_u8 = np.zeros((H, W), np.uint8)
        sev = float(tile.get("sev", 0.5))            # 0 = minor .. 1 = severe (continuous)
        if "crack" in dmg_kind or dmg_kind == "shatter":
            kind = "star" if dmg_kind == "shatter" else "any"
            wr = cfg.get("hairline_px", cfg["crack_px"]) if (sev < 0.3 and dmg_kind != "shatter") else cfg["crack_px"]
            ncr = 1 + int(sev ** 1.1 * 4.99) if dmg_kind == "crack" else 1 + int(sev * 2.99)
            for pts, th in make_cracks(rng, w, h, kind, wr, tile["sc"], ncr=ncr, through_p=0.35 + 0.6 * sev,
                                       nrays=3 + int(sev * 5.99), branch_rate=0.015 + 0.05 * sev,
                                       branch_max=1 + int(4 * sev)):
                draw_poly(crack_u8, to_patch(pts), th)
        if "chip" in dmg_kind:
            nchip = 1 + int(sev * 3.99)
            for _ in range(nchip):
                rr = max(cfg["chip_min_px"] * tile["sc"], rng.uniform(0.8, 1.2) * (0.08 + 0.50 * sev) * min(w, h))
                cc = chip_center(rng, w, h)
                poly = to_patch(jagged_poly(rng, cc, rr, rng.integers(5, 9)))
                busy = lot["material"] == "encaustic" or str(lot.get("tex_name", "")).startswith("terrazzo")
                through = rng.random() < (0.8 if busy else 0.45)   # a notch stays visible on busy surfaces
                tgt = notch_u8 if through else flake_u8
                cv2.fillPoly(tgt, [np.round(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
                if not through and rng.random() < 0.6:   # stepped flake: deeper inner layer
                    inner = to_patch(jagged_poly(rng, cc, 0.55 * rr, rng.integers(5, 8)))
                    cv2.fillPoly(flake2_u8, [np.round(inner * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        crack = crack_u8.astype(F32) / 255
        flake = flake_u8.astype(F32) / 255 * alpha
        flake2 = flake2_u8.astype(F32) / 255 * flake
        notch = notch_u8.astype(F32) / 255
        missing = np.zeros((H, W), F32)
        if dmg_kind == "shatter" or (dmg_kind != "chip" and rng.random() < cfg["crack_displacement_p"]):
            region = ((alpha > 0.5) & (crack < 0.25)).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=4)
            areas = stats[1:, cv2.CC_STAT_AREA]
            big = [i + 1 for i in np.argsort(-areas) if areas[i] > 0.02 * w * h]
            if len(big) >= 2:
                pieces = (lab, big)
                if dmg_kind == "shatter":
                    k = min(len(big) - 1, 1 + int(float(tile.get("sev", 0.5)) * 3.99))
                    gone = set(int(i) for i in rng.choice(big[1:], k, replace=False))
                    keep = np.isin(lab, [i for i in range(1, n) if i not in gone]).astype(np.uint8)
                    # everything not within ~half a crack width of a kept piece falls away (no orphan
                    # crack strands left floating over the belt); fracture edges of kept pieces stay
                    rk = int(math.ceil(0.5 * max(cfg["crack_px"]) * tile["sc"])) + 1
                    near = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rk + 1, 2 * rk + 1)))
                    missing = ((alpha > 0.02) & (near == 0)).astype(F32)
                    rem = ((alpha > 0.5) & (missing < 0.5)).astype(np.uint8)
                    nr, lr, st, _ = cv2.connectedComponentsWithStats(rem, connectivity=8)
                    if nr > 2:   # only the largest connected remainder stays on the belt
                        main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                        missing = np.maximum(missing, ((lr > 0) & (lr != main)).astype(F32))
                        missing = np.maximum(missing, ((alpha > 0.02) & (lr == 0) & (cv2.dilate(
                            (lr == main).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)).astype(F32))
        dmg = np.maximum.reduce([crack, flake, notch * alpha, missing * alpha])
        dpx = dmg > 0.5   # label must be learnable: most of the defect has to be inside the frame
        nvis = int((dpx & (vis > 0)).sum())
        if nvis >= 100 * tile["sc"] ** 2 and nvis >= cfg["min_visible_damage"] * max(1, int(dpx.sum())):
            break
        if attempt == 3:
            dmg_kind = None
    tile["damage"] = dmg_kind
    if dmg_kind is None:
        crack = flake = notch = missing = None
        pieces = None

    # ---------------- material + height field
    col = material_color(rng, lot, tile, u, v, H, W, pc)
    bev = lot["bevel_cm"] * pc if cfg["bevel"] else 0.5
    t = np.clip(sd / bev, 0, 1)
    height = 0.6 * bev * (1 - (1 - t) ** 2)
    gloss = np.full((H, W), lot["gloss"], F32)
    wav = fbm(rng, H, W, 5 * pc, 2)
    height += (0.3 if lot["material"] in ("glazed", "marble") else 0.35) * wav
    if lot["material"] in ("terracotta", "concrete"):
        height += 0.35 * rng.standard_normal((H, W), dtype=F32)
    if lot["material"] == "glazed":   # glaze pooling at the edges
        col *= (1 - tile["pool"] * (1 - smoothstep(0, 2.2 * bev, sd)))[..., None]

    # ---------------- distractors (do NOT change the label)
    distract = tile.setdefault("distract", [])
    if cfg["distractors"]:
        ext = min(w, h)
        if rng.random() < cfg["p_dirt"]:
            distract.append("dirt")
            n = fbm(rng, H, W, rng.uniform(0.12, 0.35) * ext, 3)
            th = rng.uniform(-1.0, 0.8)
            edge_acc = 1 + 0.8 * np.exp(-np.maximum(sd, 0) / (0.08 * ext))
            m = smoothstep(th, th + 2.5, n) * rng.uniform(0.12, 0.35) * edge_acc * (0.75 + 0.25 * rng.random((H, W), dtype=F32))
            dc = [np.array([0.72, 0.70, 0.64], F32), np.array([0.30, 0.25, 0.20], F32),
                  np.array([0.48, 0.40, 0.31], F32)][rng.integers(3)]
            col = col * (1 - m[..., None]) + m[..., None] * dc
            gloss *= 1 - m
        if rng.random() < cfg["p_stain"]:
            distract.append("stain")
            for _ in range(rng.integers(1, 3)):
                q = (rng.uniform(-0.4, 0.4) * w, rng.uniform(-0.4, 0.4) * h)
                R = rng.uniform(0.1, 0.35) * ext
                rho = np.hypot(u - q[0], v - q[1]) / R + 0.12 * fbm(rng, H, W, R, 2)
                inside = smoothstep(1.05, 0.9, rho)
                rim = np.exp(-((rho - 1) / 0.1) ** 2) * inside
                tint = [np.array([0.65, 0.45, 0.30], F32), np.array([0.55, 0.45, 0.38], F32),
                        np.array([0.6, 0.62, 0.62], F32)][rng.integers(3)]
                s = rng.uniform(0.08, 0.22) * inside + rng.uniform(0.04, 0.12) * rim
                col = col * (1 - s[..., None] * (1 - tint))
        if rng.random() < cfg["p_scratch"]:
            distract.append("scratch")
            sc = np.zeros((H, W), np.uint8)
            a0 = rng.uniform(0, np.pi)
            for _ in range(rng.integers(2, 9)):
                p0 = np.array([rng.uniform(-0.45, 0.45) * w, rng.uniform(-0.45, 0.45) * h])
                a = a0 + rng.normal(0, 0.25)
                L = rng.uniform(0.1, 0.4) * math.hypot(w, h)
                bend = rng.normal(0, 0.05) * L
                p2 = p0 + L * np.array([math.cos(a), math.sin(a)])
                p1 = (p0 + p2) / 2 + bend * np.array([-math.sin(a), math.cos(a)])
                sw = rng.choice([1, 1, 2]) * max(1.0, tile["sc"])
                draw_poly(sc, to_patch(np.stack([p0, p1, p2]).astype(F32)), [sw, sw], 255)
            m = sc.astype(F32) / 255 * rng.uniform(0.15, 0.3)
            if rng.random() < 0.75:
                col = col + m[..., None] * (0.9 - col)
            else:
                col = col * (1 - m[..., None])
        if rng.random() < cfg["p_grout"]:
            distract.append("grout")
            side = rng.integers(4)
            dist = [v + h / 2, h / 2 - v, u + w / 2, w / 2 - u][side]
            n = fbm(rng, H, W, rng.uniform(0.04, 0.1) * ext, 4, 0.6)
            wgt = np.exp(-np.maximum(dist, 0) / (rng.uniform(0.05, 0.15) * ext))
            m = smoothstep(0.4, 0.8, 0.5 * n + 1.7 * wgt - rng.uniform(0.8, 1.1)) * rng.uniform(0.6, 0.95)
            gc = hsv(rng.uniform(0.08, 0.12), rng.uniform(0.02, 0.1), rng.uniform(0.6, 0.8))
            gtex = (1 + 0.08 * rng.standard_normal((H, W), dtype=F32))[..., None] * gc
            col = col * (1 - m[..., None]) + m[..., None] * gtex
            height += 1.3 * smoothstep(0.1, 0.6, m) * alpha
            gloss *= 1 - m

    # ---------------- edge wear (all tiles, not damage): rubbed arris, grime in the bevel
    body = lot["body"]
    if cfg.get("edge_wear", True):
        rim = 1 - smoothstep(0, 1.3 * bev, sd)
        ew = rim * smoothstep(0.2, 1.2, fbm(rng, H, W, 0.8 * pc, 3)) * rng.uniform(0.2, 0.45)
        col = col * (1 - ew[..., None]) + ew[..., None] * (0.5 * col + 0.5 * body)
        col *= (1 - rng.uniform(0.03, 0.12) * (1 - smoothstep(0, 2.5 * bev, sd)))[..., None]
        gloss *= 1 - ew

    # ---------------- damage appearance
    if dmg_kind is not None:
        body = body_color(rng, lot, H, W)
        LW = np.array([0.299, 0.587, 0.114], F32)
        inside = alpha0 > 0.5
        Lt = float(np.median(col[inside] @ LW)) if inside.any() else 0.5
        Lb = float(lot["body"] @ LW)
        mc = cfg["min_body_contrast"]
        if abs(Lb - Lt) < mc:   # push the exposed body away from the tile's luminance (else chip invisible)
            tgt = Lt - mc - rng.uniform(0, 0.1) if (Lt > 0.5 or Lt + mc > 0.95) else Lt + mc + rng.uniform(0, 0.1)
            tgt = float(np.clip(tgt, 0.08, 0.92))
            body = body * (tgt / max(Lb, 1e-3))
        if flake.any():
            depth = rng.uniform(1.0, 2.5)
            fl = cv2.GaussianBlur(flake, (0, 0), 0.7)
            col = col * (1 - flake[..., None]) + flake[..., None] * body
            height -= depth * fl + 0.8 * depth * cv2.GaussianBlur(flake2, (0, 0), 0.7)
            col *= (1 - 0.08 * flake2)[..., None]
            gloss *= 1 - flake
        if notch.any():
            band = np.clip(cv2.GaussianBlur(notch, (0, 0), 2.2 * tile["sc"]) * 2.5, 0, 1) * (1 - notch)
            col = col * (1 - 0.9 * band[..., None]) + 0.9 * band[..., None] * body
            height -= 1.5 * band
            gloss *= 1 - band
        if missing.any():
            band = np.clip(cv2.GaussianBlur(missing, (0, 0), 1.6) * 2.5, 0, 1) * (1 - missing)
            col = col * (1 - 0.7 * band[..., None]) + 0.7 * band[..., None] * body * 0.85
            height -= 1.5 * band
        if pieces is not None and dmg_kind != "shatter":   # small displacement across crack
            lab, big = pieces
            pm = lab == big[1]
            ddx, ddy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
            shifted = np.roll(col, (ddy, ddx), (0, 1))
            col[pm] = shifted[pm]
            height[pm] -= rng.uniform(0.5, 1.5)
        if crack.any():
            cb = cv2.GaussianBlur(crack, (0, 0), 0.8)
            halo = cv2.GaussianBlur(crack, (0, 0), 2.5)
            dark = rng.uniform(0.55, 0.8)
            if cfg.get("adaptive_crack", True):
                if lot["material"] == "encaustic":   # multi-colour pattern: decide per pixel, narrow transition
                    L = cv2.GaussianBlur(col @ LW, (0, 0), 3.0)
                    lightw = smoothstep(0.28, 0.25, L)[..., None]
                else:                                # one decision per tile (a grey blend would be invisible)
                    lightw = np.full((H, W, 1), float(Lt < 0.26), F32)
                dust = np.array([0.78, 0.76, 0.72], F32) * rng.uniform(0.9, 1.05)
                ccol = (1 - lightw) * col * (1 - dark) + lightw * dust
                col = col * (1 - crack[..., None]) + crack[..., None] * ccol
                col *= (1 - 0.1 * halo[..., None] * (1 - lightw))
            else:
                col = col * (1 - dark * crack[..., None]) * (1 - 0.1 * halo[..., None])
            height -= rng.uniform(1.2, 2.2) * cb
            gloss *= 1 - np.clip(halo * 2, 0, 1)
        alpha = alpha * (1 - notch) * (1 - missing)

    # ---------------- shading: normal from height field, Lambert + specular
    lx, ly, lz = light["L"]
    gy, gx = np.gradient(height)
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1)
    ndl = (-gx * lx - gy * ly + lz) * nz / lz
    amb = 0.4
    shade = np.clip(amb + (1 - amb) * ndl, 0.15, 1.8)
    col = col * shade[..., None]
    if cfg["specular"]:
        hx, hy, hz = light["Hv"]
        ndh = np.clip((-gx * hx - gy * hy + hz) * nz, 0, 1)
        edge_spec = np.clip(ndh ** 60 - hz ** 60, 0, 1)
        X = dx + cx + 25 * (-gx * nz)
        Y = dy + cy + 25 * (-gy * nz)
        sc_, ss_ = light["spot"], light["spot_sigma"]
        spot = np.exp(-((X - sc_[0]) ** 2 + (Y - sc_[1]) ** 2) / (2 * ss_ * ss_))
        if light.get("box") is not None:   # rectangular soft-box reflection, rippled by the glaze normals
            bx, by, bw_, bh_, soft, bI = light["box"]
            spot = spot + bI / max(light["spot_I"], 1e-3) * (smoothstep(bw_ + soft, bw_ - soft, np.abs(X - bx)) *
                                                             smoothstep(bh_ + soft, bh_ - soft, np.abs(Y - by)))
        spec = gloss * (light["spot_I"] * spot + 0.8 * edge_spec)
        col = col + spec[..., None] * light["spec_col"]
    dmgmap = dmgf = None
    if dmg_kind is not None:   # raw defect pixels inside the tile's nominal outline (incl. broken-off areas)
        dmgf = np.maximum.reduce([crack * alpha0, flake, notch * alpha0, missing * alpha0]).astype(F32)
        dmgmap = dmgf > 0.5
    return {"bbox": (y0, y1, x0, x1), "col": np.clip(col, 0, 1.5).astype(F32), "alpha": alpha.astype(F32),
            "intact": dmg_kind is None, "dmg": dmgmap, "dmgf": dmgf, "alpha0": alpha0.astype(F32)}


