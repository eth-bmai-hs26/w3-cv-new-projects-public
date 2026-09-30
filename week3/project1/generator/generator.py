#!/usr/bin/env python
"""Synthetic inspection-station photos: ONE reclaimed tile on a conveyor belt, with pixel-exact masks.

Scene: top-down machine-vision camera over a rubber / PVC / PU belt (CC0 photo-texture structure,
re-tinted), optional side rails or belt edge at the frame border, splice seams, dust and small debris;
light-box illumination (mostly diffuse, soft short shadow, mild gradient, exposure / white-balance
jitter, sometimes a soft-box reflection on glossy tiles); slight perspective, blur, motion blur along
the belt, sensor noise, JPEG.  Tile materials, damage and distractors come from tilelib.py.

MASKS (all 0/255 PNG, same homography + 2x->1x area filter as the photo, so they align pixel-exactly):
  tiles/     TILE      every visible tile-material pixel (broken-off chips / missing pieces are NOT tile).
  damage/    DAMAGE    raw defect pixels inside the tile's nominal outline: crack lines, glaze flakes,
                       broken-off chip areas and missing pieces (the latter two lie outside TILE).
  segmented/ USABLE    TILE minus the unusable zone, where
                         zone = dilate(DAMAGE, disk of radius margin) AND TILE,
                         margin = cut_margin_frac (default 3 %) x the tile's short side in pixels,
                       i.e. what remains after a cutter removes every defect plus a safety margin.
                       Guaranteed USABLE is a subset of TILE; the damaged zone = TILE and not USABLE.
  Distractors (dirt, stains, grout residue, faint scratches, glaze/edge wear) are never DAMAGE.

  usable_fraction = |USABLE| / |TILE|.  Default decision rule: APPROVE if usable_fraction >= 0.85
  (cfg "approve_threshold"); pristine tiles have usable_fraction = 1.

API:  generate(seed, size=512, cfg=None, jpeg=True) -> (rgb uint8 HxWx3, masks dict of bool HxW
      {"tile", "damage", "usable"}, info dict)
CLI:  python generator.py --n 12 --out samples/
      python generator.py --dataset OUTDIR --n 2500 --workers 8 --seed 0
      -> OUTDIR/original/tile_0001.jpg (q90), tiles/tile_0001-tiles.png, damage/tile_0001-damage.png,
         segmented/tile_0001-segmented.png, ground_truth.csv
         (filename, tile_pixels, usable_pixels, usable_fraction, damage_pixels, has_damage, damage_types,
          material, background_variant, seed, coverage_pct, status)
      coverage_pct = usable_pixels / image pixels (kept for the course's build_cache), status uses the rule.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilelib as T   # noqa: E402
from tilelib import F32, fbm, hsv, pick, smoothstep, vnoise, jagged_poly  # noqa: E402

CFG = {
    **T.DEFAULT_CFG,
    # --- tile + damage
    "tile_sizes": [(10, 10), (15, 15), (20, 20), (30, 30), (20, 10), (30, 15), (40, 20), (30, 20), (15, 7.5)],
    "tile_frac": (0.40, 0.85),       # tile long side / frame width
    "tile_rot_deg": 12.0,
    "p_pristine": 0.38,
    "damage_types": {"crack": 0.37, "chip": 0.23, "crack+chip": 0.20, "shatter": 0.20},
    "severity_pow": 0.6,             # severity = U(0,1) ** pow (larger -> milder damage on average)
    "crack_px": (6, 9),              # crack width, 2x px at 512 -> >= 1.5 px at 256
    "hairline_px": (5, 6.5),         # low-severity cracks: >= 1.25 px at 256, still high contrast
    "chip_min_px": 40,               # min chip radius, 2x px at 512 (10 px at 256)
    "min_visible_damage": 0.6,
    "cut_margin_frac": 0.03,         # unusable zone = defect dilated by 3 % of the tile short side
    "approve_threshold": 0.85,
    # --- belt / scene
    "belts": {"black_rubber": 0.32, "green_pvc": 0.22, "grey_pvc": 0.2, "blue_pu": 0.14, "white_pu": 0.12},
    "rails_p": 0.55, "belt_edge_p": 0.15, "seam_p": 0.3, "debris_p": 0.4,
    "highlight_p": 0.35,             # soft-box reflection on glossy tiles
    "light_el_deg": (60, 85),        # light box: high, diffuse
    # --- camera
    "persp_jitter": 0.015, "global_rot_deg": 1.5,
    "blur_sigma": (0.0, 0.6), "motion_blur_p": 0.3, "noise_sigma": (0.003, 0.012),
    "wb_jitter": 0.04, "exposure": (0.9, 1.1), "vignette": (0.0, 0.2), "gradient": (0.0, 0.12),
    "jpeg_quality": (85, 95),
}


# ----------------------------------------------------------------------------- belt background
def _belt_texture(rng, cfg, name, C, pc):
    lst = [t for t in T.tex_list(cfg, "belt") if t[0] == name] or T.tex_list(cfg, "belt")
    if not lst:
        return None
    tex = lst[0][1]
    s = float(np.clip(rng.uniform(1.5, 3.0) * 30.0 / pc, 0.8, 6.0))   # texture px per canvas px
    a = math.radians(90 * rng.integers(4))
    ca, sa = math.cos(a) * s, math.sin(a) * s
    ox, oy = rng.uniform(0, tex.shape[0], 2)
    M = np.array([[ca, -sa, ox], [sa, ca, oy]], np.float32)
    g = cv2.cvtColor(tex, cv2.COLOR_RGB2GRAY)
    img = cv2.warpAffine(g, M, (C, C), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_WRAP)
    L = img.astype(F32)
    return L / max(1.0, float(L.mean()))


def _streaks(rng, C, pc, vertical, cell=0.6):
    """Noise stretched 30x along the belt motion (fbm's aniso stretches along x)."""
    return fbm(rng, C, C, 18 * pc, 3, aniso=1 / 30.0) if vertical else fbm(rng, C, C, cell * pc, 3, aniso=30.0)


def bg_belt(rng, C, pc, cfg, keep, frame):
    """keep = (x0, y0, x1, y1) tile bounding box, frame = visible box, both in canvas px."""
    variant = pick(rng, cfg["belts"])
    vertical = rng.random() < 0.5           # belt motion along the image y axis
    if variant == "black_rubber":
        base, tname, k = hsv(rng.random(), rng.uniform(0, 0.08), rng.uniform(0.10, 0.2)), \
            ["rubber004", "rubberized_track"][rng.integers(2)], rng.uniform(0.5, 1.0)
    elif variant == "green_pvc":
        base, tname, k = hsv(rng.uniform(0.3, 0.42), rng.uniform(0.35, 0.6), rng.uniform(0.25, 0.45)), "fabric030", rng.uniform(0.2, 0.5)
    elif variant == "blue_pu":
        base, tname, k = hsv(rng.uniform(0.55, 0.62), rng.uniform(0.35, 0.6), rng.uniform(0.3, 0.5)), \
            ["fabric030", "rubber004"][rng.integers(2)], rng.uniform(0.2, 0.5)
    elif variant == "grey_pvc":
        base, tname, k = hsv(rng.random(), rng.uniform(0, 0.06), rng.uniform(0.35, 0.55)), "fabric030", rng.uniform(0.3, 0.6)
    else:  # white_pu
        base, tname, k = hsv(rng.uniform(0.08, 0.15), rng.uniform(0, 0.06), rng.uniform(0.7, 0.82)), "fabric030", rng.uniform(0.15, 0.3)
    L = _belt_texture(rng, cfg, tname, C, pc)
    if L is None:   # procedural fallback
        L = 1 + 0.25 * rng.standard_normal((C, C), dtype=F32) * (0.5 + vnoise(rng, C, C, 0.3 * pc))
    streak = _streaks(rng, C, pc, vertical)       # wear streaks run along the motion
    lum = (1 + k * (L - 1)) * (1 + 0.035 * streak + 0.04 * fbm(rng, C, C, 12 * pc, 2))
    img = lum[..., None] * base
    # dust: soft patches + fine specks
    dust = np.array([0.62, 0.6, 0.56], F32) * rng.uniform(0.85, 1.1)
    m = smoothstep(0.3, 2.0, fbm(rng, C, C, 6 * pc, 3)) * rng.uniform(0.03, 0.14)
    spk = rng.random((C, C), dtype=F32) < rng.uniform(0.0003, 0.002)
    m = np.maximum(m, spk * rng.uniform(0.3, 0.6))
    img = img * (1 - m[..., None]) + m[..., None] * dust
    info = {"background_variant": variant, "belt_vertical": bool(vertical)}
    yy = np.arange(C, dtype=F32)[:, None]
    xx = np.arange(C, dtype=F32)[None, :]
    along_c, across_c = (yy, xx) if vertical else (xx, yy)   # coordinate along / across the motion
    # splice seam across the belt (finger splice = zig-zag, or skived = straight band)
    if rng.random() < cfg["seam_p"]:
        p = rng.uniform(0.1, 0.9) * C
        if rng.random() < 0.6:
            per, amp = rng.uniform(2, 4) * pc, rng.uniform(0.8, 2) * pc
            tri = amp * (2 * np.abs((across_c / per) % 1 - 0.5) - 0.5)
            d = along_c - p - tri
            img *= (1 - 0.35 * np.exp(-(d / max(1.0, 0.04 * pc)) ** 2))[..., None]
            img *= (1 + rng.uniform(-0.05, 0.05) * (d > 0))[..., None]
        else:
            wdt = rng.uniform(1, 3) * pc
            band = (np.abs(along_c - p) < wdt).astype(F32)
            img *= (1 + rng.uniform(-0.07, 0.07) * band)[..., None]
        info["seam"] = True
    # side rails / belt edge at the frame border (only where the tile leaves room)
    x0, y0, x1, y1 = keep
    lo, hi = (x0, x1) if vertical else (y0, y1)
    flo, fhi = (frame[0], frame[2]) if vertical else (frame[1], frame[3])
    rails = []
    for side in (0, 1):
        room = (lo - flo) if side == 0 else (fhi - hi)
        r = rng.random()
        wr = rng.uniform(0.04, 0.09) * C
        if r < cfg["rails_p"] and room > wr + 0.05 * C:
            kind = "rail"
        elif r < cfg["rails_p"] + cfg["belt_edge_p"] and room > wr + 0.05 * C:
            kind = "edge"
        else:
            continue
        edge = flo + wr if side == 0 else fhi - wr
        dist = (edge - across_c) if side == 0 else (across_c - edge)   # >0 inside the rail / off-belt zone
        if kind == "rail":
            t = np.clip(dist / wr, 0, 1)
            metal = rng.uniform(0.5, 0.72) * (1 + 0.05 * _streaks(rng, C, pc, vertical, 0.5))
            shade = 0.8 + 0.35 * np.exp(-((t - rng.uniform(0.3, 0.7)) / 0.15) ** 2) - 0.25 * np.exp(-(t / 0.05) ** 2)
            rail = (metal * shade)[..., None] * np.array([0.97, 0.99, 1.02], F32)
            inside = (dist > 0).astype(F32)[..., None]
            shadow = np.clip(1 - 0.45 * np.exp(-np.maximum(-dist, 0) / (0.6 * pc)), 0, 1) * (dist <= 0)
            img = img * (1 - inside) * np.where(dist <= 0, shadow, 1)[..., None] + inside * rail
        else:   # belt ends: dark machine bed beyond a slightly lighter, worn belt edge
            inside = (dist > 0).astype(F32)[..., None]
            bed = np.array([0.07, 0.07, 0.075], F32) * (1 + 0.2 * fbm(rng, C, C, 3 * pc, 2))[..., None]
            lip = np.exp(-(np.maximum(-dist, 0) / (0.15 * pc)) ** 2) * (dist <= 0)
            img = img * (1 + 0.25 * lip)[..., None]
            img = img * (1 - inside) + inside * bed
        rails.append(kind)
    info["rails"] = rails
    # debris: crumbs of grout / clay / chipped glaze, kept away from the tile
    if rng.random() < cfg["debris_p"]:
        for _ in range(rng.integers(1, 6)):
            for _try in range(10):
                px, py = rng.uniform(0.03, 0.97, 2) * C
                if not (x0 - 0.06 * C < px < x1 + 0.06 * C and y0 - 0.06 * C < py < y1 + 0.06 * C):
                    break
            else:
                continue
            rr = rng.uniform(0.15, 0.5) * pc
            poly = jagged_poly(rng, (px, py), rr, rng.integers(5, 8), 1, 0.2)
            colr = [np.array([0.7, 0.68, 0.63], F32), np.array([0.62, 0.38, 0.26], F32),
                    np.array([0.85, 0.84, 0.8], F32)][rng.integers(3)] * rng.uniform(0.85, 1.05)
            sh = np.zeros((C, C), np.uint8)
            cv2.fillPoly(sh, [np.round((poly + 0.25 * pc) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
            img *= (1 - 0.35 * sh.astype(F32)[..., None] / 255)
            cv2.fillPoly(img, [np.round(poly * 16).astype(np.int32)], tuple(float(c) for c in colr), cv2.LINE_AA, 4)
        info["debris"] = True
    return img, info


# ----------------------------------------------------------------------------- main
def _soft(m, sigma, q=4):
    """Gaussian blur with a large sigma, computed at 1/q resolution (shadows / AO are smooth)."""
    C = m.shape[0]
    small = cv2.resize(m, (C // q, C // q), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / q))
    return cv2.resize(small, (C, C), interpolation=cv2.INTER_LINEAR)



def camera_homography(rng, cfg, S2, C):
    src = np.array([[0, 0], [S2, 0], [S2, S2], [0, S2]], np.float32)
    rot = math.radians(rng.uniform(-cfg["global_rot_deg"], cfg["global_rot_deg"]))
    jit = rng.uniform(-1, 1, (4, 2)) * cfg["persp_jitter"] * S2
    scale = rng.uniform(0.95, 1.0)
    for _ in range(30):
        c = (src - S2 / 2) * scale
        dst = np.stack([c[:, 0] * math.cos(rot) - c[:, 1] * math.sin(rot),
                        c[:, 0] * math.sin(rot) + c[:, 1] * math.cos(rot)], 1) + jit * scale + C / 2
        if dst.min() >= 2 and dst.max() <= C - 3:
            break
        scale *= 0.97
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32)), dst


def generate(seed, size=512, cfg=None, jpeg=True):
    cfg = {**CFG, **(cfg or {})}
    rng = np.random.default_rng(seed)
    ss = cfg["supersample"]
    S2 = size * ss
    C = int(S2 * 1.06)
    sc = S2 / 1024.0
    Hm, quad = camera_homography(rng, cfg, S2, C)
    qx0, qx1 = max(quad[0, 0], quad[3, 0]), min(quad[1, 0], quad[2, 0])
    qy0, qy1 = max(quad[0, 1], quad[1, 1]), min(quad[2, 1], quad[3, 1])
    fw, fh = qx1 - qx0, qy1 - qy0

    # ---- the tile: material lot, size, pose (fully inside the frame)
    lot = T.make_lot(rng, cfg)
    lot["dmg_off"] = 0.0
    wcm, hcm = lot["size"]
    L = rng.uniform(*cfg["tile_frac"]) * fw
    ang = math.radians(rng.uniform(-cfg["tile_rot_deg"], cfg["tile_rot_deg"]))
    for _ in range(40):
        w, h = (L, L * hcm / wcm) if wcm >= hcm else (L * wcm / hcm, L)
        bw = w * abs(math.cos(ang)) + h * abs(math.sin(ang))
        bh = w * abs(math.sin(ang)) + h * abs(math.cos(ang))
        if bw <= 0.94 * fw and bh <= 0.94 * fh:
            break
        L *= 0.96
    pc = L / max(wcm, hcm)
    slx, sly = max(0.0, (fw - bw) / 2 - 0.02 * fw), max(0.0, (fh - bh) / 2 - 0.02 * fh)
    cx = (qx0 + qx1) / 2 + rng.uniform(-slx, slx)
    cy = (qy0 + qy1) / 2 + rng.uniform(-sly, sly)
    damage = None if rng.random() < cfg["p_pristine"] else pick(rng, cfg["damage_types"])
    sev = float(rng.random() ** cfg["severity_pow"]) if damage else 0.0
    tile = {"cx": cx, "cy": cy, "w": w, "h": h, "ang": ang, "lot": 0, "damage": damage, "sev": sev,
            "tint": (1 + rng.normal(0, 0.035)) * (1 + rng.normal(0, 0.015, 3)).astype(F32),
            "grad": rng.uniform(0.03, 0.12), "gdir": rng.uniform(0, 2 * np.pi), "pool": rng.uniform(-0.04, 0.1),
            "rot90": int(rng.integers(4)), "phase": rng.uniform(0, 10), "sc": sc,
            "thick": lot["thick_cm"] * pc * rng.uniform(0.93, 1.07)}
    info = {"seed": int(seed), "size": size, "material": lot["material"], "texture": lot.get("tex_name"),
            "tile_cm": [float(wcm), float(hcm)], "severity": round(sev, 3), "rot_deg": round(math.degrees(ang), 2)}

    # ---- light box
    az = rng.uniform(0, 2 * np.pi)
    el = math.radians(rng.uniform(*cfg["light_el_deg"]))
    Lv = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    Hv = Lv + np.array([0, 0, 1.0])
    Hv /= np.linalg.norm(Hv)
    light = {"L": Lv, "Hv": Hv, "spot": (cx + rng.normal(0, 0.3) * w, cy + rng.normal(0, 0.3) * h),
             "spot_sigma": rng.uniform(0.3, 0.6) * C, "spot_I": rng.uniform(0.02, 0.1),
             "spec_col": np.array([1.0, 0.99, 0.97], F32), "box": None}
    if lot["gloss"] > 0.15 and rng.random() < cfg["highlight_p"]:
        light["box"] = (cx + rng.uniform(-0.35, 0.35) * w, cy + rng.uniform(-0.35, 0.35) * h,
                        rng.uniform(0.05, 0.25) * w, rng.uniform(0.05, 0.25) * h, rng.uniform(0.02, 0.06) * w,
                        rng.uniform(0.15, 0.45))
        info["highlight"] = True

    # ---- belt + tile
    ex = abs(w / 2 * math.cos(ang)) + abs(h / 2 * math.sin(ang))
    ey = abs(w / 2 * math.sin(ang)) + abs(h / 2 * math.cos(ang))
    canvas, binfo = bg_belt(rng, C, pc, cfg, (cx - ex, cy - ey, cx + ex, cy + ey), (qx0, qy0, qx1, qy1))
    info.update(binfo)
    visible = np.zeros((C, C), np.uint8)
    cv2.fillPoly(visible, [np.round(quad).astype(np.int32)], 1)
    p = T.render_tile(rng, tile, lot, pc, light, cfg, C, visible)
    y0, y1, x0, x1 = p["bbox"]
    a = p["alpha"]
    union = np.zeros((C, C), F32)
    union[y0:y1, x0:x1] = a
    # soft light-box shadow: contact AO + short, blurred directional shadow
    off = tile["thick"] / math.tan(el)
    ox, oy = -int(round(math.cos(az) * off)), -int(round(math.sin(az) * off))
    sh = _soft(np.roll(union, (oy, ox), (0, 1)), rng.uniform(0.3, 0.8) * pc)
    ao = _soft(union, 0.25 * pc)
    canvas *= (1 - rng.uniform(0.2, 0.4) * sh - 0.3 * ao * (1 - union))[..., None]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - a[..., None]) + p["col"] * a[..., None]
    dmg2 = np.zeros((C, C), F32)
    if p["dmgf"] is not None:
        dmg2[y0:y1, x0:x1] = p["dmgf"]
    # mild light-box gradient (scene-fixed)
    g = rng.uniform(*cfg["gradient"])
    yy = (np.arange(C, dtype=F32)[:, None] - C / 2) / C
    xx = (np.arange(C, dtype=F32)[None, :] - C / 2) / C
    canvas *= (1 + g * (xx * math.cos(az) + yy * math.sin(az)) + 0.02 * fbm(rng, C, C, 0.5 * C, 2))[..., None]

    # ---- camera warp: identical homography + area filter for photo and masks
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    img = cv2.resize(cv2.warpPerspective(canvas, Hm, (S2, S2), flags=flags, borderMode=cv2.BORDER_REFLECT),
                     (size, size), interpolation=cv2.INTER_AREA)
    warp = lambda m: cv2.resize(cv2.warpPerspective(m, Hm, (S2, S2), flags=flags, borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0), (size, size), interpolation=cv2.INTER_AREA)
    tile_m = warp(union) > 0.5
    dmg_m = warp(dmg2) > 0.3
    r = max(1, int(round(cfg["cut_margin_frac"] * min(w, h) / ss)))
    zone = cv2.dilate(dmg_m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    usable_m = tile_m & ~zone
    assert not (usable_m & ~tile_m).any()

    # ---- camera effects
    vy = (np.arange(size, dtype=F32)[:, None] - size / 2) / (size / 2)
    vx = (np.arange(size, dtype=F32)[None, :] - size / 2) / (size / 2)
    img *= (1 - rng.uniform(*cfg["vignette"]) * 0.5 * (vx * vx + vy * vy))[..., None]
    img *= ((1 + rng.uniform(-cfg["wb_jitter"], cfg["wb_jitter"], 3)) * rng.uniform(*cfg["exposure"])).astype(F32)
    bs = rng.uniform(*cfg["blur_sigma"])
    if bs > 0.25:
        img = cv2.GaussianBlur(img, (0, 0), bs)
    if rng.random() < cfg["motion_blur_p"]:   # belt motion during exposure
        n = int(rng.integers(2, 4))
        kern = np.zeros((n, n), np.float32)
        if info["belt_vertical"]:
            kern[:, n // 2] = 1.0 / n
        else:
            kern[n // 2, :] = 1.0 / n
        img = cv2.filter2D(img, -1, kern)
        info["motion_blur_px"] = n
    ns = rng.uniform(*cfg["noise_sigma"])
    img = img + ns * np.sqrt(np.clip(img, 0.02, 1.5)) * (1.5 * rng.standard_normal((size, size, 1), dtype=F32)
                                                         + 0.5 * rng.standard_normal((size, size, 3), dtype=F32))
    img = np.clip(img, 0, 1) ** (1 / rng.uniform(0.94, 1.06))
    rgb = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    q = int(rng.integers(cfg["jpeg_quality"][0], cfg["jpeg_quality"][1] + 1))
    if jpeg:
        ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), q])
        rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()

    tp, up, dp = int(tile_m.sum()), int(usable_m.sum()), int(dmg_m.sum())
    uf = up / max(1, tp)
    dmg_kind = tile["damage"]           # render_tile may drop a defect that could not be made visible
    info.update({"tile_pixels": tp, "usable_pixels": up, "usable_fraction": round(uf, 4), "damage_pixels": dp,
                 "has_damage": int(dmg_kind is not None), "damage_types": dmg_kind or "none",
                 "distractors": tile.get("distract", []), "cut_margin_px": r, "jpeg_quality": q,
                 "status": "APPROVE" if uf >= cfg["approve_threshold"] else "REJECT",
                 "tile_px_512": [round(w / ss, 1), round(h / ss, 1)]})
    return rgb, {"tile": tile_m, "damage": dmg_m, "usable": usable_m}, info


# ----------------------------------------------------------------------------- dataset writer / CLI
CSV_COLS = ["filename", "tile_pixels", "usable_pixels", "usable_fraction", "damage_pixels", "has_damage",
            "damage_types", "material", "background_variant", "seed", "coverage_pct", "status"]


def _write_png(path, m):
    cv2.imwrite(path, m.astype(np.uint8) * 255)


def _work_ds(args):
    idx, seed, out, size, ndig = args
    cv2.setNumThreads(1)
    rgb, masks, info = generate(seed, size, jpeg=False)
    stem = f"tile_{idx:0{ndig}d}"
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    with open(os.path.join(out, "original", stem + ".jpg"), "wb") as f:
        f.write(buf.tobytes())
    _write_png(os.path.join(out, "tiles", stem + "-tiles.png"), masks["tile"])
    _write_png(os.path.join(out, "damage", stem + "-damage.png"), masks["damage"])
    _write_png(os.path.join(out, "segmented", stem + "-segmented.png"), masks["usable"])
    row = {"filename": stem + ".jpg", **{k: info[k] for k in ("tile_pixels", "usable_pixels", "damage_pixels",
                                                             "has_damage", "damage_types", "material",
                                                             "background_variant", "seed", "status")},
           "usable_fraction": f"{info['usable_fraction']:.4f}",
           "coverage_pct": f"{info['usable_pixels'] / float(size * size):.4f}"}
    return row


def write_dataset(out, n, seed=0, workers=1, size=512):
    for d in ("original", "tiles", "damage", "segmented"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    ndig = max(4, len(str(n)))
    jobs = [(i, seed * 1_000_003 + i, out, size, ndig) for i in range(1, n + 1)]
    t0 = time.time()
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            rows = pool.map(_work_ds, jobs, chunksize=4)
    else:
        rows = [_work_ds(j) for j in jobs]
    with open(os.path.join(out, "ground_truth.csv"), "w") as f:
        f.write(",".join(CSV_COLS) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in CSV_COLS) + "\n")
    uf = np.array([float(r["usable_fraction"]) for r in rows])
    print(f"{len(rows)} images -> {out} in {time.time() - t0:.1f}s; APPROVE (>= {CFG['approve_threshold']}) "
          f"{(uf >= CFG['approve_threshold']).mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.dataset:
        write_dataset(a.dataset, a.n, a.seed, a.workers, a.size)
        return
    os.makedirs(a.out, exist_ok=True)
    for s in range(a.start, a.start + a.n):
        rgb, masks, info = generate(s, a.size)
        cv2.imwrite(os.path.join(a.out, f"tile_{s:05d}.jpg"), rgb[..., ::-1])
        for k, m in masks.items():
            _write_png(os.path.join(a.out, f"tile_{s:05d}-{k}.png"), m)
    print(f"{a.n} samples -> {a.out}")


if __name__ == "__main__":
    main()
