"""Review helpers for the tile-inspection generator.

python tools.py sheet sheet.png [seeds]      photo | usable mask | damage mask, 12 samples (4 x 3)
python tools.py zoom zoom.png                 2x crops of crack / chip / shatter / distractor detail
python tools.py check256 check_256.png [seeds]  what the notebook sees: 256 px grayscale + 3-class label
python tools.py overlay overlay.png [seeds]   tile (green) + usable (yellow) + damage (red) edges on the photo
python tools.py stats N                       timing + usable_fraction distribution
"""
import sys
import time

import cv2
import numpy as np
from PIL import Image

from generator import generate, CFG

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, txt, org=(6, 20), scale=0.5):
    cv2.putText(img, txt, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def sheet(path, seeds, size=512):
    cells = []
    for s in seeds:
        rgb, m, info = generate(s, size)
        half = size // 2
        ph = cv2.resize(rgb, (half, half), interpolation=cv2.INTER_AREA)
        us = np.dstack([m["usable"].astype(np.uint8) * 255] * 3)
        # usable mask with the damaged zone in grey so the three classes are visible
        us[m["tile"] & ~m["usable"]] = (110, 110, 110)
        us = cv2.resize(us, (half, half), interpolation=cv2.INTER_NEAREST)
        dm = cv2.resize(np.dstack([m["damage"].astype(np.uint8) * 255] * 3), (half, half), interpolation=cv2.INTER_NEAREST)
        dm[..., 1:] //= 3   # red-ish
        c = np.hstack([ph, us, dm])
        _label(c, f"s{s} usable {info['usable_fraction']:.3f} {info['status']}", (4, 16), 0.45)
        _label(c, f"{info['damage_types']} sev {info['severity']:.2f} | {info['material']} | {info['background_variant']}",
               (4, half - 8), 0.38)
        cells.append(cv2.copyMakeBorder(c, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    rows = [np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)]
    cv2.imwrite(path, np.vstack(rows)[..., ::-1])


def _crop_around(mask, rgb, want_on=True):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    i = len(ys) // 2
    y, x = int(np.clip(ys[i], 64, rgb.shape[0] - 64)), int(np.clip(xs[i], 64, rgb.shape[1] - 64))
    return rgb[y - 64:y + 64, x - 64:x + 64], (y, x)


def zoom(path, size=512):
    want = ["crack", "chip", "shatter", "distractor"]
    got = {}
    s = 0
    while len(got) < 4 and s < 400:
        rgb, m, info = generate(s, size)
        k = info["damage_types"]
        key = "crack" if k == "crack" else "chip" if k == "chip" else "shatter" if k == "shatter" else \
            ("distractor" if k == "none" and len(set(info["distractors"]) & {"stain", "scratch", "grout"}) >= 2 else None)
        if key in want and key not in got:
            if key == "distractor":
                r = _crop_around(m["tile"], rgb)
                txt = f"s{s} intact, {'+'.join(info['distractors'])} ({info['material']})"
            else:   # centre on defect pixels that touch the tile (missing pieces lie on the belt)
                near = cv2.dilate(m["tile"].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                r = _crop_around(m["damage"] & near & ~cv2.erode(m["damage"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool), rgb)
                txt = f"s{s} {key} ({info['material']}) usable {info['usable_fraction']:.2f}"
            if r is not None:
                c = cv2.resize(r[0], (256, 256), interpolation=cv2.INTER_NEAREST)
                _label(c, txt, (4, 14), 0.38)
                got[key] = c
        s += 1
    out = [cv2.copyMakeBorder(got[k], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for k in want if k in got]
    cv2.imwrite(path, np.hstack(out)[..., ::-1])


def check256(path, seeds, size=512):
    cells = []
    for s in seeds:
        rgb, m, info = generate(s, size)
        g = np.asarray(Image.fromarray(rgb).convert("L").resize((256, 256), Image.BILINEAR))
        lab = np.zeros((size, size), np.uint8)
        lab[m["tile"]] = 128      # damaged zone (tile & ~usable) stays 128
        lab[m["usable"]] = 255
        lab = np.asarray(Image.fromarray(lab).resize((256, 256), Image.NEAREST))
        c = np.hstack([g, lab])
        cv2.putText(c, f"s{s} {info['usable_fraction']:.2f} {info['damage_types']}", (262, 16), FONT, 0.45, 60, 1)
        cells.append(cv2.copyMakeBorder(c, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255))
    rows = [np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]
    cv2.imwrite(path, np.vstack(rows))


def overlay(path, seeds, size=512):
    rows = []
    k3 = np.ones((3, 3), np.uint8)
    for s in seeds:
        rgb, m, info = generate(s, size)
        o = rgb.copy()
        for mk, col in (("tile", (0, 255, 0)), ("usable", (255, 230, 0)), ("damage", (255, 0, 0))):
            mm = m[mk].astype(np.uint8)
            o[(mm - cv2.erode(mm, k3)) > 0] = col
        ys, xs = np.nonzero(m["damage"] if m["damage"].any() else m["tile"])
        i = len(ys) // 3
        y, x = int(np.clip(ys[i], 43, size - 43)), int(np.clip(xs[i], 43, size - 43))
        crop = cv2.resize(o[y - 42:y + 43, x - 42:x + 43], (size, size), interpolation=cv2.INTER_NEAREST)
        rows.append(np.hstack([rgb, o, crop]))
    cv2.imwrite(path, np.vstack(rows)[..., ::-1])


def _stat(s):
    cv2.setNumThreads(1)
    t0 = time.time()
    _, m, info = generate(20000 + s)
    return time.time() - t0, info["usable_fraction"], info["damage_types"]


def stats(n):
    from multiprocessing import Pool
    ts = [_stat(s)[0] for s in range(20)]
    print(f"single-process: {np.mean(ts):.3f}s/img (median {np.median(ts):.3f}, max {np.max(ts):.3f})")
    t1 = time.time()
    with Pool(8) as p:
        res = p.map(_stat, range(n))
    print(f"pool(8): {(time.time() - t1) / n:.3f}s/img wall")
    uf = np.array([r[1] for r in res])
    kinds = np.array([r[2] for r in res])
    dmg = uf[kinds != "none"]
    edges = [0, .3, .4, .5, .6, .7, .8, .85, .9, .95, .999, 1.001]
    h, _ = np.histogram(uf, bins=edges)
    for i in range(len(h)):
        lab = "1.0 (pristine-like)" if edges[i] >= .999 else f"{edges[i]:.2f}-{edges[i + 1]:.2f}"
        print(f"  {lab:>20s}: {h[i]:4d} " + "#" * int(h[i] // 2))
    thr = CFG["approve_threshold"]
    print(f"pristine {np.mean(kinds == 'none'):.3f} | <0.5 {np.mean(uf < .5):.3f} <0.7 {np.mean(uf < .7):.3f} "
          f"<0.85 {np.mean(uf < .85):.3f} <0.95 {np.mean(uf < .95):.3f}")
    print(f"APPROVE (>= {thr}) {np.mean(uf >= thr):.3f} | within +-0.05 of {thr}: {np.mean(np.abs(uf - thr) <= 0.05):.3f}"
          f" | damaged tiles: min {dmg.min():.2f} median {np.median(dmg):.2f}, >= {thr}: {np.mean(dmg >= thr):.3f}")
    for k in sorted(set(kinds)):
        u = uf[kinds == k]
        print(f"  {k:12s} n={len(u):4d} usable median {np.median(u):.2f} p10 {np.percentile(u, 10):.2f} p90 {np.percentile(u, 90):.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    if cmd == "sheet":
        sheet(sys.argv[2], seeds or list(range(12)))
    elif cmd == "zoom":
        zoom(sys.argv[2])
    elif cmd == "check256":
        check256(sys.argv[2], seeds or list(range(6)))
    elif cmd == "overlay":
        overlay(sys.argv[2], seeds or [0, 4])
    elif cmd == "stats":
        stats(int(sys.argv[2]))
