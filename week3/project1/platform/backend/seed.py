"""
Fill the database with realistic history: a few hundred inspections over the
last 30 days, lots from several suppliers whose quality differs, a handful of
human overrides (audit trail + retraining feedback) and an open review queue.

Every photo is really inspected: the current model (or DEMO MODE) runs once per
distinct image from the configured folder, then the results are spread over
lots and days. If the folder holds no photos, placeholder tiles are generated.

    python -m backend seed --tiles 360 --days 30 [--images FOLDER]
"""

from __future__ import annotations

import math
import os
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config, imaging, rules
from .core import Platform
from .db import now

# name, quality (-1 poor .. +1 excellent), share of deliveries, usual tile types
SUPPLIERS = [
    ("Bauteilbörse Basel", 0.9, 0.22, ["Terracotta 30×30", "Encaustic cement 20×20"]),
    ("Rückbau Zürich AG", 0.5, 0.26, ["Porcelain 30×30", "Glazed ceramic 20×20"]),
    ("Keramik Kreislauf Ost", 0.1, 0.20, ["Glazed ceramic 20×20", "Terracotta 30×30"]),
    ("Altbau Rheintal GmbH", -0.3, 0.17, ["Slate 30×30", "Porcelain 30×30"]),
    ("Demolizioni Ticino SA", -0.8, 0.15, ["Terracotta 30×30", "Glazed ceramic 20×20"]),
]
INSPECTORS = ["M. Keller", "A. Rossi", "S. Brunner", "L. Meier"]
NOTES_APPROVE = ["Crack is only in the glaze, body intact", "Chip within cutting margin, sellable",
                 "Stain, not damage", "Usable area fine after trimming edge"]
NOTES_REJECT = ["Hairline crack runs through the body", "Corner missing, not sellable",
                "Crack continues on the back", "Glaze delaminated"]


def _photos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in config.VALID_EXT and not p.name.startswith("."))


def _ground_truth(folder: Path) -> dict:
    csv = folder.parent / "ground_truth.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    col = "usable_fraction" if "usable_fraction" in df else ("coverage_pct" if "coverage_pct" in df else None)
    return dict(zip(df["filename"], df[col])) if col else {}


def ensure_images(plat: Platform, image_dir: str | None, n: int = 240) -> Path:
    folder = Path(image_dir or plat.image_source())
    if config.has_photos(folder):
        return folder
    from .sample_tiles import write_folder
    out = plat.data_dir / "sample_tiles"
    print(f"no photos in {folder}; generating {n} placeholder tiles in {out}")
    write_folder(str(out), n=n, seed=7)
    return out / "original"


def _day_weights(days: int) -> list[tuple[date, float]]:
    today = date.today()
    out = []
    for k in range(days):
        d = today - timedelta(days=days - 1 - k)
        w = {5: 0.45, 6: 0.15}.get(d.weekday(), 1.0) * (1 + 0.15 * math.sin(k / 3))
        if d == today:
            w = max(w, 0.6)                        # the station is running today
        out.append((d, w))
    return out


def seed(plat: Platform, n_tiles: int = 360, days: int = 30, image_dir: str | None = None, wipe: bool = True,
         pool: int = 240, seed_value: int = 11, actor: str = "seed", verbose: bool = True) -> dict:
    rng = random.Random(seed_value)
    nrng = np.random.default_rng(seed_value)
    folder = ensure_images(plat, image_dir)
    photos = _photos(folder)
    if len(photos) > pool:
        photos = [photos[i] for i in sorted(nrng.choice(len(photos), pool, replace=False))]
    gt = _ground_truth(folder)
    if wipe:
        plat.db.wipe()
        for p in plat.media_dir.glob("*"):
            if p.is_dir():
                for f in p.glob("*"):
                    f.unlink()
                p.rmdir()

    # 1. inspect every distinct photo once
    cfg = plat.rule_config()
    inspected = []
    for k, p in enumerate(photos):
        img = Image.open(p).convert("RGB")
        pred = plat.model.predict(img)
        files = imaging.store(plat.media_dir, img, pred.classes, datetime.now())
        inspected.append((p, pred, files))
        if verbose and (k + 1) % 40 == 0:
            print(f"  inspected {k + 1}/{len(photos)} photos")
    usable = np.array([pr.usable_fraction if pr.usable_fraction is not None else 0.8 for _, pr, _ in inspected])

    # 2. per-supplier sampling weights: good suppliers send more intact tiles
    def weights(q):
        w = np.exp(q * 9 * (usable - 0.8))
        return w / w.sum()

    sup_w = {name: weights(q) for name, q, _, _ in SUPPLIERS}

    # 3. lots over the days, then tiles inside each lot
    day_w = _day_weights(days)
    total_w = sum(w for _, w in day_w)
    lots_made, tiles_made = 0, 0
    today = date.today()
    now_dt = datetime.now()
    per_day: dict = {}
    for d, w in day_w:
        with plat.db.connect() as c:          # one short transaction per day
            n_day = int(round(n_tiles * w / total_w))
            clock = datetime.combine(d, time(6, 30)) + timedelta(minutes=rng.randint(0, 40))
            while n_day > 0:
                name, q, _, types = rng.choices(SUPPLIERS, weights=[s[2] for s in SUPPLIERS])[0]
                size = min(n_day, rng.randint(10, 28))
                n_day -= size
                per_day[d] = per_day.get(d, 0) + 1
                lot_id = f"L{d.strftime('%y%m%d')}-{per_day[d]:02d}"
                age = (today - d).days
                plat.db.insert_lot({"id": lot_id, "reference": f"DN-{rng.randint(20000, 99999)}", "supplier": name,
                                    "tile_type": rng.choice(types), "received_date": d.isoformat(),
                                    "created_at": clock.isoformat(timespec="seconds")}, conn=c)
                lots_made += 1
                idx = nrng.choice(len(inspected), size=size, p=sup_w[name])
                counts = {v: 0 for v in rules.VERDICTS}
                for j in idx:
                    clock += timedelta(seconds=rng.randint(25, 140))
                    if clock > now_dt:
                        break
                    p, pred, files = inspected[j]
                    rec = plat.build_record(pred, files, lot_id, p.name, "seed", clock, gt.get(p.name), cfg)
                    # older borderline tiles were already handled by an inspector
                    if rec["verdict_auto"] == rules.REVIEW and age >= 3:
                        uf = rec["usable_fraction"] or 0
                        ok = uf + rng.gauss(0, 0.04) >= cfg.threshold
                        verdict = rules.APPROVE if ok else rules.REJECT
                        who = rng.choice(INSPECTORS)
                        ts = (clock + timedelta(minutes=rng.randint(20, 300))).isoformat(timespec="seconds")
                        note = rng.choice(NOTES_APPROVE if ok else NOTES_REJECT)
                        rec.update(verdict_final=verdict, value=rules.value_for(verdict, cfg),
                                   reviewed_by=who, reviewed_at=ts, review_note=note)
                        iid = plat.db.insert_inspection(rec, conn=c)
                        c.execute("""INSERT INTO feedback(inspection_id, ts, inspector, model_verdict, human_verdict,
                                     usable_fraction, note, photo_path, mask_path, model_version)
                                     VALUES(?,?,?,?,?,?,?,?,?,?)""",
                                  (iid, ts, who, rules.REVIEW, verdict, rec["usable_fraction"], note,
                                   rec["photo_path"], rec["mask_path"], rec["model_version"]))
                        plat.db.audit(who, "inspection.review", "inspection", iid,
                                      {"from": "REVIEW", "to": verdict, "model": "REVIEW", "note": note}, ts=ts, conn=c)
                    else:
                        plat.db.insert_inspection(rec, conn=c)
                    counts[rec["verdict_final"]] += 1
                    tiles_made += 1
                # lots older than two days have been accepted or rejected as a whole
                if age >= 3 and counts[rules.REVIEW] == 0 and sum(counts.values()):
                    rate = counts[rules.APPROVE] / sum(counts.values())
                    status = "accepted" if rate >= 0.45 else "rejected"
                    who = rng.choice(INSPECTORS)
                    ts = (clock + timedelta(hours=rng.randint(2, 20))).isoformat(timespec="seconds")
                    note = None if status == "accepted" else "Too few sellable tiles; returned to supplier"
                    c.execute("UPDATE lots SET status=?, decided_by=?, decided_at=?, decision_note=? WHERE id=?",
                              (status, who, ts, note, lot_id))
                    plat.db.audit(who, f"lot.{'accept' if status == 'accepted' else 'reject'}", "lot", lot_id,
                                  {"from": "open", "to": status, "note": note}, ts=ts, conn=c)
                clock += timedelta(minutes=rng.randint(10, 50))
                if clock.time() > time(16, 30):
                    break
    plat.db.audit(actor, "demo.seed", "system", None,
                  {"tiles": tiles_made, "lots": lots_made, "images": str(folder), "model": plat.model.info.get("name")},
                  ts=now())
    plat.db.set_setting("image_source", str(folder))
    out = {"tiles": tiles_made, "lots": lots_made, "photos": len(photos), "image_dir": str(folder),
           "model": plat.model.info.get("name")}
    if verbose:
        print(f"seeded {tiles_made} inspections in {lots_made} lots from {len(photos)} photos ({folder})")
    return out
