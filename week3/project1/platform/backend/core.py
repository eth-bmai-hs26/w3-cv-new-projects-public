"""
The platform's business logic, independent of HTTP: inspecting photos, the
review queue, lot decisions, dashboard numbers and reports. The FastAPI app
(app.py) and the seed script both call into `Platform`.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from PIL import Image

from . import config, imaging, rules
from .db import Database, now
from .model_service import ModelService


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


def _pct(x):
    return None if x is None else round(100 * x, 1)


class Platform:
    def __init__(self, data_dir: Path | None = None, checkpoint: str | None = None, device: str | None = None,
                 load_model: bool = True):
        self.data_dir = Path(data_dir) if data_dir else config.data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir = self.data_dir / "uploads"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.data_dir / "platform.db")

        model_cfg = self.model_settings()
        if checkpoint is None:
            checkpoint = config.env_checkpoint()
        if checkpoint is not None:                    # explicit argument / env var wins
            model_cfg["checkpoint"] = checkpoint
        if device:
            model_cfg["device"] = device
        self.db.set_setting("model", model_cfg)
        self.model = ModelService(model_cfg["checkpoint"] if load_model else None, model_cfg["device"])

    # ── settings ─────────────────────────────────────────────────────────────
    def rule_config(self) -> rules.RuleConfig:
        return rules.RuleConfig.from_dict(self.db.get_setting("rules", {}))

    def model_settings(self) -> dict:
        d = self.db.get_setting("model", None)
        if d is None:
            d = {"checkpoint": config.env_checkpoint() or str(config.DEFAULT_CHECKPOINT), "device": config.env_device()}
        return {**d, "checkpoint": config.relocate(d.get("checkpoint"))}

    def image_source(self) -> str:
        return config.relocate(self.db.get_setting("image_source", None)) or str(config.default_image_dir())

    def settings(self) -> dict:
        return {
            "rules": self.rule_config().to_dict(),
            "model": self.model_settings(),
            "image_source": self.image_source(),
            "defaults": rules.RuleConfig().to_dict(),
            "default_checkpoint": str(config.DEFAULT_CHECKPOINT),
        }

    def update_settings(self, patch: dict, actor: str = "settings") -> dict:
        before = self.settings()
        if "rules" in patch and patch["rules"] is not None:
            merged = {**self.rule_config().to_dict(), **patch["rules"]}
            cfg = rules.RuleConfig.from_dict(merged)          # raises ValueError on bad input
            self.db.set_setting("rules", cfg.to_dict())
        if patch.get("image_source") is not None:
            self.db.set_setting("image_source", str(patch["image_source"]))
        model_changed = False
        if patch.get("model") is not None:
            m = {**self.model_settings(), **{k: v for k, v in patch["model"].items() if v is not None}}
            model_changed = m != self.model_settings()
            self.db.set_setting("model", m)
        after = self.settings()
        changes = {k: {"from": before[k], "to": after[k]} for k in ("rules", "model", "image_source") if before[k] != after[k]}
        if changes:
            self.db.audit(actor, "settings.update", "settings", None, changes)
        if model_changed:
            self.reload_model(actor=actor)
        return after

    def reload_model(self, checkpoint: str | None = None, actor: str = "settings") -> dict:
        m = self.model_settings()
        if checkpoint is not None:
            m["checkpoint"] = checkpoint
            self.db.set_setting("model", m)
        info = self.model.load(m.get("checkpoint"), m.get("device"))
        self.db.audit(actor, "model.load", "model", info.get("version"),
                      {"checkpoint": m.get("checkpoint"), "mode": info["mode"], "warning": info.get("warning")})
        return info

    # ── lots ─────────────────────────────────────────────────────────────────
    def create_lot(self, supplier: str, tile_type: str, received_date: str | None = None,
                   reference: str | None = None, actor: str = "inspector") -> dict:
        supplier, tile_type = (supplier or "").strip(), (tile_type or "").strip()
        if not supplier or not tile_type:
            raise ValueError("supplier and tile type are required")
        received_date = received_date or date.today().isoformat()
        date.fromisoformat(received_date)
        lot_id = self.db.next_lot_id(received_date)
        self.db.insert_lot({"id": lot_id, "reference": reference, "supplier": supplier,
                            "tile_type": tile_type, "received_date": received_date})
        self.db.audit(actor, "lot.create", "lot", lot_id, {"supplier": supplier, "tile_type": tile_type})
        return self.lot(lot_id)

    def _lot_stats_sql(self, where="1=1"):
        return f"""
            SELECT l.*,
                   COUNT(i.id) AS n,
                   SUM(i.verdict_final='APPROVE') AS approved,
                   SUM(i.verdict_final='REJECT')  AS rejected,
                   SUM(i.verdict_final='REVIEW')  AS review,
                   AVG(i.usable_fraction)         AS avg_usable,
                   COALESCE(SUM(i.value), 0)      AS value,
                   MAX(i.created_at)              AS last_inspection
            FROM lots l LEFT JOIN inspections i ON i.lot_id = l.id
            WHERE {where}
            GROUP BY l.id
        """

    def _decorate_lot(self, r: dict) -> dict:
        cfg = self.rule_config()
        n = r["n"] or 0
        for k in ("approved", "rejected", "review"):
            r[k] = r[k] or 0
        r["approval_rate"] = round(r["approved"] / n, 4) if n else None
        r["avg_usable"] = None if r["avg_usable"] is None else round(r["avg_usable"], 4)
        r["value"] = round(r["value"] or 0, 2)
        r["errors_avoided"] = round(r["rejected"] * cfg.cost_bad_tile_shipped, 2)
        return r

    def lots(self, status: str | None = None, q: str | None = None) -> list[dict]:
        where, args = ["1=1"], []
        if status:
            where.append("l.status = ?"); args.append(status)
        if q:
            where.append("(l.id LIKE ? OR l.supplier LIKE ? OR l.tile_type LIKE ? OR l.reference LIKE ?)")
            args += [f"%{q}%"] * 4
        rows = self.db.query(self._lot_stats_sql(" AND ".join(where)) + " ORDER BY l.received_date DESC, l.id DESC", args)
        return [self._decorate_lot(r) for r in rows]

    def lot(self, lot_id: str) -> dict:
        rows = self.db.query(self._lot_stats_sql("l.id = ?"), (lot_id,))
        if not rows:
            raise NotFound(f"lot {lot_id} not found")
        return self._decorate_lot(rows[0])

    def decide_lot(self, lot_id: str, decision: str, actor: str, note: str | None = None, force: bool = False) -> dict:
        lot = self.lot(lot_id)
        status = {"accept": "accepted", "reject": "rejected", "reopen": "open"}.get(decision)
        if not status:
            raise ValueError("decision must be accept, reject or reopen")
        if status == "accepted" and lot["review"] and not force:
            raise Conflict(f"{lot['review']} tile(s) in this lot still need review")
        self.db.execute("UPDATE lots SET status=?, decided_by=?, decided_at=?, decision_note=? WHERE id=?",
                        (status, actor if status != "open" else None, now() if status != "open" else None, note, lot_id))
        self.db.audit(actor, f"lot.{decision}", "lot", lot_id, {"from": lot["status"], "to": status, "note": note})
        return self.lot(lot_id)

    # ── inspections ──────────────────────────────────────────────────────────
    def build_record(self, pred, files: dict, lot_id: str, filename=None, source="upload", when=None,
                     gt_usable_fraction=None, cfg: rules.RuleConfig | None = None) -> dict:
        """Prediction + current rule -> an inspections row (not yet stored)."""
        cfg = cfg or self.rule_config()
        v = rules.decide(pred.usable_fraction, pred.mask_certainty, cfg, pred.error)
        info = self.model.info
        return {
            "lot_id": lot_id, "created_at": (when or datetime.now()).isoformat(timespec="seconds"),
            "filename": filename, **files,
            "usable_fraction": None if pred.usable_fraction is None else round(pred.usable_fraction, 4),
            "tile_share": round(pred.tile_share, 4), "mask_certainty": pred.mask_certainty,
            "confidence": v.confidence, "margin": v.decision_margin,
            "verdict_auto": v.verdict, "verdict_final": v.verdict, "reason": v.reason, "error": pred.error,
            "threshold": cfg.threshold, "review_band": cfg.review_band, "value": v.value,
            "model_name": info.get("name"), "model_version": info.get("version"), "model_kind": info.get("kind"),
            "t_preprocess": pred.timings["preprocess"], "t_inference": pred.timings["inference"],
            "t_postprocess": pred.timings["postprocess"], "t_total": pred.timings["total"],
            "source": source, "gt_usable_fraction": gt_usable_fraction,
        }

    def inspect(self, image: Image.Image, lot_id: str, filename: str | None = None, source: str = "upload",
                when: datetime | None = None, gt_usable_fraction: float | None = None, store: bool = True,
                conn=None) -> dict:
        if conn is None:
            lot = self.lot(lot_id)
            if lot["status"] != "open":
                raise Conflict(f"lot {lot_id} is {lot['status']}; reopen it to add tiles")
        pred = self.model.predict(image)
        files = imaging.store(self.media_dir, image, pred.classes, when) if store else {}
        rec = self.build_record(pred, files, lot_id, filename, source, when, gt_usable_fraction)
        iid = self.db.insert_inspection(rec, conn=conn)
        if conn is not None:
            return {**rec, "id": iid}
        out = self.inspection(iid)
        out["warnings"] = pred.warnings
        return out

    def _decorate(self, r: dict) -> dict:
        for k in ("photo", "overlay", "mask"):
            p = r.get(f"{k}_path")
            r[f"{k}_url"] = f"/media/{p}" if p else None
        r["usable_pct"] = _pct(r.get("usable_fraction"))
        r["overridden"] = bool(r.get("reviewed_by")) and r["verdict_final"] != r["verdict_auto"]
        return r

    def inspection(self, iid: int) -> dict:
        r = self.db.one("""SELECT i.*, l.supplier, l.tile_type, l.status AS lot_status
                           FROM inspections i JOIN lots l ON l.id = i.lot_id WHERE i.id = ?""", (iid,))
        if not r:
            raise NotFound(f"inspection {iid} not found")
        return self._decorate(r)

    def inspections(self, lot_id=None, verdict=None, limit=50, offset=0, reviewed=None) -> dict:
        where, args = ["1=1"], []
        if lot_id:
            where.append("i.lot_id = ?"); args.append(lot_id)
        if verdict:
            where.append("i.verdict_final = ?"); args.append(verdict)
        if reviewed is not None:
            where.append("i.reviewed_by IS NOT NULL" if reviewed else "i.reviewed_by IS NULL")
        w = " AND ".join(where)
        total = self.db.one(f"SELECT COUNT(*) AS n FROM inspections i WHERE {w}", args)["n"]
        rows = self.db.query(f"""SELECT i.*, l.supplier, l.tile_type, l.status AS lot_status
                                 FROM inspections i JOIN lots l ON l.id = i.lot_id WHERE {w}
                                 ORDER BY i.created_at DESC, i.id DESC LIMIT ? OFFSET ?""", args + [limit, offset])
        return {"total": total, "items": [self._decorate(r) for r in rows]}

    def review_queue(self, limit=100) -> dict:
        rows = self.db.query("""SELECT i.*, l.supplier, l.tile_type, l.status AS lot_status
                                FROM inspections i JOIN lots l ON l.id = i.lot_id
                                WHERE i.verdict_final = 'REVIEW'
                                ORDER BY i.confidence ASC, i.created_at ASC LIMIT ?""", (limit,))
        total = self.db.one("SELECT COUNT(*) AS n FROM inspections WHERE verdict_final='REVIEW'")["n"]
        return {"total": total, "items": [self._decorate(r) for r in rows]}

    def review(self, iid: int, verdict: str, inspector: str, note: str | None = None) -> dict:
        verdict = (verdict or "").upper()
        if verdict not in (rules.APPROVE, rules.REJECT):
            raise ValueError("verdict must be APPROVE or REJECT")
        inspector = (inspector or "").strip()
        if not inspector:
            raise ValueError("inspector name is required for the audit trail")
        rec = self.inspection(iid)
        cfg = self.rule_config()
        ts = now()
        with self.db.connect() as c:
            c.execute("""UPDATE inspections SET verdict_final=?, value=?, reviewed_by=?, reviewed_at=?, review_note=?
                         WHERE id=?""", (verdict, rules.value_for(verdict, cfg), inspector, ts, note, iid))
            c.execute("""INSERT INTO feedback(inspection_id, ts, inspector, model_verdict, human_verdict,
                         usable_fraction, note, photo_path, mask_path, model_version)
                         VALUES(?,?,?,?,?,?,?,?,?,?)""",
                      (iid, ts, inspector, rec["verdict_auto"], verdict, rec["usable_fraction"], note,
                       rec["photo_path"], rec["mask_path"], rec["model_version"]))
            self.db.audit(inspector, "inspection.review", "inspection", iid,
                          {"from": rec["verdict_final"], "to": verdict, "model": rec["verdict_auto"], "note": note},
                          ts=ts, conn=c)
        return self.inspection(iid)

    def rescore_open(self, actor: str = "settings") -> dict:
        """Re-apply the current rule to un-reviewed tiles in open lots (the stored usable share is reused)."""
        cfg = self.rule_config()
        rows = self.db.query("""SELECT i.id, i.usable_fraction, i.mask_certainty, i.error, i.verdict_final
                                FROM inspections i JOIN lots l ON l.id = i.lot_id
                                WHERE l.status='open' AND i.reviewed_by IS NULL""")
        changed = 0
        with self.db.connect() as c:
            for r in rows:
                v = rules.decide(r["usable_fraction"], r["mask_certainty"] or 0, cfg, r["error"])
                if v.verdict != r["verdict_final"]:
                    changed += 1
                c.execute("""UPDATE inspections SET verdict_auto=?, verdict_final=?, reason=?, confidence=?, margin=?,
                             value=?, threshold=?, review_band=? WHERE id=?""",
                          (v.verdict, v.verdict, v.reason, v.confidence, v.decision_margin, v.value,
                           cfg.threshold, cfg.review_band, r["id"]))
            self.db.audit(actor, "inspection.rescore", "settings", None,
                          {"tiles": len(rows), "changed": changed, "threshold": cfg.threshold}, conn=c)
        return {"tiles": len(rows), "changed": changed}

    def preview_rules(self, patch: dict, days: int = 30) -> dict:
        """What-if: verdict mix and money over the last `days` under the current vs a proposed rule."""
        cur = self.rule_config()
        new = rules.RuleConfig.from_dict({**cur.to_dict(), **(patch or {})})
        since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        rows = self.db.query("SELECT usable_fraction, mask_certainty, error FROM inspections WHERE created_at >= ?", (since,))
        out = {}
        for name, cfg in (("current", cur), ("proposed", new)):
            counts = {k: 0 for k in rules.VERDICTS}
            for r in rows:
                counts[rules.decide(r["usable_fraction"], r["mask_certainty"] or 0, cfg, r["error"]).verdict] += 1
            n = max(1, len(rows))
            out[name] = {"counts": counts, "approval_rate": round(counts["APPROVE"] / n, 4),
                         "review_rate": round(counts["REVIEW"] / n, 4),
                         **rules.business_summary(counts, cfg),
                         "review_cost": round(counts["REVIEW"] * cfg.cost_manual_check, 2)}
        out["tiles"] = len(rows)
        out["days"] = days
        return out

    # ── dashboard ────────────────────────────────────────────────────────────
    def dashboard(self, days: int = 30) -> dict:
        cfg = self.rule_config()
        today = date.today()
        start = today - timedelta(days=days - 1)
        week_start = today - timedelta(days=6)
        prev_week_start = today - timedelta(days=13)

        def count(where, args=()):
            return self.db.one(f"SELECT COUNT(*) AS n FROM inspections WHERE {where}", args)["n"]

        def mix(since, until=None):
            args = [since.isoformat()]
            w = "created_at >= ?"
            if until:
                w += " AND created_at < ?"; args.append(until.isoformat())
            r = self.db.one(f"""SELECT COUNT(*) AS n, SUM(verdict_final='APPROVE') AS a, SUM(verdict_final='REJECT') AS r,
                                SUM(verdict_final='REVIEW') AS v, AVG(usable_fraction) AS u, COALESCE(SUM(value),0) AS val,
                                SUM(reviewed_by IS NULL AND verdict_final != 'REVIEW') AS auto
                                FROM inspections WHERE {w}""", args)
            n = r["n"] or 0
            return {"n": n, "APPROVE": r["a"] or 0, "REJECT": r["r"] or 0, "REVIEW": r["v"] or 0,
                    "avg_usable": None if r["u"] is None else round(r["u"], 4), "value": round(r["val"], 2),
                    "auto_decided": r["auto"] or 0,
                    "approval_rate": round((r["a"] or 0) / n, 4) if n else None}

        period = mix(start)
        week = mix(week_start)
        prev_week = mix(prev_week_start, week_start)
        money = rules.business_summary(period, cfg)
        kpis = {
            "today": count("created_at >= ?", (today.isoformat(),)),
            "yesterday": count("created_at >= ? AND created_at < ?", ((today - timedelta(days=1)).isoformat(), today.isoformat())),
            "week": week["n"], "prev_week": prev_week["n"],
            "approval_rate": week["approval_rate"], "prev_approval_rate": prev_week["approval_rate"],
            "avg_usable": week["avg_usable"], "prev_avg_usable": prev_week["avg_usable"],
            "review_queue": count("verdict_final = 'REVIEW'"),
            "oldest_review": (self.db.one("SELECT MIN(created_at) AS t FROM inspections WHERE verdict_final='REVIEW'") or {}).get("t"),
            "recovered_value": round(period["value"], 2),
            "errors_avoided": money["errors_avoided"],
            "labour_saved": money["labour_saved"],
            "period_tiles": period["n"],
            "automation_rate": round(period["auto_decided"] / period["n"], 4) if period["n"] else None,
            "currency": cfg.currency,
            "avg_ms": (lambda r: None if r["t"] is None else round(r["t"], 1))(
                self.db.one("SELECT AVG(t_total) AS t FROM inspections WHERE created_at >= ?", (start.isoformat(),))),
            "overrides": self.db.one("SELECT COUNT(*) AS n FROM inspections WHERE reviewed_by IS NOT NULL AND created_at >= ?",
                                     (start.isoformat(),))["n"],
        }

        daily = self.db.query("""SELECT substr(created_at, 1, 10) AS day,
                                        SUM(verdict_final='APPROVE') AS approve, SUM(verdict_final='REVIEW') AS review,
                                        SUM(verdict_final='REJECT') AS reject, COUNT(*) AS n, AVG(usable_fraction) AS avg_usable
                                 FROM inspections WHERE created_at >= ? GROUP BY day ORDER BY day""", (start.isoformat(),))
        by_day = {r["day"]: r for r in daily}
        throughput = []
        for k in range(days):
            d = (start + timedelta(days=k)).isoformat()
            r = by_day.get(d, {})
            throughput.append({"day": d, "approve": r.get("approve") or 0, "review": r.get("review") or 0,
                               "reject": r.get("reject") or 0, "n": r.get("n") or 0,
                               "avg_usable": None if r.get("avg_usable") is None else round(r["avg_usable"], 4)})

        bins = 40
        hist = [{"lo": i / bins, "hi": (i + 1) / bins, "n": 0} for i in range(bins)]
        for r in self.db.query("SELECT usable_fraction AS u FROM inspections WHERE created_at >= ? AND usable_fraction IS NOT NULL",
                               (start.isoformat(),)):
            hist[min(bins - 1, int(r["u"] * bins))]["n"] += 1
        lo, hi = cfg.threshold - cfg.review_band, cfg.threshold + cfg.review_band
        for h in hist:
            mid = (h["lo"] + h["hi"]) / 2
            h["zone"] = "REVIEW" if lo <= mid < hi else ("APPROVE" if mid >= cfg.threshold else "REJECT")

        suppliers = self.db.query("""SELECT l.supplier, COUNT(i.id) AS n, SUM(i.verdict_final='APPROVE') AS approved,
                                            SUM(i.verdict_final='REJECT') AS rejected, SUM(i.verdict_final='REVIEW') AS review,
                                            AVG(i.usable_fraction) AS avg_usable, COALESCE(SUM(i.value),0) AS value,
                                            COUNT(DISTINCT l.id) AS lots
                                     FROM inspections i JOIN lots l ON l.id = i.lot_id
                                     WHERE i.created_at >= ? GROUP BY l.supplier ORDER BY avg_usable DESC""",
                                  (start.isoformat(),))
        for s in suppliers:
            s["approval_rate"] = round((s["approved"] or 0) / s["n"], 4) if s["n"] else None
            s["avg_usable"] = None if s["avg_usable"] is None else round(s["avg_usable"], 4)
            s["value"] = round(s["value"], 2)

        return {
            "kpis": kpis, "throughput": throughput, "histogram": hist, "suppliers": suppliers,
            "mix": {k: period[k] for k in rules.VERDICTS},
            "recent": self.inspections(limit=10)["items"],
            "rules": cfg.to_dict(), "model": self.model.info, "days": days,
        }

    # ── audit / feedback / reports ───────────────────────────────────────────
    def audit_log(self, limit=100, entity=None, entity_id=None) -> list[dict]:
        where, args = ["1=1"], []
        if entity:
            where.append("entity = ?"); args.append(entity)
        if entity_id is not None:
            where.append("entity_id = ?"); args.append(str(entity_id))
        rows = self.db.query(f"SELECT * FROM audit_log WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?", args + [limit])
        for r in rows:
            r["detail"] = json.loads(r["detail"]) if r["detail"] else None
        return rows

    def feedback(self, limit=500) -> list[dict]:
        rows = self.db.query("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["photo_url"] = f"/media/{r['photo_path']}" if r["photo_path"] else None
            r["agrees"] = r["model_verdict"] == r["human_verdict"]
        return rows

    def feedback_csv(self) -> str:
        rows = self.db.query("SELECT * FROM feedback ORDER BY id")
        buf = io.StringIO()
        cols = ["id", "inspection_id", "ts", "inspector", "model_verdict", "human_verdict", "usable_fraction",
                "note", "model_version", "photo_file", "mask_file"]
        w = csv.writer(buf)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["id"], r["inspection_id"], r["ts"], r["inspector"], r["model_verdict"], r["human_verdict"],
                        r["usable_fraction"], r["note"] or "", r["model_version"],
                        str(self.media_dir / r["photo_path"]) if r["photo_path"] else "",
                        str(self.media_dir / r["mask_path"]) if r["mask_path"] else ""])
        return buf.getvalue()

    def lot_report(self, lot_id: str) -> dict:
        lot = self.lot(lot_id)
        items = self.inspections(lot_id=lot_id, limit=100000)["items"]
        items.sort(key=lambda r: r["id"])
        cfg = self.rule_config()
        usable = [r["usable_fraction"] for r in items if r["usable_fraction"] is not None]
        times = [r["t_total"] for r in items if r["t_total"] is not None]
        return {
            "lot": lot, "items": items, "rules": cfg.to_dict(),
            "summary": {
                "tiles": len(items), "approved": lot["approved"], "rejected": lot["rejected"], "review": lot["review"],
                "approval_rate": lot["approval_rate"], "avg_usable": lot["avg_usable"],
                "min_usable": min(usable) if usable else None, "max_usable": max(usable) if usable else None,
                "value": lot["value"], "errors_avoided": lot["errors_avoided"],
                "overrides": sum(1 for r in items if r["reviewed_by"]),
                "avg_ms": round(sum(times) / len(times), 1) if times else None,
                "models": sorted({f"{r['model_name']} ({r['model_version']})" for r in items if r["model_name"]}),
            },
            "audit": self.audit_log(limit=50, entity="lot", entity_id=lot_id),
            "generated_at": now(),
        }

    def lot_csv(self, lot_id: str) -> str:
        rep = self.lot_report(lot_id)
        lot = rep["lot"]
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["# lot", lot["id"], "supplier", lot["supplier"], "tile type", lot["tile_type"],
                    "received", lot["received_date"], "status", lot["status"]])
        w.writerow(["# rule", f"approve if usable >= {rep['rules']['threshold']:.2f}",
                    f"review band +/- {rep['rules']['review_band']:.2f}"])
        w.writerow(["inspection_id", "time", "filename", "usable_pct", "verdict_model", "verdict_final", "confidence",
                    "value_eur", "reviewed_by", "review_note", "reason", "model", "processing_ms"])
        for r in rep["items"]:
            w.writerow([r["id"], r["created_at"], r["filename"] or "", r["usable_pct"] if r["usable_pct"] is not None else "",
                        r["verdict_auto"], r["verdict_final"], r["confidence"], r["value"], r["reviewed_by"] or "",
                        r["review_note"] or "", r["reason"] or "", f"{r['model_name']} {r['model_version']}", r["t_total"]])
        return buf.getvalue()

    def suppliers(self) -> list[str]:
        return [r["supplier"] for r in self.db.query("SELECT DISTINCT supplier FROM lots ORDER BY supplier")]

    def tile_types(self) -> list[str]:
        return [r["tile_type"] for r in self.db.query("SELECT DISTINCT tile_type FROM lots ORDER BY tile_type")]
