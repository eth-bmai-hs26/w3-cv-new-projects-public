"""SQLite storage: lots, inspections, audit trail, retraining feedback and settings."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS lots (
    id            TEXT PRIMARY KEY,
    reference     TEXT,                 -- supplier's delivery note / shipment number
    supplier      TEXT NOT NULL,
    tile_type     TEXT NOT NULL,
    received_date TEXT NOT NULL,        -- YYYY-MM-DD
    status        TEXT NOT NULL DEFAULT 'open',   -- open | accepted | rejected
    decided_by    TEXT,
    decided_at    TEXT,
    decision_note TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inspections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id          TEXT NOT NULL REFERENCES lots(id),
    created_at      TEXT NOT NULL,
    filename        TEXT,
    photo_path      TEXT,
    overlay_path    TEXT,
    mask_path       TEXT,
    usable_fraction REAL,               -- NULL when no tile was found
    tile_share      REAL,
    mask_certainty  REAL,
    confidence      REAL,
    margin          REAL,
    verdict_auto    TEXT NOT NULL,      -- what the model + rules said
    verdict_final   TEXT NOT NULL,      -- after a human override (= auto otherwise)
    reason          TEXT,
    error           TEXT,
    threshold       REAL,
    review_band     REAL,
    value           REAL DEFAULT 0,
    model_name      TEXT,
    model_version   TEXT,
    model_kind      TEXT,
    t_preprocess    REAL, t_inference REAL, t_postprocess REAL, t_total REAL,
    reviewed_by     TEXT,
    reviewed_at     TEXT,
    review_note     TEXT,
    source          TEXT DEFAULT 'upload',
    gt_usable_fraction REAL             -- ground truth, when the photo came from a labelled dataset
);
CREATE INDEX IF NOT EXISTS ix_insp_created ON inspections(created_at);
CREATE INDEX IF NOT EXISTS ix_insp_lot ON inspections(lot_id);
CREATE INDEX IF NOT EXISTS ix_insp_final ON inspections(verdict_final);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,
    entity    TEXT NOT NULL,
    entity_id TEXT,
    detail    TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id   INTEGER NOT NULL REFERENCES inspections(id),
    ts              TEXT NOT NULL,
    inspector       TEXT NOT NULL,
    model_verdict   TEXT NOT NULL,
    human_verdict   TEXT NOT NULL,
    usable_fraction REAL,
    note            TEXT,
    photo_path      TEXT,
    mask_path       TEXT,
    model_version   TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        c = self._conn()
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)
        c.commit()

    def _conn(self) -> sqlite3.Connection:
        """One long-lived connection per thread. Opening and closing a connection per query
        made concurrent requests stall inside sqlite3.connect()/close() under load."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def connect(self):
        """A transaction on this thread's connection: committed on success, rolled back on error."""
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    def query(self, sql, args=()) -> list[dict]:
        with self.connect() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def one(self, sql, args=()) -> dict | None:
        rows = self.query(sql, args)
        return rows[0] if rows else None

    def execute(self, sql, args=()) -> int:
        with self.connect() as c:
            cur = c.execute(sql, args)
            return cur.lastrowid

    def wipe(self):
        with self.connect() as c:
            for t in ("feedback", "audit_log", "inspections", "lots"):
                c.execute(f"DELETE FROM {t}")
            c.execute("DELETE FROM sqlite_sequence")

    # ── settings ─────────────────────────────────────────────────────────────
    def get_setting(self, key, default=None):
        row = self.one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set_setting(self, key, value):
        self.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, json.dumps(value)))

    # ── audit ────────────────────────────────────────────────────────────────
    def audit(self, actor, action, entity, entity_id=None, detail=None, ts=None, conn=None):
        args = (ts or now(), actor or "unknown", action, entity, None if entity_id is None else str(entity_id),
                json.dumps(detail) if detail is not None else None)
        sql = "INSERT INTO audit_log(ts, actor, action, entity, entity_id, detail) VALUES(?,?,?,?,?,?)"
        if conn is not None:
            conn.execute(sql, args)
        else:
            self.execute(sql, args)

    # ── lots ─────────────────────────────────────────────────────────────────
    def next_lot_id(self, received_date: str) -> str:
        d = received_date.replace("-", "")[2:]          # YYMMDD
        n = self.one("SELECT COUNT(*) AS n FROM lots WHERE id LIKE ?", (f"L{d}-%",))["n"]
        return f"L{d}-{n + 1:02d}"

    def insert_lot(self, lot: dict, conn=None) -> str:
        cols = ("id", "reference", "supplier", "tile_type", "received_date", "status",
                "decided_by", "decided_at", "decision_note", "created_at")
        row = {c: lot.get(c) for c in cols}
        row["status"] = row["status"] or "open"
        row["created_at"] = row["created_at"] or now()
        sql = f"INSERT INTO lots({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
        if conn is not None:
            conn.execute(sql, tuple(row.values()))
        else:
            self.execute(sql, tuple(row.values()))
        return row["id"]

    # ── inspections ──────────────────────────────────────────────────────────
    INSPECTION_COLS = (
        "lot_id", "created_at", "filename", "photo_path", "overlay_path", "mask_path", "usable_fraction",
        "tile_share", "mask_certainty", "confidence", "margin", "verdict_auto", "verdict_final", "reason",
        "error", "threshold", "review_band", "value", "model_name", "model_version", "model_kind",
        "t_preprocess", "t_inference", "t_postprocess", "t_total", "reviewed_by", "reviewed_at",
        "review_note", "source", "gt_usable_fraction",
    )

    def insert_inspection(self, rec: dict, conn=None) -> int:
        cols = self.INSPECTION_COLS
        sql = f"INSERT INTO inspections({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
        vals = tuple(rec.get(c) for c in cols)
        if conn is not None:
            return conn.execute(sql, vals).lastrowid
        return self.execute(sql, vals)
