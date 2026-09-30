"""
FastAPI app: JSON API under /api, stored photos under /media, and the built
React frontend (frontend/dist) for every other path, so one process serves
the whole platform.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .core import Conflict, NotFound, Platform
from .imaging import open_image

MAX_UPLOAD_MB = 25


class LotIn(BaseModel):
    supplier: str
    tile_type: str
    received_date: Optional[str] = None
    reference: Optional[str] = None
    actor: Optional[str] = "inspector"


class LotDecisionIn(BaseModel):
    decision: str                   # accept | reject | reopen
    actor: str
    note: Optional[str] = None
    force: bool = False


class ReviewIn(BaseModel):
    verdict: str                    # APPROVE | REJECT
    inspector: str
    note: Optional[str] = None


class SettingsIn(BaseModel):
    rules: Optional[dict] = None
    model: Optional[dict] = None
    image_source: Optional[str] = None
    actor: Optional[str] = "settings"
    rescore_open_lots: bool = False


class ReloadIn(BaseModel):
    checkpoint: Optional[str] = None
    actor: Optional[str] = "settings"


class ResetIn(BaseModel):
    tiles: int = 360
    days: int = 30
    actor: Optional[str] = "settings"
    image_dir: Optional[str] = None


def _errors(fn):
    try:
        return fn()
    except NotFound as e:
        raise HTTPException(404, str(e))
    except Conflict as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


def create_app(platform: Platform | None = None, frontend_dist: Path | None = None) -> FastAPI:
    plat = platform or Platform()
    app = FastAPI(title="Tile inspection platform", version="1.0",
                  description="Reclaimed-tile inspection station: U-Net segmentation, verdicts, lots, review queue.")
    app.state.platform = plat

    # ── system ───────────────────────────────────────────────────────────────
    @app.get("/api/health")
    def health():
        return {"ok": True, "model_mode": plat.model.info["mode"], "time": datetime.now().isoformat(timespec="seconds")}

    @app.get("/api/model")
    def model_info():
        return plat.model.info

    @app.post("/api/model/reload")
    def model_reload(body: ReloadIn):
        return plat.reload_model(body.checkpoint, actor=body.actor or "settings")

    @app.post("/api/model/upload")
    def model_upload(file: UploadFile = File(...), actor: str = Form("settings")):
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(file.filename or "model.pt").name)
        if not name.lower().endswith((".pt", ".pth", ".ts")):
            raise HTTPException(422, "upload a .pt / .pth / .ts model file")
        config.UPLOADED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = config.UPLOADED_MODELS_DIR / name
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        info = plat.reload_model(str(dest), actor=actor)
        return info

    # ── settings ─────────────────────────────────────────────────────────────
    @app.get("/api/settings")
    def get_settings():
        return plat.settings()

    @app.put("/api/settings")
    def put_settings(body: SettingsIn):
        out = _errors(lambda: plat.update_settings(body.model_dump(exclude={"actor", "rescore_open_lots"}),
                                                    actor=body.actor or "settings"))
        if body.rescore_open_lots:
            out = {**out, "rescored": plat.rescore_open(actor=body.actor or "settings")}
        return out

    @app.post("/api/settings/preview")
    def preview(rules: dict, days: int = 30):
        return _errors(lambda: plat.preview_rules(rules, days))

    @app.post("/api/admin/reset-demo")
    def reset_demo(body: ResetIn):
        from .seed import seed
        return _errors(lambda: seed(plat, n_tiles=body.tiles, days=body.days, image_dir=body.image_dir,
                                    wipe=True, actor=body.actor or "settings"))

    # ── dashboard ────────────────────────────────────────────────────────────
    @app.get("/api/dashboard")
    def dashboard(days: int = Query(30, ge=1, le=365)):
        return plat.dashboard(days)

    # ── lots ─────────────────────────────────────────────────────────────────
    @app.get("/api/lots")
    def lots(status: Optional[str] = None, q: Optional[str] = None):
        return plat.lots(status, q)

    @app.post("/api/lots", status_code=201)
    def create_lot(body: LotIn):
        return _errors(lambda: plat.create_lot(body.supplier, body.tile_type, body.received_date, body.reference,
                                               actor=body.actor or "inspector"))

    @app.get("/api/lots/{lot_id}")
    def lot(lot_id: str):
        return _errors(lambda: plat.lot(lot_id))

    @app.post("/api/lots/{lot_id}/decision")
    def lot_decision(lot_id: str, body: LotDecisionIn):
        return _errors(lambda: plat.decide_lot(lot_id, body.decision, body.actor, body.note, body.force))

    @app.get("/api/lots/{lot_id}/report")
    def lot_report(lot_id: str):
        return _errors(lambda: plat.lot_report(lot_id))

    @app.get("/api/lots/{lot_id}/report.csv")
    def lot_report_csv(lot_id: str):
        text = _errors(lambda: plat.lot_csv(lot_id))
        return Response(text, media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="lot-{lot_id}.csv"'})

    @app.get("/api/meta")
    def meta():
        return {"suppliers": plat.suppliers(), "tile_types": plat.tile_types()}

    # ── inspections ──────────────────────────────────────────────────────────
    @app.post("/api/inspections", status_code=201)
    def inspect(lot_id: str = Form(...), files: list[UploadFile] = File(...), source: str = Form("upload")):
        out = []                    # sync endpoint: runs in a worker thread, inference does not block the server
        for f in files:
            data = f.file.read()
            if len(data) > MAX_UPLOAD_MB * 1e6:
                raise HTTPException(413, f"{f.filename}: larger than {MAX_UPLOAD_MB} MB")
            try:
                img = open_image(data)
            except Exception:
                raise HTTPException(422, f"{f.filename}: not an image the platform can read")
            out.append(_errors(lambda: plat.inspect(img, lot_id, filename=f.filename, source=source)))
        return out

    @app.get("/api/inspections")
    def inspections(lot_id: Optional[str] = None, verdict: Optional[str] = None,
                    limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)):
        return plat.inspections(lot_id, verdict, limit, offset)

    @app.get("/api/inspections/{iid}")
    def inspection(iid: int):
        rec = _errors(lambda: plat.inspection(iid))
        rec["audit"] = plat.audit_log(entity="inspection", entity_id=iid)
        return rec

    @app.post("/api/inspections/{iid}/review")
    def review(iid: int, body: ReviewIn):
        return _errors(lambda: plat.review(iid, body.verdict, body.inspector, body.note))

    @app.get("/api/review-queue")
    def review_queue(limit: int = Query(100, ge=1, le=1000)):
        return plat.review_queue(limit)

    @app.get("/api/audit")
    def audit(limit: int = Query(100, ge=1, le=1000), entity: Optional[str] = None):
        return plat.audit_log(limit, entity)

    @app.get("/api/feedback")
    def feedback(limit: int = Query(500, ge=1, le=5000)):
        return plat.feedback(limit)

    @app.get("/api/feedback/export.csv")
    def feedback_csv():
        return Response(plat.feedback_csv(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="retraining-feedback.csv"'})

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
    def api_404(rest: str):
        raise HTTPException(404, f"no endpoint /api/{rest}")

    # ── files + frontend ─────────────────────────────────────────────────────
    app.mount("/media", StaticFiles(directory=plat.media_dir), name="media")

    dist = Path(frontend_dist) if frontend_dist else config.FRONTEND_DIST
    if (dist / "index.html").exists():
        if (dist / "assets").exists():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            f = (dist / path).resolve()
            if path and f.is_file() and dist.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(dist / "index.html")
    else:
        @app.get("/{path:path}", include_in_schema=False)
        def no_frontend(path: str):
            return HTMLResponse(
                "<h1>Tile inspection platform</h1><p>The API is running (see <a href='/docs'>/docs</a>), "
                "but the frontend has not been built yet.</p><pre>cd frontend\nnpm install\nnpm run build</pre>",
                status_code=200)

    return app
