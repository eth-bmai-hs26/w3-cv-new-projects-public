"""API tests with FastAPI's TestClient: every request goes through the real app, DB and model service."""

import csv
import io

import pytest


def make_lot(client, supplier="Rückbau Zürich AG", tile_type="Terracotta 30×30"):
    r = client.post("/api/lots", json={"supplier": supplier, "tile_type": tile_type, "received_date": "2026-09-01",
                                       "reference": "DN-1", "actor": "tester"})
    assert r.status_code == 201, r.text
    return r.json()


def upload(client, lot_id, path):
    with open(path, "rb") as f:
        r = client.post("/api/inspections", data={"lot_id": lot_id}, files=[("files", (path.name, f, "image/jpeg"))])
    assert r.status_code == 201, r.text
    return r.json()[0]


def put_rules(client, **rules):
    r = client.put("/api/settings", json={"rules": rules, "actor": "tester"})
    assert r.status_code == 200, r.text
    return r.json()


# ── basics ───────────────────────────────────────────────────────────────────

def test_health_and_demo_mode(client):
    assert client.get("/api/health").json()["ok"] is True
    m = client.get("/api/model").json()
    assert m["mode"] == "demo" and m["classes"] == 3


def test_frontend_missing_page_explains_build(client):
    r = client.get("/")
    assert r.status_code == 200 and "npm run build" in r.text


def test_unknown_api_path_is_404(client):
    assert client.get("/api/nope").status_code == 404


# ── upload -> inspection stored -> verdict ───────────────────────────────────

def test_upload_is_inspected_and_stored(client, photo):
    path, gt = photo
    lot = make_lot(client)
    rec = upload(client, lot["id"], path)

    assert rec["lot_id"] == lot["id"] and rec["filename"] == path.name
    assert rec["verdict_auto"] in ("APPROVE", "REVIEW", "REJECT")
    assert rec["verdict_final"] == rec["verdict_auto"]
    assert 0 <= rec["usable_fraction"] <= 1
    assert abs(rec["usable_fraction"] - gt) < 0.2            # demo heuristic is roughly right
    assert rec["t_total"] > 0 and rec["model_kind"] == "demo"
    for k in ("photo_url", "overlay_url", "mask_url"):
        assert client.get(rec[k]).status_code == 200

    # stored: visible through the list, the lot and the dashboard
    got = client.get(f"/api/inspections/{rec['id']}").json()
    assert got["usable_fraction"] == rec["usable_fraction"]
    assert client.get(f"/api/lots/{lot['id']}").json()["n"] == 1
    assert client.get("/api/dashboard").json()["kpis"]["today"] == 1


def test_non_image_upload_is_refused(client):
    lot = make_lot(client)
    r = client.post("/api/inspections", data={"lot_id": lot["id"]}, files=[("files", ("x.jpg", b"not an image", "image/jpeg"))])
    assert r.status_code == 422


def test_upload_to_unknown_lot_is_404(client, photo):
    with open(photo[0], "rb") as f:
        r = client.post("/api/inspections", data={"lot_id": "L000000-99"}, files=[("files", ("a.jpg", f, "image/jpeg"))])
    assert r.status_code == 404


def test_blank_photo_means_no_tile(client, tmp_path):
    from PIL import Image
    p = tmp_path / "belt.png"
    Image.new("RGB", (300, 300), (40, 42, 41)).save(p)
    rec = upload(client, make_lot(client)["id"], p)
    assert rec["error"] == "No tile detected" and rec["verdict_final"] == "REVIEW" and rec["usable_fraction"] is None


# ── settings change the verdict ──────────────────────────────────────────────

def test_settings_change_affects_verdict(client, photo):
    lot = make_lot(client)
    first = upload(client, lot["id"], photo[0])
    uf = first["usable_fraction"]

    put_rules(client, threshold=max(0.02, uf - 0.2), review_band=0.0, min_confidence=0.0)
    assert upload(client, lot["id"], photo[0])["verdict_final"] == "APPROVE"

    put_rules(client, threshold=min(0.98, uf + 0.1), review_band=0.0, min_confidence=0.0)
    rej = upload(client, lot["id"], photo[0])
    assert rej["verdict_final"] == "REJECT" and rej["value"] == 0

    put_rules(client, threshold=round(uf, 3), review_band=0.05)
    assert upload(client, lot["id"], photo[0])["verdict_final"] == "REVIEW"

    audit = client.get("/api/audit?entity=settings").json()
    assert any(a["action"] == "settings.update" for a in audit)


def test_invalid_settings_are_refused(client):
    r = client.put("/api/settings", json={"rules": {"threshold": 3}})
    assert r.status_code == 422
    assert client.get("/api/settings").json()["rules"]["threshold"] == 0.85


def test_rescore_open_lots_and_preview(client, photo):
    lot = make_lot(client)
    rec = upload(client, lot["id"], photo[0])
    uf = rec["usable_fraction"]
    prev = client.post("/api/settings/preview", json={"threshold": max(0.02, uf - 0.3), "review_band": 0, "min_confidence": 0}).json()
    assert prev["tiles"] == 1 and prev["proposed"]["counts"]["APPROVE"] == 1

    r = client.put("/api/settings", json={"rules": {"threshold": min(0.98, uf + 0.1), "review_band": 0, "min_confidence": 0},
                                          "rescore_open_lots": True, "actor": "tester"})
    assert r.json()["rescored"]["tiles"] == 1
    assert client.get(f"/api/inspections/{rec['id']}").json()["verdict_final"] == "REJECT"


# ── review queue: override, audit trail, feedback ────────────────────────────

def test_review_override_is_audited_and_saved_as_feedback(client, photo):
    lot = make_lot(client)
    uf = upload(client, lot["id"], photo[0])["usable_fraction"]
    put_rules(client, threshold=round(uf, 3), review_band=0.05)      # this tile is now borderline
    rec = upload(client, lot["id"], photo[0])
    assert rec["verdict_final"] == "REVIEW"

    queue = client.get("/api/review-queue").json()
    assert rec["id"] in [i["id"] for i in queue["items"]]

    # inspector name is required
    assert client.post(f"/api/inspections/{rec['id']}/review", json={"verdict": "APPROVE", "inspector": " "}).status_code == 422
    assert client.post(f"/api/inspections/{rec['id']}/review", json={"verdict": "MAYBE", "inspector": "A"}).status_code == 422

    r = client.post(f"/api/inspections/{rec['id']}/review",
                    json={"verdict": "APPROVE", "inspector": "M. Keller", "note": "crack only in the glaze"})
    assert r.status_code == 200
    out = r.json()
    assert out["verdict_final"] == "APPROVE" and out["verdict_auto"] == "REVIEW"
    assert out["reviewed_by"] == "M. Keller" and out["value"] > 0

    assert rec["id"] not in [i["id"] for i in client.get("/api/review-queue").json()["items"]]
    detail = client.get(f"/api/inspections/{rec['id']}").json()
    assert detail["audit"][0]["actor"] == "M. Keller" and detail["audit"][0]["detail"]["to"] == "APPROVE"

    fb = client.get("/api/feedback").json()
    assert fb[0]["inspection_id"] == rec["id"] and fb[0]["human_verdict"] == "APPROVE" and fb[0]["note"] == "crack only in the glaze"
    csv_text = client.get("/api/feedback/export.csv").text
    assert "crack only in the glaze" in csv_text and "classes.png" in csv_text


# ── lots: summary, decision, reports ─────────────────────────────────────────

def test_lot_summary_decision_and_reports(client, sample_dir):
    lot = make_lot(client, supplier="Bauteilbörse Basel")
    photos = sorted((sample_dir / "original").glob("*.jpg"))[:5]
    recs = [upload(client, lot["id"], p) for p in photos]

    summary = client.get(f"/api/lots/{lot['id']}").json()
    assert summary["n"] == 5
    assert summary["approved"] + summary["rejected"] + summary["review"] == 5
    assert summary["value"] == pytest.approx(sum(r["value"] for r in recs))

    rep = client.get(f"/api/lots/{lot['id']}/report").json()
    assert rep["summary"]["tiles"] == 5 and len(rep["items"]) == 5

    r = client.get(f"/api/lots/{lot['id']}/report.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(r.text)))
    header = rows[2]
    assert header[:3] == ["inspection_id", "time", "filename"] and len(rows) == 3 + 5

    # accept is blocked while tiles wait for review; reject always works
    if summary["review"]:
        assert client.post(f"/api/lots/{lot['id']}/decision", json={"decision": "accept", "actor": "A"}).status_code == 409
    r = client.post(f"/api/lots/{lot['id']}/decision", json={"decision": "reject", "actor": "A. Rossi", "note": "too many cracks"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"

    # a closed lot takes no more tiles until reopened
    with open(photos[0], "rb") as f:
        assert client.post("/api/inspections", data={"lot_id": lot["id"]}, files=[("files", ("a.jpg", f, "image/jpeg"))]).status_code == 409
    assert client.post(f"/api/lots/{lot['id']}/decision", json={"decision": "reopen", "actor": "A"}).json()["status"] == "open"
    assert any(a["action"] == "lot.reject" for a in client.get("/api/audit?entity=lot").json())


def test_lot_validation(client):
    assert client.post("/api/lots", json={"supplier": "", "tile_type": "x"}).status_code == 422
    assert client.post("/api/lots", json={"supplier": "a", "tile_type": "x", "received_date": "27/09"}).status_code == 422
    assert client.get("/api/lots/L999999-01").status_code == 404


# ── seed + dashboard ─────────────────────────────────────────────────────────

def test_reset_demo_fills_dashboard(client, sample_dir):
    r = client.post("/api/admin/reset-demo", json={"tiles": 80, "days": 10, "image_dir": str(sample_dir / "original")})
    assert r.status_code == 200, r.text
    assert r.json()["tiles"] > 40
    d = client.get("/api/dashboard?days=10").json()
    assert d["kpis"]["period_tiles"] == r.json()["tiles"]
    assert len(d["throughput"]) == 10 and len(d["suppliers"]) >= 3
    assert sum(h["n"] for h in d["histogram"]) > 0
    assert sum(d["mix"].values()) == d["kpis"]["period_tiles"]
    assert client.get("/api/lots").json()


# ── model loading ────────────────────────────────────────────────────────────

def test_missing_checkpoint_falls_back_to_demo(client):
    m = client.post("/api/model/reload", json={"checkpoint": "/nope/model.pt"}).json()
    assert m["mode"] == "demo" and "not found" in m["warning"]
