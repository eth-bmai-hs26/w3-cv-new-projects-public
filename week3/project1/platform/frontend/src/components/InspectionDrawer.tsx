import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Inspection } from "../api";
import { dateTime, ms, pct } from "../lib/format";
import { useInspector, useToast } from "../lib/hooks";
import { CompareImage } from "./CompareImage";
import { Ruler } from "./Ruler";
import { VerdictTag } from "./Verdict";

/** Side panel with everything known about one inspected tile, plus an override. */
export function InspectionDrawer({ id, onClose, onChanged }: { id: number | null; onClose: () => void; onChanged?: (i: Inspection) => void }) {
  const [rec, setRec] = useState<Inspection | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const { name } = useInspector();
  const toast = useToast();

  useEffect(() => {
    setRec(null);
    setNote("");
    if (id != null) api.inspection(id).then(setRec).catch((e) => toast.push(e.message, "error"));
  }, [id, toast]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (id == null) return null;

  const decide = async (verdict: "APPROVE" | "REJECT") => {
    if (!name.trim()) return toast.push("Enter your inspector name at the top of the page first", "error");
    setBusy(true);
    try {
      const r = await api.review(id, verdict, name, note || undefined);
      const full = await api.inspection(id);
      setRec(full);
      onChanged?.(r);
      toast.push(`Tile #${id} ${verdict === "APPROVE" ? "approved" : "rejected"} by ${name}`);
    } catch (e) {
      toast.push((e as Error).message, "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="drawer-wrap" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} aria-label={`Tile ${id}`}>
        <header className="drawer-head">
          <div>
            <h2>Tile #{id}</h2>
            {rec && (
              <p>
                <Link to={`/lots/${rec.lot_id}`} onClick={onClose}>{rec.lot_id}</Link>, {rec.supplier}, {dateTime(rec.created_at)}
              </p>
            )}
          </div>
          <button className="btn btn-ghost" onClick={onClose} aria-label="Close">Close</button>
        </header>
        {!rec ? (
          <div className="skeleton" style={{ height: 320 }} />
        ) : (
          <div className="drawer-body">
            <CompareImage photo={rec.photo_url} overlay={rec.overlay_url} alt={`Tile ${rec.id}`} />
            <div className="drawer-verdict">
              <VerdictTag verdict={rec.verdict_final} size="lg" overridden={rec.overridden} />
              <span className="big-num">{rec.usable_pct == null ? "–" : `${rec.usable_pct.toFixed(1)}%`}<small>usable</small></span>
            </div>
            <Ruler value={rec.usable_fraction} threshold={rec.threshold} band={rec.review_band} />
            <p className="reason">{rec.error ?? rec.reason}</p>
            <dl className="facts">
              <dt>Model said</dt><dd><VerdictTag verdict={rec.verdict_auto} size="sm" /></dd>
              <dt>Confidence</dt><dd>{pct(rec.confidence)} <span className="muted">(mask certainty {pct(rec.mask_certainty)})</span></dd>
              <dt>Processing</dt><dd>{ms(rec.t_total)} <span className="muted">({ms(rec.t_inference)} inference)</span></dd>
              <dt>Model</dt><dd>{rec.model_name} <span className="muted">{rec.model_version?.slice(0, 8)}</span></dd>
              <dt>Rule at the time</dt><dd>approve from {pct(rec.threshold)}, review ±{pct(rec.review_band)}</dd>
              <dt>Value</dt><dd>€{rec.value.toFixed(2)}</dd>
              {rec.filename && (<><dt>File</dt><dd className="mono-ish">{rec.filename}</dd></>)}
            </dl>

            {rec.reviewed_by && (
              <div className="review-done">
                Decided by <b>{rec.reviewed_by}</b> {dateTime(rec.reviewed_at)}
                {rec.review_note && <q>{rec.review_note}</q>}
              </div>
            )}

            <section className="override">
              <h3>{rec.verdict_final === "REVIEW" ? "Decide this tile" : "Override the verdict"}</h3>
              <textarea placeholder="Note for the audit trail (optional)" value={note} onChange={(e) => setNote(e.target.value)} rows={2} />
              <div className="btn-row">
                <button className="btn btn-approve" disabled={busy} onClick={() => decide("APPROVE")}>Approve tile</button>
                <button className="btn btn-reject" disabled={busy} onClick={() => decide("REJECT")}>Reject tile</button>
              </div>
            </section>

            {rec.audit && rec.audit.length > 0 && (
              <section>
                <h3>History</h3>
                <ul className="audit-list">
                  {rec.audit.map((a) => (
                    <li key={a.id}>
                      <span>{dateTime(a.ts)}</span> <b>{a.actor}</b> changed {String(a.detail?.from ?? "")} to {String(a.detail?.to ?? "")}
                      {a.detail?.note ? <q>{String(a.detail.note)}</q> : null}
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}
