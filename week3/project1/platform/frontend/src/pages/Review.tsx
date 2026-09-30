import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Inspection } from "../api";
import { CompareImage } from "../components/CompareImage";
import { IconDownload } from "../components/Icons";
import { Ruler } from "../components/Ruler";
import { PageHeader } from "../components/Shell";
import { VerdictTag } from "../components/Verdict";
import { ago, dateTime, ms, num, pct } from "../lib/format";
import { useAsync, useInspector, useToast } from "../lib/hooks";

export default function Review() {
  const { data: queue, reload, setData } = useAsync(() => api.reviewQueue(), []);
  const { data: feedback, reload: reloadFeedback } = useAsync(() => api.feedback(), []);
  const [sel, setSel] = useState<number | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const { name } = useInspector();
  const toast = useToast();

  const items = queue?.items ?? [];
  const current: Inspection | undefined = items.find((i) => i.id === sel) ?? items[0];

  useEffect(() => {
    if (!sel && items.length) setSel(items[0].id);
  }, [items, sel]);

  const decide = useCallback(
    async (verdict: "APPROVE" | "REJECT") => {
      if (!current || busy) return;
      if (!name.trim()) return toast.push("Enter your inspector name at the top of the page first", "error");
      setBusy(true);
      try {
        await api.review(current.id, verdict, name, note || undefined);
        toast.push(`Tile #${current.id} ${verdict === "APPROVE" ? "approved" : "rejected"}. Saved as training feedback.`);
        const idx = items.findIndex((i) => i.id === current.id);
        const rest = items.filter((i) => i.id !== current.id);
        setData({ total: (queue?.total ?? 1) - 1, items: rest });
        setSel(rest[Math.min(idx, rest.length - 1)]?.id ?? null);
        setNote("");
        reloadFeedback();
      } catch (e) {
        toast.push((e as Error).message, "error");
        reload();
      } finally {
        setBusy(false);
      }
    },
    [current, busy, name, note, items, queue, setData, reload, reloadFeedback, toast],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea, select")) return;
      if (e.key === "a" || e.key === "A") decide("APPROVE");
      if (e.key === "r" || e.key === "R") decide("REJECT");
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        const idx = items.findIndex((i) => i.id === current?.id);
        const next = items[idx + (e.key === "ArrowDown" ? 1 : -1)];
        if (next) {
          e.preventDefault();
          setSel(next.id);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [decide, items, current]);

  const decided = feedback ?? [];
  const disagreements = decided.filter((f) => f.model_verdict !== "REVIEW" && !f.agrees).length;

  return (
    <div className="page">
      <PageHeader title="Review queue" sub="Borderline tiles and tiles the model is unsure about. Your decision is final and is kept as training data." />

      {queue && !items.length ? (
        <section className="panel empty-state">
          <h2>The queue is clear</h2>
          <p>Every inspected tile has a decision. New borderline tiles appear here as the line runs.</p>
          <Link className="btn" to="/inspect">Inspect more tiles</Link>
        </section>
      ) : (
        <div className="review-grid">
          <section className="panel panel-flush queue">
            <header className="panel-head"><h2>{num(queue?.total)} waiting</h2><span className="muted">least confident first</span></header>
            <ul className="queue-list">
              {items.map((i) => (
                <li key={i.id}>
                  <button className={`queue-item ${current?.id === i.id ? "on" : ""}`} onClick={() => setSel(i.id)}>
                    {i.overlay_url && <img src={i.overlay_url} alt="" loading="lazy" />}
                    <span className="qi-main">
                      <b>#{i.id} <em>{i.usable_pct == null ? "no tile" : `${i.usable_pct.toFixed(1)}%`}</em></b>
                      <small>{i.lot_id}, {i.supplier}</small>
                      <small>{i.error ? i.error : i.reason?.startsWith("Low") ? "Model unsure" : "Near the line"}, {ago(i.created_at)}</small>
                    </span>
                    <span className="qi-conf" title="Confidence">{pct(i.confidence)}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>

          {current && (
            <section className="panel review-detail">
              <div className="rd-image">
                <CompareImage photo={current.photo_url} overlay={current.overlay_url} alt={`Tile ${current.id}`} />
              </div>
              <div className="rd-side">
                <header>
                  <h2>Tile #{current.id}</h2>
                  <p><Link to={`/lots/${current.lot_id}`}>{current.lot_id}</Link>, {current.supplier}, {current.tile_type}</p>
                  <p className="muted">{dateTime(current.created_at)}</p>
                </header>
                <div className="rd-num">
                  <span className="big-num">{current.usable_pct == null ? "–" : `${current.usable_pct.toFixed(1)}%`}<small>usable</small></span>
                  <VerdictTag verdict="REVIEW" />
                </div>
                <Ruler value={current.usable_fraction} threshold={current.threshold} band={current.review_band} />
                <p className="reason">{current.error ?? current.reason}</p>
                <dl className="facts">
                  <dt>Confidence</dt><dd>{pct(current.confidence)} <span className="muted">(mask certainty {pct(current.mask_certainty)})</span></dd>
                  <dt>Model</dt><dd>{current.model_name}</dd>
                  <dt>Processing</dt><dd>{ms(current.t_total)}</dd>
                </dl>
                <p className="hint">Green is the area the model would keep; red is cut away as damaged. Drag the divider to compare with the photo.</p>
                <textarea rows={2} placeholder="Note, e.g. crack only in the glaze (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
                <div className="btn-row decide-row">
                  <button className="btn btn-approve btn-big" disabled={busy} onClick={() => decide("APPROVE")}>Approve<kbd>A</kbd></button>
                  <button className="btn btn-reject btn-big" disabled={busy} onClick={() => decide("REJECT")}>Reject<kbd>R</kbd></button>
                </div>
                {!name && <p className="warn-line">Enter your name in the Inspector field so the decision is signed.</p>}
              </div>
            </section>
          )}
        </div>
      )}

      <section className="panel">
        <header className="panel-head">
          <h2>Inspector decisions</h2>
          <a className="btn btn-small" href={api.feedbackCsvUrl} download><IconDownload /> Export for retraining</a>
        </header>
        <p className="panel-note">
          {num(decided.length)} tiles labelled by inspectors, ready to add to the training set.
          {disagreements > 0 && ` On ${disagreements} of them the inspector overruled a clear model verdict.`}
        </p>
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr><th aria-label="Photo" /><th>When</th><th>Inspector</th><th>Tile</th><th className="num">Usable</th><th>Model</th><th>Inspector</th><th>Note</th></tr>
            </thead>
            <tbody>
              {decided.slice(0, 12).map((f) => (
                <tr key={f.id}>
                  <td className="thumb-cell">{f.photo_url && <img src={f.photo_url} alt="" loading="lazy" />}</td>
                  <td>{dateTime(f.ts)}</td>
                  <td>{f.inspector}</td>
                  <td>#{f.inspection_id}</td>
                  <td className="num">{pct(f.usable_fraction, 1)}</td>
                  <td><VerdictTag verdict={f.model_verdict} size="sm" /></td>
                  <td><VerdictTag verdict={f.human_verdict} size="sm" /></td>
                  <td className="note-cell">{f.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
