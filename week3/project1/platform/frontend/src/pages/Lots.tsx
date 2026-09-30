import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { IconDownload, IconPrint } from "../components/Icons";
import { InspectionDrawer } from "../components/InspectionDrawer";
import { PageHeader } from "../components/Shell";
import { LotStatus } from "../components/Verdict";
import { dateShort, dateTime, dateTimeAbs, money, num, pct } from "../lib/format";
import { useAsync, useInspector, useToast } from "../lib/hooks";
import { RecentTable } from "./Dashboard";

function Split({ a, v, r }: { a: number; v: number; r: number }) {
  const n = a + v + r;
  if (!n) return <span className="muted">no tiles</span>;
  return (
    <span className="split" aria-label={`${a} approved, ${v} review, ${r} rejected`}>
      <span className="mix-bar mix-bar-sm">
        <span className="mix-seg seg-approve" style={{ flexGrow: a }} />
        <span className="mix-seg seg-review" style={{ flexGrow: v }} />
        <span className="mix-seg seg-reject" style={{ flexGrow: r }} />
      </span>
    </span>
  );
}

export function LotsPage() {
  const [status, setStatus] = useState("");
  const [q, setQ] = useState("");
  const { data, error } = useAsync(() => api.lots(status || undefined, q || undefined), [status, q]);
  const nav = useNavigate();
  return (
    <div className="page">
      <PageHeader title="Lots" sub="Every shipment received, with how many of its tiles can be resold." />
      <div className="toolbar">
        <div className="seg" role="group" aria-label="Status">
          {[["", "All"], ["open", "Open"], ["accepted", "Accepted"], ["rejected", "Rejected"]].map(([k, l]) => (
            <button key={k} className={status === k ? "on" : ""} onClick={() => setStatus(k)}>{l}</button>
          ))}
        </div>
        <input className="search" type="search" placeholder="Search lot, supplier, tile type" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      {error && <div className="error-box">{error}</div>}
      <section className="panel panel-flush">
        <div className="table-scroll">
          <table className="table table-click">
            <thead>
              <tr>
                <th>Lot</th>
                <th>Supplier</th>
                <th>Received</th>
                <th className="num">Tiles</th>
                <th>Verdicts</th>
                <th className="num">Approved</th>
                <th className="num">Avg usable</th>
                <th className="num">Value</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {data?.map((l) => (
                <tr key={l.id} onClick={() => nav(`/lots/${l.id}`)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && nav(`/lots/${l.id}`)}>
                  <td><b>{l.id}</b>{l.reference && <small>{l.reference}</small>}</td>
                  <td>{l.supplier}<small>{l.tile_type}</small></td>
                  <td>{dateShort(l.received_date)}</td>
                  <td className="num">{num(l.n)}</td>
                  <td><Split a={l.approved} v={l.review} r={l.rejected} /></td>
                  <td className="num">{pct(l.approval_rate)}</td>
                  <td className="num">{pct(l.avg_usable, 1)}</td>
                  <td className="num">{money(l.value)}</td>
                  <td><LotStatus status={l.status} />{l.review > 0 && <small>{l.review} to review</small>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {data && !data.length && <p className="empty">No lots match. Create one on the Inspect page.</p>}
        </div>
      </section>
    </div>
  );
}

export function LotDetailPage() {
  const { id = "" } = useParams();
  const { data: lot, reload } = useAsync(() => api.lot(id), [id]);
  const { data: tiles, reload: reloadTiles } = useAsync(() => api.inspections({ lot_id: id, limit: 500 }), [id]);
  const [open, setOpen] = useState<number | null>(null);
  const [note, setNote] = useState("");
  const { name } = useInspector();
  const toast = useToast();

  if (!lot) return <div className="page"><PageHeader title={`Lot ${id}`} /><div className="skeleton" style={{ height: 300 }} /></div>;

  const decide = async (d: "accept" | "reject" | "reopen", force = false) => {
    if (!name.trim()) return toast.push("Enter your inspector name at the top of the page first", "error");
    try {
      await api.decideLot(id, d, name, note || undefined, force);
      toast.push(d === "accept" ? `Lot ${id} accepted` : d === "reject" ? `Lot ${id} rejected` : `Lot ${id} reopened`);
      setNote("");
      reload();
    } catch (e) {
      toast.push((e as Error).message, "error");
    }
  };

  return (
    <div className="page">
      <PageHeader title={`Lot ${lot.id}`} sub={<>{lot.supplier}, {lot.tile_type}, received {lot.received_date}{lot.reference ? `, delivery note ${lot.reference}` : ""}</>}>
        <a className="btn" href={api.lotCsvUrl(lot.id)} download><IconDownload /> Export CSV</a>
        <Link className="btn" to={`/lots/${lot.id}/report`} target="_blank"><IconPrint /> Printable report</Link>
      </PageHeader>

      <section className="lot-sheet">
        <div className="lot-figs">
          <div><span>Tiles</span><b>{num(lot.n)}</b></div>
          <div><span>Approved</span><b>{num(lot.approved)}</b><small>{pct(lot.approval_rate)}</small></div>
          <div><span>Needs review</span><b>{num(lot.review)}</b></div>
          <div><span>Rejected</span><b>{num(lot.rejected)}</b></div>
          <div><span>Average usable</span><b>{pct(lot.avg_usable, 1)}</b></div>
          <div><span>Resale value</span><b>{money(lot.value)}</b></div>
        </div>
        <Split a={lot.approved} v={lot.review} r={lot.rejected} />
      </section>

      <section className={`panel lot-decision lot-decision-${lot.status}`}>
        <header className="panel-head">
          <h2>Lot decision</h2>
          <LotStatus status={lot.status} />
        </header>
        {lot.status === "open" ? (
          <>
            <p>
              Accept the lot to put its approved tiles into stock, or reject it and return it to {lot.supplier}.
              {lot.review > 0 && <> {lot.review} tile{lot.review > 1 ? "s" : ""} still need a decision in the <Link to="/review">review queue</Link>.</>}
            </p>
            <textarea rows={2} placeholder="Note for the supplier and the audit trail (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
            <div className="btn-row">
              <button className="btn btn-approve" onClick={() => decide("accept")} disabled={lot.review > 0 || !lot.n}>Accept lot</button>
              <button className="btn btn-reject" onClick={() => decide("reject")} disabled={!lot.n}>Reject lot</button>
              {lot.review > 0 && <span className="muted">Accepting is possible once the review queue for this lot is empty.</span>}
            </div>
          </>
        ) : (
          <div className="decided">
            <p>
              {lot.status === "accepted" ? "Accepted" : "Rejected"} by <b>{lot.decided_by}</b> {dateTime(lot.decided_at)}.
              {lot.decision_note && <q>{lot.decision_note}</q>}
            </p>
            <button className="btn btn-small" onClick={() => decide("reopen")}>Reopen lot</button>
          </div>
        )}
      </section>

      <section className="panel">
        <header className="panel-head"><h2>Tiles in this lot</h2><span className="muted">{num(tiles?.total)} tiles</span></header>
        {tiles && <RecentTable rows={tiles.items} onOpen={setOpen} showLot={false} />}
      </section>

      <InspectionDrawer id={open} onClose={() => setOpen(null)} onChanged={() => { reload(); reloadTiles(); }} />
    </div>
  );
}

export function LotReportPage() {
  const { id = "" } = useParams();
  const { data, error } = useAsync(() => api.lotReport(id), [id]);
  if (error) return <div className="report"><p>{error}</p></div>;
  if (!data) return <div className="report"><p>Loading report…</p></div>;
  const { lot, summary: s, items, rules } = data;
  return (
    <div className="report">
      <div className="report-actions no-print">
        <button className="btn btn-primary" onClick={() => window.print()}><IconPrint /> Print</button>
        <Link className="btn" to={`/lots/${lot.id}`}>Back to lot</Link>
      </div>
      <header className="report-head">
        <div>
          <p className="report-kicker">Inspection report</p>
          <h1>Lot {lot.id}</h1>
          <p>{lot.supplier}, {lot.tile_type}. Received {lot.received_date}{lot.reference ? `, delivery note ${lot.reference}` : ""}.</p>
        </div>
        <div className={`report-status rs-${lot.status}`}>
          <b>{lot.status === "open" ? "Decision pending" : lot.status === "accepted" ? "Lot accepted" : "Lot rejected"}</b>
          {lot.decided_by && <span>{lot.decided_by}, {dateTimeAbs(lot.decided_at)}</span>}
        </div>
      </header>
      <section className="report-figs">
        <div><span>Tiles inspected</span><b>{num(s.tiles)}</b></div>
        <div><span>Approved</span><b>{num(s.approved)} ({pct(s.approval_rate)})</b></div>
        <div><span>Rejected</span><b>{num(s.rejected)}</b></div>
        <div><span>Open for review</span><b>{num(s.review)}</b></div>
        <div><span>Average usable area</span><b>{pct(s.avg_usable, 1)}</b></div>
        <div><span>Resale value</span><b>{money(s.value)}</b></div>
      </section>
      <p className="report-rule">
        Rule: a tile is approved when at least {pct(rules.threshold)} of its area is usable; tiles within ±{pct(rules.review_band)} of that line,
        or where the model is unsure, are decided by an inspector. {s.overrides} tile{s.overrides === 1 ? " was" : "s were"} decided by an inspector.
        Model: {s.models.join(", ") || "–"}. Average processing time {s.avg_ms ?? "–"} ms per tile.
      </p>
      <table className="table report-table">
        <thead>
          <tr><th>#</th><th>Photo</th><th>Time</th><th className="num">Usable</th><th>Model</th><th>Final</th><th>Decided by</th><th className="num">Value</th></tr>
        </thead>
        <tbody>
          {items.map((i, k) => (
            <tr key={i.id}>
              <td>{k + 1}</td>
              <td className="thumb-cell">{i.overlay_url && <img src={i.overlay_url} alt="" />}</td>
              <td>{dateTimeAbs(i.created_at)}</td>
              <td className="num">{i.usable_pct == null ? "–" : `${i.usable_pct.toFixed(1)}%`}</td>
              <td>{i.verdict_auto === "APPROVE" ? "Approve" : i.verdict_auto === "REJECT" ? "Reject" : "Review"}</td>
              <td><b>{i.verdict_final === "APPROVE" ? "Approved" : i.verdict_final === "REJECT" ? "Rejected" : "Pending"}</b></td>
              <td>{i.reviewed_by ? `${i.reviewed_by}${i.review_note ? `: ${i.review_note}` : ""}` : "Model"}</td>
              <td className="num">{money(i.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <footer className="report-foot">Generated {dateTimeAbs(data.generated_at)} by the tile inspection platform.</footer>
    </div>
  );
}
