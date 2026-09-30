import { useState } from "react";
import { Link } from "react-router-dom";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { api, type Inspection } from "../api";
import { C, HistogramChart, SupplierTable, ThroughputChart, VerdictMix } from "../components/Charts";
import { InspectionDrawer } from "../components/InspectionDrawer";
import { PageHeader } from "../components/Shell";
import { VerdictTag } from "../components/Verdict";
import { ago, dateTime, money, ms, num, pct } from "../lib/format";
import { useAsync } from "../lib/hooks";

function Delta({ now, prev, kind = "count", goodUp = true }: { now: number | null; prev: number | null; kind?: "count" | "rate"; goodUp?: boolean }) {
  if (now == null || prev == null || (kind === "count" && prev === 0)) return <span className="delta">no earlier data</span>;
  const diff = kind === "rate" ? (now - prev) * 100 : ((now - prev) / prev) * 100;
  if (Math.abs(diff) < 0.5) return <span className="delta">same as last week</span>;
  const up = diff > 0;
  const good = up === goodUp;
  const txt = kind === "rate" ? `${up ? "+" : "−"}${Math.abs(diff).toFixed(1)} pts` : `${up ? "+" : "−"}${Math.abs(diff).toFixed(0)}%`;
  return (
    <span className={`delta ${good ? "delta-good" : "delta-bad"}`}>
      <i aria-hidden>{up ? "▲" : "▼"}</i> {txt} vs last week
    </span>
  );
}

export default function Dashboard() {
  const [days, setDays] = useState(30);
  const { data, error, reload } = useAsync(() => api.dashboard(days), [days]);
  const [open, setOpen] = useState<number | null>(null);

  if (error) return <div className="page"><PageHeader title="Dashboard" /><div className="error-box">Could not load the dashboard: {error}</div></div>;
  if (!data) return <div className="page"><PageHeader title="Dashboard" /><div className="skeleton" style={{ height: 480 }} /></div>;

  const k = data.kpis;
  const r = data.rules;
  const spark = data.throughput.map((d) => ({ day: d.day, v: d.approve }));

  return (
    <div className="page">
      <PageHeader title="Dashboard" sub={`Line 1, last ${days} days. Rule: approve a tile when at least ${pct(r.threshold)} of it is usable.`}>
        <div className="seg" role="group" aria-label="Period">
          {[7, 30, 90].map((d) => (
            <button key={d} className={d === days ? "on" : ""} onClick={() => setDays(d)}>{d} days</button>
          ))}
        </div>
      </PageHeader>

      <section className="hero">
        <div className="hero-value">
          <span className="hero-label">Recovered value, last {days} days</span>
          <span className="hero-num">{money(k.recovered_value, k.currency)}</span>
          <span className="hero-sub">
            {num(data.mix.APPROVE)} tiles cleared for resale at {money(r.price_basis === "per_m2" ? r.price_per_m2 * r.tile_area_m2 : r.price_per_tile, k.currency)} each
          </span>
          <div className="hero-spark" aria-hidden>
            <ResponsiveContainer>
              <AreaChart data={spark} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
                <Area type="monotone" dataKey="v" stroke={C.approve} strokeWidth={2} fill={C.approve} fillOpacity={0.1} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="hero-side">
          <div className="money-line">
            <span>Defective tiles kept off the market</span>
            <b>{money(k.errors_avoided, k.currency)}</b>
            <small>{num(data.mix.REJECT)} rejects × {money(r.cost_bad_tile_shipped, k.currency)} return cost</small>
          </div>
          <div className="money-line">
            <span>Inspector time saved</span>
            <b>{money(k.labour_saved, k.currency)}</b>
            <small>{pct(k.automation_rate)} of tiles decided without a person</small>
          </div>
        </div>
      </section>

      <section className="kpis" aria-label="Key figures">
        <div className="kpi">
          <span className="kpi-label">Inspected today</span>
          <span className="kpi-value">{num(k.today)}</span>
          <span className="delta">{num(k.yesterday)} yesterday</span>
        </div>
        <div className="kpi">
          <span className="kpi-label">This week</span>
          <span className="kpi-value">{num(k.week)}</span>
          <Delta now={k.week} prev={k.prev_week} />
        </div>
        <div className="kpi">
          <span className="kpi-label">Approval rate</span>
          <span className="kpi-value">{pct(k.approval_rate)}</span>
          <Delta now={k.approval_rate} prev={k.prev_approval_rate} kind="rate" />
        </div>
        <div className="kpi">
          <span className="kpi-label">Average usable area</span>
          <span className="kpi-value">{pct(k.avg_usable, 1)}</span>
          <Delta now={k.avg_usable} prev={k.prev_avg_usable} kind="rate" />
        </div>
        <Link to="/review" className={`kpi kpi-link ${k.review_queue ? "kpi-attn" : ""}`}>
          <span className="kpi-label">Waiting for review</span>
          <span className="kpi-value">{num(k.review_queue)}</span>
          <span className="delta">{k.review_queue ? `oldest ${ago(k.oldest_review)}` : "queue is clear"}</span>
        </Link>
      </section>

      <div className="grid grid-2-1">
        <section className="panel">
          <header className="panel-head"><h2>Tiles inspected per day</h2></header>
          <ThroughputChart data={data.throughput} />
        </section>
        <section className="panel">
          <header className="panel-head"><h2>Verdicts</h2></header>
          <VerdictMix mix={data.mix} />
          <p className="panel-note mix-note">
            Tiles in the review band (±{pct(r.review_band)} around {pct(r.threshold)}) or with low model confidence go to an inspector.
          </p>
          <dl className="station-facts">
            <div><dt>Per tile</dt><dd>{ms(k.avg_ms)}</dd></div>
            <div><dt>Decided by inspectors</dt><dd>{num(k.overrides)}</dd></div>
            <div><dt>Model</dt><dd>{data.model.mode === "demo" ? "Demo heuristic" : `${data.model.name}`}</dd></div>
          </dl>
        </section>
      </div>

      <div className="grid grid-1-1">
        <section className="panel">
          <header className="panel-head"><h2>Usable area per tile</h2></header>
          <HistogramChart data={data.histogram} threshold={r.threshold} band={r.review_band} />
        </section>
        <section className="panel">
          <header className="panel-head"><h2>Quality by supplier</h2></header>
          <SupplierTable rows={data.suppliers} threshold={r.threshold} />
        </section>
      </div>

      <section className="panel">
        <header className="panel-head">
          <h2>Latest inspections</h2>
          <Link to="/lots" className="link">All lots</Link>
        </header>
        <RecentTable rows={data.recent} onOpen={setOpen} />
      </section>

      <InspectionDrawer id={open} onClose={() => setOpen(null)} onChanged={reload} />
    </div>
  );
}

export function RecentTable({ rows, onOpen, showLot = true }: { rows: Inspection[]; onOpen: (id: number) => void; showLot?: boolean }) {
  if (!rows.length) return <p className="empty">No tiles inspected yet. Start on the Inspect page.</p>;
  return (
    <div className="table-scroll">
      <table className="table table-click">
        <thead>
          <tr>
            <th aria-label="Photo" />
            <th>Tile</th>
            {showLot && <th>Lot</th>}
            <th>Time</th>
            <th className="num">Usable</th>
            <th>Verdict</th>
            <th className="num">Confidence</th>
            <th className="num">Time taken</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((i) => (
            <tr key={i.id} onClick={() => onOpen(i.id)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onOpen(i.id)}>
              <td className="thumb-cell">{i.overlay_url && <img src={i.overlay_url} alt="" loading="lazy" />}</td>
              <td><b>#{i.id}</b></td>
              {showLot && <td>{i.lot_id}<small>{i.supplier}</small></td>}
              <td>{dateTime(i.created_at)}</td>
              <td className="num">{i.usable_pct == null ? "–" : `${i.usable_pct.toFixed(1)}%`}</td>
              <td><VerdictTag verdict={i.verdict_final} size="sm" overridden={i.overridden} /></td>
              <td className="num">{pct(i.confidence)}</td>
              <td className="num">{ms(i.t_total)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
