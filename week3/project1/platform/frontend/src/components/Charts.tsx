import { Bar, BarChart, CartesianGrid, Cell, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Dashboard, DayPoint, Verdict } from "../api";
import { dateShort, num, pct, verdictLabel } from "../lib/format";
import { VerdictDot } from "./Verdict";

export const C = {
  approve: "#2A8C55",
  review: "#D08A00",
  reject: "#C63B26",
  grid: "#E3E7E6",
  axis: "#7A858A",
  surface: "#FFFFFF",
  ink: "#1C2326",
};

const axisProps = {
  tick: { fill: C.axis, fontSize: 12, fontFamily: "Barlow, system-ui, sans-serif" },
  tickLine: false,
  axisLine: { stroke: "#CBD2D2" },
};

function Legend({ items }: { items: { v: Verdict; n?: number }[] }) {
  return (
    <div className="legend">
      {items.map(({ v, n }) => (
        <span key={v}>
          <VerdictDot verdict={v} /> {verdictLabel[v]}
          {n != null && <b>{num(n)}</b>}
        </span>
      ))}
    </div>
  );
}

function TipBox({ title, rows }: { title: string; rows: { v?: Verdict; label: string; value: string }[] }) {
  return (
    <div className="chart-tip">
      <b>{title}</b>
      {rows.map((r) => (
        <div key={r.label}>
          {r.v && <VerdictDot verdict={r.v} />}
          <span>{r.label}</span>
          <em>{r.value}</em>
        </div>
      ))}
    </div>
  );
}

export function ThroughputChart({ data }: { data: DayPoint[] }) {
  const total = data.reduce((s, d) => s + d.n, 0);
  return (
    <>
      <Legend items={[{ v: "APPROVE" }, { v: "REVIEW" }, { v: "REJECT" }]} />
      <div className="chart-box" style={{ height: 230 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 4, bottom: 0, left: -18 }} barCategoryGap="22%">
            <CartesianGrid vertical={false} stroke={C.grid} />
            <XAxis dataKey="day" {...axisProps} tickFormatter={dateShort} minTickGap={24} />
            <YAxis {...axisProps} axisLine={false} allowDecimals={false} />
            <Tooltip
              cursor={{ fill: "rgba(28,35,38,0.05)" }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload as DayPoint;
                return (
                  <TipBox
                    title={`${new Date(d.day).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" })}: ${d.n} tiles`}
                    rows={[
                      { v: "APPROVE", label: "Approved", value: num(d.approve) },
                      { v: "REVIEW", label: "Needs review", value: num(d.review) },
                      { v: "REJECT", label: "Rejected", value: num(d.reject) },
                      { label: "Avg usable", value: pct(d.avg_usable) },
                    ]}
                  />
                );
              }}
            />
            <Bar dataKey="approve" stackId="v" fill={C.approve} stroke={C.surface} strokeWidth={1.5} maxBarSize={22} />
            <Bar dataKey="review" stackId="v" fill={C.review} stroke={C.surface} strokeWidth={1.5} maxBarSize={22} />
            <Bar dataKey="reject" stackId="v" fill={C.reject} stroke={C.surface} strokeWidth={1.5} radius={[3, 3, 0, 0]} maxBarSize={22} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="chart-foot">{num(total)} tiles in {data.length} days</p>
    </>
  );
}

export function HistogramChart({ data, threshold, band }: { data: Dashboard["histogram"]; threshold: number; band: number }) {
  // start the axis a little below the lowest tile (or 20 pts under the threshold), in 10 % steps
  const first = data.find((h) => h.n > 0)?.lo ?? 0;
  const lo = Math.max(0, Math.floor(Math.min(first, threshold - 0.2) * 10) / 10);
  const rows = data.filter((h) => h.lo >= lo - 1e-9).map((h) => ({ ...h, mid: (h.lo + h.hi) / 2 }));
  const ticks = Array.from({ length: Math.round((1 - lo) * 10) + 1 }, (_, i) => Math.round((lo + i / 10) * 10) / 10);
  const col = (z: Verdict) => (z === "APPROVE" ? C.approve : z === "REVIEW" ? C.review : C.reject);
  return (
    <>
      <Legend items={[{ v: "REJECT" }, { v: "REVIEW" }, { v: "APPROVE" }]} />
      <div className="chart-box" style={{ height: 230 }}>
        <ResponsiveContainer>
          <BarChart data={rows} margin={{ top: 18, right: 8, bottom: 0, left: -18 }} barCategoryGap={2}>
            <CartesianGrid vertical={false} stroke={C.grid} />
            <XAxis dataKey="mid" type="number" domain={[lo, 1]} ticks={ticks} tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axisProps} />
            <YAxis {...axisProps} axisLine={false} allowDecimals={false} />
            <ReferenceArea x1={threshold - band} x2={threshold + band} fill={C.review} fillOpacity={0.08} />
            <ReferenceLine x={threshold} stroke={C.ink} strokeWidth={1.5}
              label={{ value: `approve from ${Math.round(threshold * 100)}%`, position: "top", fill: C.ink, fontSize: 12, fontFamily: "Barlow" }} />
            <Tooltip
              cursor={{ fill: "rgba(28,35,38,0.05)" }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const h = payload[0].payload as (typeof rows)[number];
                return <TipBox title={`${Math.round(h.lo * 100)}–${Math.round(h.hi * 100)}% usable`} rows={[{ v: h.zone, label: verdictLabel[h.zone], value: `${h.n} tiles` }]} />;
              }}
            />
            <Bar dataKey="n" radius={[3, 3, 0, 0]}>
              {rows.map((r, i) => (
                <Cell key={i} fill={col(r.zone)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="chart-foot">Share of each tile that can be sold after cutting away damage{lo > 0 ? `; axis starts at ${Math.round(lo * 100)}%` : ""}</p>
    </>
  );
}

export function VerdictMix({ mix }: { mix: Record<Verdict, number> }) {
  const total = mix.APPROVE + mix.REVIEW + mix.REJECT || 1;
  const order: Verdict[] = ["APPROVE", "REVIEW", "REJECT"];
  return (
    <div className="mix">
      <div className="mix-bar" role="img" aria-label={order.map((v) => `${verdictLabel[v]} ${pct(mix[v] / total)}`).join(", ")}>
        {order.map((v) => (
          <span key={v} className={`mix-seg seg-${v.toLowerCase()}`} style={{ flexGrow: mix[v] }} title={`${verdictLabel[v]}: ${mix[v]}`} />
        ))}
      </div>
      <ul className="mix-rows">
        {order.map((v) => (
          <li key={v}>
            <VerdictDot verdict={v} />
            <span>{verdictLabel[v]}</span>
            <b>{pct(mix[v] / total)}</b>
            <em>{num(mix[v])}</em>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function SupplierTable({ rows, threshold }: { rows: Dashboard["suppliers"]; threshold: number }) {
  return (
    <div className="table-scroll">
      <table className="table supplier-table">
        <thead>
          <tr>
            <th>Supplier</th>
            <th className="num">Tiles</th>
            <th className="bar-col">Approved</th>
            <th className="num">Avg usable</th>
            <th className="num">Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => {
            const rate = s.approval_rate ?? 0;
            const poor = (s.avg_usable ?? 0) < threshold;
            return (
              <tr key={s.supplier}>
                <td>
                  <b>{s.supplier}</b>
                  <small>{s.lots} lots</small>
                </td>
                <td className="num">{num(s.n)}</td>
                <td className="bar-col">
                  <span className="meter" aria-label={`${pct(rate)} approved`}>
                    <i style={{ width: `${rate * 100}%` }} />
                  </span>
                  <span className="meter-v">{pct(rate)}</span>
                </td>
                <td className={`num ${poor ? "below" : ""}`}>{pct(s.avg_usable)}</td>
                <td className="num">€{num(s.value)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
