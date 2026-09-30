import { pct } from "../lib/format";

/**
 * The decision ruler: 0-100 % usable area with the reject / review / approve
 * zones of the current rule and a marker for this tile. It is the rule, drawn.
 */
export function Ruler({
  value,
  threshold,
  band,
  compact = false,
  showScale = true,
}: {
  value?: number | null;
  threshold: number;
  band: number;
  compact?: boolean;
  showScale?: boolean;
}) {
  const lo = Math.max(0, threshold - band);
  const hi = Math.min(1, threshold + band);
  const v = value == null ? null : Math.max(0, Math.min(1, value));
  const zone = v == null ? null : v >= hi ? "approve" : v < lo ? "reject" : "review";
  return (
    <div className={`ruler ${compact ? "ruler-compact" : ""}`}
      role="img"
      aria-label={`Usable ${v == null ? "unknown" : pct(v)}; approve from ${pct(threshold)}, review between ${pct(lo)} and ${pct(hi)}`}>
      <div className="ruler-track">
        <span className="ruler-zone z-reject" style={{ left: 0, width: `${lo * 100}%` }} />
        <span className="ruler-zone z-review" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
        <span className="ruler-zone z-approve" style={{ left: `${hi * 100}%`, width: `${(1 - hi) * 100}%` }} />
        <span className="ruler-line" style={{ left: `${threshold * 100}%` }} />
        {v != null && <span className={`ruler-mark m-${zone}`} style={{ left: `${v * 100}%` }} />}
      </div>
      {showScale && !compact && (
        <div className="ruler-scale">
          <span style={{ left: 0 }}>0</span>
          <span style={{ left: "50%" }}>50</span>
          <span className="ruler-th" style={{ left: `${threshold * 100}%` }}>{Math.round(threshold * 100)}</span>
          <span style={{ left: "100%" }}>100%</span>
        </div>
      )}
    </div>
  );
}
