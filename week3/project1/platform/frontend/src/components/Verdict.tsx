import type { Verdict } from "../api";
import { verdictLabel } from "../lib/format";

/** The inspection tag hung on a tile: shape + word + colour, never colour alone. */
export function VerdictTag({ verdict, size = "md", overridden = false }: { verdict: Verdict; size?: "sm" | "md" | "lg"; overridden?: boolean }) {
  return (
    <span className={`tag tag-${verdict.toLowerCase()} tag-${size}`} title={overridden ? "Set by an inspector" : undefined}>
      {verdictLabel[verdict]}
      {overridden && <span className="tag-by">by inspector</span>}
    </span>
  );
}

export function LotStatus({ status }: { status: string }) {
  return <span className={`lot-status lot-${status}`}>{status === "open" ? "Open" : status === "accepted" ? "Accepted" : "Rejected"}</span>;
}

/** Colour dot + text, for legends and tables. */
export function VerdictDot({ verdict }: { verdict: Verdict }) {
  return <span className={`dot dot-${verdict.toLowerCase()}`} aria-hidden />;
}
