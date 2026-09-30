const nf0 = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1, minimumFractionDigits: 1 });

export const num = (n: number | null | undefined) => (n == null ? "–" : nf0.format(n));

export const pct = (f: number | null | undefined, digits = 0) =>
  f == null ? "–" : `${(100 * f).toFixed(digits)}%`;

export const money = (n: number | null | undefined, currency = "EUR", compact = false) => {
  if (n == null) return "–";
  const sym = currency === "EUR" ? "€" : currency + " ";
  if (compact && Math.abs(n) >= 10000) return `${sym}${nf1.format(n / 1000)}k`;
  const digits = Math.abs(n) < 100 && n % 1 !== 0 ? 2 : 0;
  return `${sym}${new Intl.NumberFormat("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(n)}`;
};

export const ms = (n: number | null | undefined) => (n == null ? "–" : n < 10 ? `${n.toFixed(1)} ms` : `${nf0.format(n)} ms`);

export function parseTs(ts: string) {
  return new Date(ts.length === 10 ? ts + "T00:00:00" : ts);
}

export function dateShort(ts: string | null | undefined) {
  if (!ts) return "–";
  return parseTs(ts).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}

export function dateTime(ts: string | null | undefined) {
  if (!ts) return "–";
  const d = parseTs(ts);
  const today = new Date();
  const time = d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
  if (d.toDateString() === today.toDateString()) return `Today ${time}`;
  const y = new Date(today);
  y.setDate(today.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return `Yesterday ${time}`;
  return `${d.toLocaleDateString("en-GB", { day: "numeric", month: "short" })}, ${time}`;
}

export function ago(ts: string | null | undefined) {
  if (!ts) return "–";
  const s = (Date.now() - parseTs(ts).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} days ago`;
}

export const verdictLabel: Record<string, string> = {
  APPROVE: "Approved",
  REVIEW: "Needs review",
  REJECT: "Rejected",
};

export const lotStatusLabel: Record<string, string> = {
  open: "Open",
  accepted: "Accepted",
  rejected: "Rejected",
};

/** Absolute date + time, for printed reports. */
export function dateTimeAbs(ts: string | null | undefined) {
  if (!ts) return "–";
  const d = parseTs(ts);
  return `${d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })} ${d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}`;
}
