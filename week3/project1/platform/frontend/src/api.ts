// Typed wrappers around the FastAPI backend (see backend/app.py).

export type Verdict = "APPROVE" | "REVIEW" | "REJECT";

export interface Inspection {
  id: number;
  lot_id: string;
  created_at: string;
  filename: string | null;
  photo_url: string | null;
  overlay_url: string | null;
  mask_url: string | null;
  usable_fraction: number | null;
  usable_pct: number | null;
  tile_share: number;
  mask_certainty: number;
  confidence: number;
  margin: number;
  verdict_auto: Verdict;
  verdict_final: Verdict;
  reason: string | null;
  error: string | null;
  threshold: number;
  review_band: number;
  value: number;
  model_name: string;
  model_version: string;
  model_kind: string;
  t_preprocess: number;
  t_inference: number;
  t_postprocess: number;
  t_total: number;
  reviewed_by: string | null;
  reviewed_at: string | null;
  review_note: string | null;
  source: string;
  supplier: string;
  tile_type: string;
  lot_status: string;
  overridden: boolean;
  warnings?: string[];
  audit?: AuditEntry[];
}

export interface Lot {
  id: string;
  reference: string | null;
  supplier: string;
  tile_type: string;
  received_date: string;
  status: "open" | "accepted" | "rejected";
  decided_by: string | null;
  decided_at: string | null;
  decision_note: string | null;
  created_at: string;
  n: number;
  approved: number;
  rejected: number;
  review: number;
  approval_rate: number | null;
  avg_usable: number | null;
  value: number;
  errors_avoided: number;
  last_inspection: string | null;
}

export interface Rules {
  threshold: number;
  review_band: number;
  min_confidence: number;
  price_basis: "per_tile" | "per_m2";
  price_per_tile: number;
  price_per_m2: number;
  tile_area_m2: number;
  cost_bad_tile_shipped: number;
  cost_manual_check: number;
  currency: string;
}

export interface ModelInfo {
  mode: "demo" | "model";
  kind: string;
  name: string;
  version: string;
  classes: number;
  architecture?: string;
  checkpoint: string | null;
  size_mb?: number;
  modified?: string;
  device: string;
  params: number;
  metrics: Record<string, number | string>;
  trained_on?: string | null;
  warning: string | null;
  loaded_at: string;
  description?: string;
}

export interface Settings {
  rules: Rules;
  model: { checkpoint: string | null; device: string };
  image_source: string;
  defaults: Rules;
  default_checkpoint: string;
}

export interface AuditEntry {
  id: number;
  ts: string;
  actor: string;
  action: string;
  entity: string;
  entity_id: string | null;
  detail: Record<string, unknown> | null;
}

export interface DayPoint {
  day: string;
  approve: number;
  review: number;
  reject: number;
  n: number;
  avg_usable: number | null;
}

export interface Dashboard {
  kpis: {
    today: number;
    yesterday: number;
    week: number;
    prev_week: number;
    approval_rate: number | null;
    prev_approval_rate: number | null;
    avg_usable: number | null;
    prev_avg_usable: number | null;
    review_queue: number;
    oldest_review: string | null;
    recovered_value: number;
    errors_avoided: number;
    labour_saved: number;
    period_tiles: number;
    automation_rate: number | null;
    currency: string;
    avg_ms: number | null;
    overrides: number;
  };
  throughput: DayPoint[];
  histogram: { lo: number; hi: number; n: number; zone: Verdict }[];
  suppliers: {
    supplier: string;
    n: number;
    approved: number;
    rejected: number;
    review: number;
    avg_usable: number | null;
    approval_rate: number | null;
    value: number;
    lots: number;
  }[];
  mix: Record<Verdict, number>;
  recent: Inspection[];
  rules: Rules;
  model: ModelInfo;
  days: number;
}

export interface PreviewSide {
  counts: Record<Verdict, number>;
  approval_rate: number;
  review_rate: number;
  recovered_value: number;
  errors_avoided: number;
  labour_saved: number;
  review_cost: number;
}

export interface Preview {
  current: PreviewSide;
  proposed: PreviewSide;
  tiles: number;
  days: number;
}

export interface LotReport {
  lot: Lot;
  items: Inspection[];
  rules: Rules;
  summary: {
    tiles: number;
    approved: number;
    rejected: number;
    review: number;
    approval_rate: number | null;
    avg_usable: number | null;
    min_usable: number | null;
    max_usable: number | null;
    value: number;
    errors_avoided: number;
    overrides: number;
    avg_ms: number | null;
    models: string[];
  };
  audit: AuditEntry[];
  generated_at: string;
}

export interface Feedback {
  id: number;
  inspection_id: number;
  ts: string;
  inspector: string;
  model_verdict: Verdict;
  human_verdict: Verdict;
  usable_fraction: number | null;
  note: string | null;
  photo_url: string | null;
  agrees: boolean;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  dashboard: (days = 30) => request<Dashboard>(`/api/dashboard?days=${days}`),
  model: () => request<ModelInfo>("/api/model"),
  reloadModel: (checkpoint: string | null, actor: string) =>
    request<ModelInfo>("/api/model/reload", json("POST", { checkpoint, actor })),
  uploadModel: (file: File, actor: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("actor", actor);
    return request<ModelInfo>("/api/model/upload", { method: "POST", body: fd });
  },
  settings: () => request<Settings>("/api/settings"),
  saveSettings: (patch: Partial<{ rules: Partial<Rules>; model: Partial<Settings["model"]>; image_source: string }>,
    actor: string, rescore = false) =>
    request<Settings & { rescored?: { tiles: number; changed: number } }>(
      "/api/settings", json("PUT", { ...patch, actor, rescore_open_lots: rescore })),
  preview: (rules: Partial<Rules>, days = 30) => request<Preview>(`/api/settings/preview?days=${days}`, json("POST", rules)),
  resetDemo: (actor: string, image_dir?: string) =>
    request<{ tiles: number; lots: number; photos: number; image_dir: string }>(
      "/api/admin/reset-demo", json("POST", { actor, image_dir: image_dir || null })),
  lots: (status?: string, q?: string) => {
    const p = new URLSearchParams();
    if (status) p.set("status", status);
    if (q) p.set("q", q);
    return request<Lot[]>(`/api/lots?${p}`);
  },
  lot: (id: string) => request<Lot>(`/api/lots/${encodeURIComponent(id)}`),
  createLot: (body: { supplier: string; tile_type: string; received_date?: string; reference?: string; actor: string }) =>
    request<Lot>("/api/lots", json("POST", body)),
  decideLot: (id: string, decision: "accept" | "reject" | "reopen", actor: string, note?: string, force = false) =>
    request<Lot>(`/api/lots/${encodeURIComponent(id)}/decision`, json("POST", { decision, actor, note, force })),
  lotReport: (id: string) => request<LotReport>(`/api/lots/${encodeURIComponent(id)}/report`),
  lotCsvUrl: (id: string) => `/api/lots/${encodeURIComponent(id)}/report.csv`,
  meta: () => request<{ suppliers: string[]; tile_types: string[] }>("/api/meta"),
  inspect: (lotId: string, file: File, source = "upload") => {
    const fd = new FormData();
    fd.append("lot_id", lotId);
    fd.append("source", source);
    fd.append("files", file);
    return request<Inspection[]>("/api/inspections", { method: "POST", body: fd });
  },
  inspections: (params: { lot_id?: string; verdict?: string; limit?: number; offset?: number } = {}) => {
    const p = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && p.set(k, String(v)));
    return request<{ total: number; items: Inspection[] }>(`/api/inspections?${p}`);
  },
  inspection: (id: number) => request<Inspection>(`/api/inspections/${id}`),
  review: (id: number, verdict: "APPROVE" | "REJECT", inspector: string, note?: string) =>
    request<Inspection>(`/api/inspections/${id}/review`, json("POST", { verdict, inspector, note })),
  reviewQueue: () => request<{ total: number; items: Inspection[] }>("/api/review-queue"),
  audit: (limit = 100, entity?: string) =>
    request<AuditEntry[]>(`/api/audit?limit=${limit}${entity ? `&entity=${entity}` : ""}`),
  feedback: () => request<Feedback[]>("/api/feedback"),
  feedbackCsvUrl: "/api/feedback/export.csv",
};
