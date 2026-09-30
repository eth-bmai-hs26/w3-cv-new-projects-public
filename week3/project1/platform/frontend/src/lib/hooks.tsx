import { api, type ModelInfo } from "../api";
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

/** Load data once (and again when deps change); `reload` refetches without clearing. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const d = await fnRef.current();
      setData(d);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading, reload, setData };
}

function readLocal(key: string, fallback: string) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

// ── inspector name (no login: the name goes into the audit trail) ───────────
type InspectorCtx = { name: string; setName: (n: string) => void };
const InspectorContext = createContext<InspectorCtx>({ name: "", setName: () => {} });

export function InspectorProvider({ children }: { children: ReactNode }) {
  const [name, setNameState] = useState(() => readLocal("inspector", ""));
  const setName = (n: string) => {
    setNameState(n);
    try {
      localStorage.setItem("inspector", n);
    } catch {
      /* private mode */
    }
  };
  return <InspectorContext.Provider value={{ name, setName }}>{children}</InspectorContext.Provider>;
}

export const useInspector = () => useContext(InspectorContext);

// ── toasts ──────────────────────────────────────────────────────────────────
type Toast = { id: number; text: string; tone: "ok" | "error" | "info" };
type ToastCtx = { push: (text: string, tone?: Toast["tone"]) => void };
const ToastContext = createContext<ToastCtx>({ push: () => {} });

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((text: string, tone: Toast["tone"] = "ok") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, text, tone }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4200);
  }, []);
  return (
    <ToastContext.Provider value={{ push }}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.tone}`}>
            {t.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export const useToast = () => useContext(ToastContext);

// ── shared model info (rail + pages) ────────────────────────────────────────
type ModelCtx = { info: ModelInfo | null; refresh: () => void; setInfo: (m: ModelInfo) => void };
const ModelContext = createContext<ModelCtx>({ info: null, refresh: () => {}, setInfo: () => {} });

export function ModelProvider({ children }: { children: ReactNode }) {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const refresh = useCallback(() => {
    api.model().then(setInfo).catch(() => {});
  }, []);
  useEffect(refresh, [refresh]);
  return <ModelContext.Provider value={{ info, refresh, setInfo }}>{children}</ModelContext.Provider>;
}

export const useModel = () => useContext(ModelContext);
