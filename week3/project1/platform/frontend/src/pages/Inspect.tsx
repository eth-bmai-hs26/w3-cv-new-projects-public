import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Inspection, type Lot } from "../api";
import { CompareImage } from "../components/CompareImage";
import { IconCamera, IconUpload } from "../components/Icons";
import { InspectionDrawer } from "../components/InspectionDrawer";
import { Ruler } from "../components/Ruler";
import { PageHeader } from "../components/Shell";
import { VerdictTag } from "../components/Verdict";
import { money, ms, num, pct } from "../lib/format";
import { useInspector, useModel, useToast } from "../lib/hooks";

type Job = { key: string; file: File; thumb: string; state: "waiting" | "working" | "done" | "error"; result?: Inspection; error?: string };

const today = () => new Date().toISOString().slice(0, 10);

function LotPicker({ lot, setLot }: { lot: Lot | null; setLot: (l: Lot | null) => void }) {
  const [lots, setLots] = useState<Lot[]>([]);
  const [meta, setMeta] = useState<{ suppliers: string[]; tile_types: string[] }>({ suppliers: [], tile_types: [] });
  const [mode, setMode] = useState<"pick" | "new">("pick");
  const [form, setForm] = useState({ supplier: "", tile_type: "", received_date: today(), reference: "" });
  const { name } = useInspector();
  const toast = useToast();

  useEffect(() => {
    api.lots("open").then((ls) => {
      setLots(ls);
      if (!ls.length) setMode("new");
    });
    api.meta().then(setMeta);
  }, []);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const l = await api.createLot({ ...form, reference: form.reference || undefined, actor: name || "inspector" });
      setLots((ls) => [l, ...ls]);
      setLot(l);
      setMode("pick");
      toast.push(`Lot ${l.id} created`);
    } catch (err) {
      toast.push((err as Error).message, "error");
    }
  };

  return (
    <section className="panel lot-picker">
      <header className="panel-head">
        <h2>Shipment</h2>
        <div className="seg">
          <button className={mode === "pick" ? "on" : ""} onClick={() => setMode("pick")} disabled={!lots.length}>Open lot</button>
          <button className={mode === "new" ? "on" : ""} onClick={() => setMode("new")}>New lot</button>
        </div>
      </header>
      {mode === "pick" ? (
        <div className="lot-pick">
          <label className="field">
            <span>Tiles go into</span>
            <select value={lot?.id ?? ""} onChange={(e) => setLot(lots.find((l) => l.id === e.target.value) ?? null)}>
              <option value="">Choose an open lot</option>
              {lots.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.id}, {l.supplier}, {l.tile_type} ({l.n} tiles)
                </option>
              ))}
            </select>
          </label>
          {lot && (
            <p className="lot-line">
              Received {lot.received_date}{lot.reference ? `, delivery note ${lot.reference}` : ""}. {num(lot.n)} tiles inspected so far.{" "}
              <Link to={`/lots/${lot.id}`}>Open lot</Link>
            </p>
          )}
        </div>
      ) : (
        <form className="form-grid" onSubmit={create}>
          <label className="field">
            <span>Supplier</span>
            <input list="suppliers" required value={form.supplier} onChange={(e) => setForm({ ...form, supplier: e.target.value })} placeholder="e.g. Rückbau Zürich AG" />
            <datalist id="suppliers">{meta.suppliers.map((s) => <option key={s} value={s} />)}</datalist>
          </label>
          <label className="field">
            <span>Tile type</span>
            <input list="types" required value={form.tile_type} onChange={(e) => setForm({ ...form, tile_type: e.target.value })} placeholder="e.g. Terracotta 30×30" />
            <datalist id="types">{meta.tile_types.map((s) => <option key={s} value={s} />)}</datalist>
          </label>
          <label className="field">
            <span>Received</span>
            <input type="date" required value={form.received_date} onChange={(e) => setForm({ ...form, received_date: e.target.value })} />
          </label>
          <label className="field">
            <span>Delivery note</span>
            <input value={form.reference} onChange={(e) => setForm({ ...form, reference: e.target.value })} placeholder="optional" />
          </label>
          <div className="form-actions">
            <button className="btn btn-primary" type="submit">Create lot</button>
          </div>
        </form>
      )}
    </section>
  );
}

function Conveyor({ jobs }: { jobs: Job[] }) {
  const done = jobs.filter((j) => j.state === "done" || j.state === "error").length;
  const belt = useRef<HTMLDivElement>(null);
  const working = jobs.findIndex((j) => j.state === "working");
  const focus = Math.min(working >= 0 ? working : done, jobs.length - 1);   // keep the last tile under the camera
  useEffect(() => {
    const el = belt.current?.children[focus] as HTMLElement | undefined;
    if (el && belt.current) belt.current.scrollTo({ left: el.offsetLeft - belt.current.clientWidth / 2 + el.clientWidth / 2, behavior: "smooth" });
  }, [focus]);
  if (!jobs.length) return null;
  return (
    <section className="conveyor" aria-label="Batch progress">
      <div className="conveyor-head">
        <b>{done === jobs.length ? "Batch finished" : "Inspecting"}</b>
        <span>{num(done)} of {num(jobs.length)} tiles</span>
        <div className="progress"><i style={{ width: `${(100 * done) / jobs.length}%` }} /></div>
      </div>
      <div className="belt">
        <div className="belt-track" ref={belt}>
          {jobs.map((j) => (
            <div key={j.key} className={`belt-tile bt-${j.state} ${j.result ? `bt-${j.result.verdict_final.toLowerCase()}` : ""}`}
                 title={j.state === "error" ? `${j.file.name}: ${j.error}` : j.file.name}>
              <img src={j.result?.photo_url ?? j.thumb} alt="" />   {/* server JPEG once inspected: browsers can't all show HEIC */}
              {j.result && <span className="bt-stamp">{j.result.usable_pct == null ? "?" : `${Math.round(j.result.usable_pct)}%`}</span>}
              {j.state === "error" && <span className="bt-stamp">!</span>}
            </div>
          ))}
        </div>
        <div className="gate" aria-hidden><span>Camera</span></div>
      </div>
    </section>
  );
}

export default function Inspect() {
  const [lot, setLot] = useState<Lot | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [over, setOver] = useState(false);
  const [open, setOpen] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const camRef = useRef<HTMLInputElement>(null);
  const running = useRef(false);
  const toast = useToast();
  const { info } = useModel();

  const addFiles = (files: FileList | File[], source = "upload") => {
    if (!lot) return toast.push("Choose or create the lot these tiles belong to first", "error");
    const imgs = Array.from(files).filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png|webp|bmp)$/i.test(f.name));
    if (!imgs.length) return toast.push("Those files are not photos", "error");
    const add: Job[] = imgs.map((f, i) => ({ key: `${Date.now()}-${i}-${f.name}`, file: f, thumb: URL.createObjectURL(f), state: "waiting" }));
    setJobs((js) => [...js, ...add]);
    run(lot.id, add, source);
  };

  const run = async (lotId: string, add: Job[], source: string) => {
    const queue = [...add];
    const worker = async () => {
      while (queue.length) {
        const job = queue.shift()!;
        setJobs((js) => js.map((j) => (j.key === job.key ? { ...j, state: "working" } : j)));
        try {
          const [res] = await api.inspect(lotId, job.file, source);
          setJobs((js) => js.map((j) => (j.key === job.key ? { ...j, state: "done", result: res } : j)));
        } catch (e) {
          setJobs((js) => js.map((j) => (j.key === job.key ? { ...j, state: "error", error: (e as Error).message } : j)));
          toast.push(`${job.file.name} could not be inspected: ${(e as Error).message}`, "error");
        }
      }
    };
    running.current = true;
    await Promise.all([worker(), worker()]);
    running.current = false;
    api.lot(lotId).then(setLot).catch(() => {});        // refresh the lot's counts
  };

  const results = jobs.filter((j) => j.result).map((j) => j.result!);
  const summary = useMemo(() => {
    const c = { APPROVE: 0, REVIEW: 0, REJECT: 0 };
    let value = 0;
    let t = 0;
    results.forEach((r) => {
      c[r.verdict_final]++;
      value += r.value;
      t += r.t_total;
    });
    return { c, value, avg: results.length ? t / results.length : 0 };
  }, [results]);

  return (
    <div className="page">
      <PageHeader title="Inspect tiles" sub="Photograph each tile on the belt. The model marks the usable area and decides whether it can be sold." />

      <LotPicker lot={lot} setLot={setLot} />

      <section
        className={`drop ${over ? "drop-over" : ""} ${lot ? "" : "drop-off"}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          addFiles(e.dataTransfer.files);
        }}
      >
        <div className="drop-inner">
          <IconUpload size={30} />
          <p className="drop-title">{lot ? `Drop tile photos for ${lot.id} here` : "Choose a lot, then drop tile photos here"}</p>
          <p className="drop-sub">JPG or PNG, one tile per photo, as many as you like.</p>
          <div className="btn-row">
            <button className="btn btn-primary" onClick={() => fileRef.current?.click()} disabled={!lot}>
              <IconUpload size={16} /> Choose photos
            </button>
            <button className="btn" onClick={() => camRef.current?.click()} disabled={!lot}>
              <IconCamera size={16} /> Take photo
            </button>
          </div>
          {info?.mode === "demo" && <p className="drop-demo">Demo mode: verdicts come from a simple heuristic until a trained model is loaded in Settings.</p>}
        </div>
        <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={(e) => e.target.files && addFiles(e.target.files)} />
        <input ref={camRef} type="file" accept="image/*" capture="environment" hidden onChange={(e) => e.target.files && addFiles(e.target.files, "camera")} />
      </section>

      <Conveyor jobs={jobs} />

      {results.length > 0 && (
        <section className="batch-summary">
          <div><b>{num(summary.c.APPROVE)}</b> approved</div>
          <div><b>{num(summary.c.REVIEW)}</b> for review</div>
          <div><b>{num(summary.c.REJECT)}</b> rejected</div>
          <div><b>{money(summary.value)}</b> resale value</div>
          <div><b>{ms(summary.avg)}</b> per tile</div>
          {summary.c.REVIEW > 0 && <Link className="btn btn-small" to="/review">Open review queue</Link>}
        </section>
      )}

      <div className="results">
        {jobs.filter((j) => j.state === "error").map((j) => (
          <article key={j.key} className="result result-error">
            <img src={j.thumb} alt="" />
            <div className="result-body"><b>{j.file.name}</b><p>{j.error}</p></div>
          </article>
        ))}
        {[...results].reverse().map((r) => (
          <article key={r.id} className={`result r-${r.verdict_final.toLowerCase()}`}>
            <CompareImage photo={r.photo_url} overlay={r.overlay_url} alt={`Tile ${r.id}`} start={0} />
            <div className="result-body">
              <div className="result-top">
                <VerdictTag verdict={r.verdict_final} />
                <span className="result-pct">{r.usable_pct == null ? "No tile" : `${r.usable_pct.toFixed(1)}%`}<small>usable</small></span>
              </div>
              <Ruler value={r.usable_fraction} threshold={r.threshold} band={r.review_band} compact />
              <p className="reason">{r.error ?? r.reason}</p>
              <dl className="result-facts">
                <div><dt>Confidence</dt><dd>{pct(r.confidence)}</dd></div>
                <div><dt>Time</dt><dd>{ms(r.t_total)}</dd></div>
                <div><dt>Value</dt><dd>{money(r.value)}</dd></div>
              </dl>
              {r.warnings?.filter((w) => !w.startsWith("Demo")).map((w) => <p key={w} className="warn-line">{w}</p>)}
              <button className="link-btn" onClick={() => setOpen(r.id)}>Details and override</button>
            </div>
          </article>
        ))}
      </div>

      <InspectionDrawer id={open} onClose={() => setOpen(null)} onChanged={(u) => setJobs((js) => js.map((j) => (j.result?.id === u.id ? { ...j, result: { ...j.result, ...u } } : j)))} />
    </div>
  );
}
