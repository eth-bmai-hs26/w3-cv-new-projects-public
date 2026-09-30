import { Fragment, useEffect, useRef, useState } from "react";
import { api, type AuditEntry, type ModelInfo, type Preview, type Rules } from "../api";
import { IconUpload } from "../components/Icons";
import { Ruler } from "../components/Ruler";
import { PageHeader } from "../components/Shell";
import { dateTime, money, num, pct } from "../lib/format";
import { useAsync, useInspector, useModel, useToast } from "../lib/hooks";

function Slider({ label, value, min, max, step, onChange, fmt, help }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string; help?: string;
}) {
  return (
    <label className="slider">
      <span className="slider-top"><span>{label}</span><b>{fmt(value)}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {help && <small>{help}</small>}
    </label>
  );
}

function NumberField({ label, value, onChange, step = 0.01, suffix, help }: { label: string; value: number; onChange: (v: number) => void; step?: number; suffix?: string; help?: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      <span className="input-suffix">
        <input type="number" min={0} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
        {suffix && <em>{suffix}</em>}
      </span>
      {help && <small>{help}</small>}
    </label>
  );
}

function PreviewTable({ p }: { p: Preview }) {
  const rows: [string, (s: Preview["current"]) => string, (s: Preview["current"]) => number][] = [
    ["Approved", (s) => `${num(s.counts.APPROVE)} (${pct(s.approval_rate)})`, (s) => s.counts.APPROVE],
    ["To review", (s) => `${num(s.counts.REVIEW)} (${pct(s.review_rate)})`, (s) => s.counts.REVIEW],
    ["Rejected", (s) => num(s.counts.REJECT), (s) => s.counts.REJECT],
    ["Resale value", (s) => money(s.recovered_value), (s) => s.recovered_value],
    ["Inspector time for reviews", (s) => money(s.review_cost), (s) => s.review_cost],
  ];
  return (
    <table className="table preview-table">
      <thead><tr><th>Last {p.days} days, {num(p.tiles)} tiles</th><th className="num">Current rule</th><th className="num">With these settings</th></tr></thead>
      <tbody>
        {rows.map(([l, f, raw]) => {
          const changed = raw(p.current) !== raw(p.proposed);
          return (
            <tr key={l}>
              <td>{l}</td>
              <td className="num">{f(p.current)}</td>
              <td className={`num ${changed ? "changed" : ""}`}>{f(p.proposed)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function ModelCard({ info }: { info: ModelInfo }) {
  const demo = info.mode === "demo";
  const metrics = Object.entries(info.metrics || {});
  return (
    <div className={`model-card ${demo ? "model-demo" : ""}`}>
      {demo && <span className="hazard" aria-hidden />}
      <div className="mc-head">
        <b>{info.name}</b>
        <span className="mc-kind">{demo ? "Demo mode" : info.kind === "torchscript" ? "TorchScript" : "Notebook model (dill)"}</span>
      </div>
      <dl className="facts">
        {!demo && <><dt>Version</dt><dd>{info.version}</dd></>}
        {info.architecture && <><dt>Architecture</dt><dd>{info.architecture}</dd></>}
        <dt>Output</dt><dd>{info.classes === 3 ? "3 classes: belt, usable, damaged" : "1 channel (legacy): usable only"}</dd>
        {!demo && <><dt>Parameters</dt><dd>{num(info.params)}</dd></>}
        <dt>Runs on</dt><dd>{info.device.toUpperCase()}</dd>
        {info.checkpoint && <><dt>File</dt><dd className="path">{info.checkpoint}{info.size_mb ? ` (${info.size_mb} MB)` : ""}</dd></>}
        <dt>Loaded</dt><dd>{dateTime(info.loaded_at)}</dd>
        {metrics.map(([k, v]) => (<Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{typeof v === "number" ? (v <= 1 ? pct(v, 1) : v.toFixed(3)) : v}</dd></Fragment>))}
      </dl>
      {info.warning && <p className="warn-line">{info.warning}</p>}
      {demo && <p className="mc-note">{info.description}</p>}
    </div>
  );
}

export default function Settings() {
  const { data: settings, reload } = useAsync(() => api.settings(), []);
  const { data: history, reload: reloadHistory } = useAsync(() => api.audit(200), []);
  const { info, setInfo } = useModel();
  const { name } = useInspector();
  const toast = useToast();
  const [rules, setRules] = useState<Rules | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [rescore, setRescore] = useState(true);
  const [ckpt, setCkpt] = useState("");
  const [device, setDevice] = useState("auto");
  const [imageDir, setImageDir] = useState("");
  const [confirmReset, setConfirmReset] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const upRef = useRef<HTMLInputElement>(null);
  const actor = name || "settings";

  useEffect(() => {
    if (!settings) return;
    setRules(settings.rules);
    setCkpt(settings.model.checkpoint ?? "");
    setDevice(settings.model.device ?? "auto");
    setImageDir(settings.image_source);
  }, [settings]);

  useEffect(() => {
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  }, [settings]);

  // what-if preview, debounced
  useEffect(() => {
    if (!rules) return;
    const t = setTimeout(() => api.preview(rules).then(setPreview).catch(() => {}), 250);
    return () => clearTimeout(t);
  }, [rules]);

  if (!settings || !rules) return <div className="page"><PageHeader title="Settings" /><div className="skeleton" style={{ height: 400 }} /></div>;

  const set = <K extends keyof Rules>(k: K, v: Rules[K]) => setRules({ ...rules, [k]: v });
  const dirty = JSON.stringify(rules) !== JSON.stringify(settings.rules);

  const saveRules = async () => {
    setBusy("rules");
    try {
      const out = await api.saveSettings({ rules }, actor, rescore);
      toast.push(out.rescored ? `Rule saved. ${out.rescored.changed} of ${out.rescored.tiles} tiles in open lots changed verdict.` : "Rule saved");
      reload();
      reloadHistory();
    } catch (e) {
      toast.push((e as Error).message, "error");
    } finally {
      setBusy(null);
    }
  };

  const loadModel = async (path: string | null) => {
    setBusy("model");
    try {
      await api.saveSettings({ model: { checkpoint: path ?? "", device } }, actor);
      const m = await api.reloadModel(path ?? "", actor);
      setInfo(m);
      toast.push(m.mode === "demo" ? (path ? `Could not load the model: ${m.warning}` : "Switched to demo mode") : `Loaded ${m.name}`, m.mode === "demo" && path ? "error" : "ok");
      reload();
      reloadHistory();
    } catch (e) {
      toast.push((e as Error).message, "error");
    } finally {
      setBusy(null);
    }
  };

  const upload = async (f: File) => {
    setBusy("upload");
    try {
      const m = await api.uploadModel(f, actor);
      setInfo(m);
      toast.push(m.mode === "demo" ? `Upload saved, but it did not load: ${m.warning}` : `Loaded ${m.name}`, m.mode === "demo" ? "error" : "ok");
      reload();
      reloadHistory();
    } catch (e) {
      toast.push((e as Error).message, "error");
    } finally {
      setBusy(null);
    }
  };

  const reset = async () => {
    setBusy("reset");
    try {
      await api.saveSettings({ image_source: imageDir }, actor);
      const r = await api.resetDemo(actor, imageDir);
      toast.push(`Demo data rebuilt: ${r.tiles} inspections in ${r.lots} lots from ${r.photos} photos`);
      setConfirmReset(false);
      reload();
      reloadHistory();
    } catch (e) {
      toast.push((e as Error).message, "error");
    } finally {
      setBusy(null);
    }
  };

  const settingsHistory = (history ?? []).filter((a: AuditEntry) => a.entity === "settings" || a.entity === "model" || a.entity === "system").slice(0, 12);

  return (
    <div className="page">
      <PageHeader title="Settings" sub="How the station decides, what tiles are worth, and which model it runs." />

      <section className="panel settings-block" id="rule">
        <header className="panel-head"><h2>Decision rule</h2>{dirty && <span className="pill">Unsaved changes</span>}</header>
        <div className="settings-grid">
          <div>
            <p className="panel-note">A tile is approved when its usable share reaches the threshold. Tiles close to the line, or where the model is unsure, go to the review queue.</p>
            <Slider label="Approve from" value={rules.threshold} min={0.5} max={0.99} step={0.01} onChange={(v) => set("threshold", v)} fmt={(v) => `${Math.round(v * 100)}% usable`} />
            <Slider label="Review band" value={rules.review_band} min={0} max={0.15} step={0.01} onChange={(v) => set("review_band", v)} fmt={(v) => (v ? `±${Math.round(v * 100)} pts` : "off")}
              help={rules.review_band ? `Tiles between ${pct(rules.threshold - rules.review_band)} and ${pct(rules.threshold + rules.review_band)} go to a person.` : "Every tile is decided automatically unless the model is unsure."} />
            <Slider label="Minimum confidence" value={rules.min_confidence} min={0} max={0.9} step={0.05} onChange={(v) => set("min_confidence", v)} fmt={(v) => pct(v)}
              help="Below this, a tile goes to review even when it is far from the line." />
            <div className="rule-ruler">
              <Ruler threshold={rules.threshold} band={rules.review_band} />
            </div>
          </div>
          <div>
            <h3>Effect on recent tiles</h3>
            {preview ? <PreviewTable p={preview} /> : <div className="skeleton" style={{ height: 180 }} />}
            <label className="check">
              <input type="checkbox" checked={rescore} onChange={(e) => setRescore(e.target.checked)} />
              Also re-check undecided tiles in open lots
            </label>
            <div className="btn-row">
              <button className="btn btn-primary" disabled={!dirty || busy === "rules"} onClick={saveRules}>Save rule</button>
              <button className="btn btn-ghost" disabled={!dirty} onClick={() => setRules(settings.rules)}>Discard</button>
              <button className="btn btn-ghost" onClick={() => setRules({ ...rules, threshold: settings.defaults.threshold, review_band: settings.defaults.review_band, min_confidence: settings.defaults.min_confidence })}>Use defaults</button>
            </div>
          </div>
        </div>
      </section>

      <section className="panel settings-block" id="money">
        <header className="panel-head"><h2>Prices and costs</h2></header>
        <div className="form-grid">
          <label className="field">
            <span>Price basis</span>
            <select value={rules.price_basis} onChange={(e) => set("price_basis", e.target.value as Rules["price_basis"])}>
              <option value="per_tile">Per tile</option>
              <option value="per_m2">Per square metre</option>
            </select>
          </label>
          {rules.price_basis === "per_tile" ? (
            <NumberField label="Resale price per tile" value={rules.price_per_tile} onChange={(v) => set("price_per_tile", v)} suffix="€" step={0.1} />
          ) : (
            <>
              <NumberField label="Resale price per m²" value={rules.price_per_m2} onChange={(v) => set("price_per_m2", v)} suffix="€/m²" step={1} />
              <NumberField label="Tile size" value={rules.tile_area_m2} onChange={(v) => set("tile_area_m2", v)} suffix="m²" step={0.01} help="0.09 m² is a 30×30 cm tile" />
            </>
          )}
          <NumberField label="Cost of a defective tile sold" value={rules.cost_bad_tile_shipped} onChange={(v) => set("cost_bad_tile_shipped", v)} suffix="€" step={0.5} help="Return, replacement and goodwill" />
          <NumberField label="Cost of checking one tile by hand" value={rules.cost_manual_check} onChange={(v) => set("cost_manual_check", v)} suffix="€" step={0.05} />
        </div>
        <div className="btn-row">
          <button className="btn btn-primary" disabled={!dirty || busy === "rules"} onClick={saveRules}>Save prices</button>
          <span className="muted">New prices apply to tiles inspected from now on.</span>
        </div>
      </section>

      <section className="panel settings-block" id="model">
        <header className="panel-head"><h2>Model</h2></header>
        <div className="settings-grid">
          {info && <ModelCard info={info} />}
          <div className="model-actions">
            <label className="field">
              <span>Model file on this computer</span>
              <input value={ckpt} onChange={(e) => setCkpt(e.target.value)} placeholder={settings.default_checkpoint} spellCheck={false} />
              <small>A .pt file saved with save_model() in the notebook, or a TorchScript file.</small>
            </label>
            <label className="field">
              <span>Run on</span>
              <select value={device} onChange={(e) => setDevice(e.target.value)}>
                <option value="auto">Automatic (GPU if available)</option>
                <option value="cpu">CPU</option>
                <option value="cuda">NVIDIA GPU (CUDA)</option>
                <option value="mps">Apple GPU (MPS)</option>
              </select>
            </label>
            <div className="btn-row">
              <button className="btn btn-primary" disabled={busy === "model"} onClick={() => loadModel(ckpt || null)}>Load model</button>
              <button className="btn" disabled={busy === "model"} onClick={() => { setCkpt(settings.default_checkpoint); loadModel(settings.default_checkpoint); }}>Use shipped model</button>
              <button className="btn btn-ghost" disabled={busy === "model"} onClick={() => { setCkpt(""); loadModel(null); }}>Demo mode</button>
            </div>
            <div className="upload-line">
              <button className="btn" disabled={busy === "upload"} onClick={() => upRef.current?.click()}>
                <IconUpload size={16} /> {busy === "upload" ? "Uploading…" : "Upload your model .pt"}
              </button>
              <small>Downloaded from Colab after training. It is stored in platform/models/uploads and loaded at once.</small>
              <input ref={upRef} type="file" accept=".pt,.pth,.ts" hidden onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
            </div>
          </div>
        </div>
      </section>

      <section className="panel settings-block" id="demo">
        <header className="panel-head"><h2>Demo data</h2></header>
        <p className="panel-note">Rebuild 30 days of history by inspecting photos from a folder with the current model. This deletes all lots, inspections, reviews and the audit trail.</p>
        <div className="form-grid">
          <label className="field field-wide">
            <span>Photo folder</span>
            <input value={imageDir} onChange={(e) => setImageDir(e.target.value)} spellCheck={false} />
            <small>For example the course dataset's original/ folder. If it is empty, placeholder tiles are generated.</small>
          </label>
        </div>
        <div className="btn-row">
          {!confirmReset ? (
            <button className="btn btn-reject-outline" onClick={() => setConfirmReset(true)}>Reset demo data</button>
          ) : (
            <>
              <button className="btn btn-reject" disabled={busy === "reset"} onClick={reset}>{busy === "reset" ? "Rebuilding… (about 30 s)" : "Delete everything and rebuild"}</button>
              <button className="btn btn-ghost" onClick={() => setConfirmReset(false)}>Keep current data</button>
            </>
          )}
        </div>
      </section>

      <section className="panel settings-block">
        <header className="panel-head"><h2>Change history</h2></header>
        <ul className="audit-list">
          {settingsHistory.map((a) => (
            <li key={a.id}>
              <span>{dateTime(a.ts)}</span> <b>{a.actor}</b> {describe(a)}
            </li>
          ))}
          {!settingsHistory.length && <li className="muted">No changes yet.</li>}
        </ul>
      </section>
    </div>
  );
}

function describe(a: AuditEntry): string {
  const d = (a.detail ?? {}) as Record<string, any>;
  switch (a.action) {
    case "settings.update": {
      const r = d.rules;
      if (r) {
        const parts: string[] = [];
        for (const k of Object.keys(r.to)) if (r.to[k] !== r.from[k]) parts.push(`${k.replace(/_/g, " ")} ${r.from[k]} → ${r.to[k]}`);
        return `changed ${parts.join(", ")}`;
      }
      if (d.model) return `set the model file to ${d.model.to.checkpoint || "none (demo)"}`;
      if (d.image_source) return `set the photo folder to ${d.image_source.to}`;
      return "changed settings";
    }
    case "model.load":
      return d.mode === "demo" ? `switched to demo mode${d.warning ? ` (${d.warning})` : ""}` : `loaded ${d.checkpoint}`;
    case "inspection.rescore":
      return `re-checked ${d.tiles} tiles in open lots, ${d.changed} changed verdict`;
    case "demo.seed":
      return `rebuilt demo data: ${d.tiles} tiles in ${d.lots} lots`;
    default:
      return a.action;
  }
}
