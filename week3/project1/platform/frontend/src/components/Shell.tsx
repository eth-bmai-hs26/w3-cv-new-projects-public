import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "../api";
import { useInspector, useModel } from "../lib/hooks";
import { IconCamera, IconGauge, IconPallet, IconReview, IconSliders } from "./Icons";

const NAV = [
  { to: "/", label: "Dashboard", icon: IconGauge, end: true },
  { to: "/inspect", label: "Inspect", icon: IconCamera },
  { to: "/lots", label: "Lots", icon: IconPallet },
  { to: "/review", label: "Review", icon: IconReview, badge: true },
  { to: "/settings", label: "Settings", icon: IconSliders },
];

function useQueueCount() {
  const [n, setN] = useState<number | null>(null);
  const loc = useLocation();
  useEffect(() => {
    let alive = true;
    const load = () => api.reviewQueue().then((q) => alive && setN(q.total)).catch(() => {});
    load();
    const t = setInterval(load, 20000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [loc.pathname]);
  return n;
}

function StationStatus() {
  const { info } = useModel();
  if (!info) return null;
  const demo = info.mode === "demo";
  return (
    <NavLink to="/settings#model" className={`station ${demo ? "station-demo" : ""}`}>
      {demo && <span className="hazard" aria-hidden />}
      <span className="station-k">{demo ? "Demo mode" : "Model"}</span>
      <span className="station-v">{demo ? "Heuristic, not a trained model" : info.name}</span>
      {!demo && (
        <span className="station-meta">
          v{info.version.slice(0, 7)} on {info.device.toUpperCase()}
        </span>
      )}
    </NavLink>
  );
}

function InspectorField() {
  const { name, setName } = useInspector();
  return (
    <label className={`inspector ${name ? "" : "inspector-missing"}`}>
      <span>Inspector</span>
      <input value={name} placeholder="Your name" onChange={(e) => setName(e.target.value)} aria-label="Inspector name" />
    </label>
  );
}

export function Shell() {
  const queue = useQueueCount();
  return (
    <div className="app">
      <aside className="rail">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <i /><i /><i /><i />
          </span>
          <span className="brand-text">
            <b>Tile inspection</b>
            <small>Reclaimed floor tiles, line 1</small>
          </span>
        </div>
        <nav className="nav">
          {NAV.map(({ to, label, icon: Icon, end, badge }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
              <Icon />
              <span>{label}</span>
              {badge && queue ? <em className="nav-badge">{queue}</em> : null}
            </NavLink>
          ))}
        </nav>
        <StationStatus />
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}

export function PageHeader({ title, sub, children }: { title: string; sub?: ReactNode; children?: ReactNode }) {
  return (
    <header className="page-head">
      <div className="page-title">
        <h1>{title}</h1>
        {sub && <p>{sub}</p>}
      </div>
      <div className="page-actions">
        {children}
        <InspectorField />
      </div>
    </header>
  );
}
