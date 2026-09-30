"""
Self-contained HTML widgets for the warehouse notebook.

Every widget renders to one HTML string (inline CSS/SVG/vanilla JS, images as
base64 data URIs, no network access), so it works in Colab, JupyterLab and
VS Code and survives in saved notebook outputs.

    TrainingDashboard      live loss / accuracy curves while a model trains
    Scoreboard             accuracy vs. targets, with an attempt history
    test_inspector         step through test tiles: photo | truth | prediction | errors
    confusion_explorer     clickable confusion matrix -> gallery of those tiles
    threshold_explorer     true vs predicted usable share, draggable threshold, business cost
    rf_visualizer          receptive-field squares over a tile photo + size histogram
    compare_gallery        the same hard images side by side for several models
    architecture_diagram   block diagram of a model with the tensor shape after each stage
    check_dataset          compare a hand-written dataset against the reference loader
"""

import base64
import html
import io
import json
import os
import time
import uuid

import numpy as np
from IPython.display import HTML, display
from PIL import Image

from week3.project1.data import BACKGROUND, DAMAGED, USABLE, USABLE_THRESHOLD

# ── Design tokens ────────────────────────────────────────────────────────────

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FP_RGB = (227, 73, 72)      # predicted usable, actually damaged  (damage missed → sold)
FN_RGB = (42, 120, 214)     # predicted damaged, actually usable  (good surface wasted)
EDGE_RGB = (237, 161, 0)    # tile vs belt confused
CLASS_RGB = np.array([(59, 59, 56), (232, 232, 228), (208, 59, 59)], np.uint8)   # background, usable, damaged

_CSS = """
.wh{--surface:#fcfcfb;--plane:#f4f3ef;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--border:rgba(11,11,11,.10);--accent:#2a78d6;--accent-wash:rgba(42,120,214,.10);
--good:#0ca30c;--good-ink:#006300;--bad:#d03b3b;--fp:#e34948;--fn:#2a78d6;
color-scheme:light;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);
background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px;
margin:4px 0;max-width:1100px;box-sizing:border-box}
@media (prefers-color-scheme:dark){.wh{--surface:#1a1a19;--plane:#242422;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.12);--accent:#3987e5;
--accent-wash:rgba(57,135,229,.16);--good-ink:#0ca30c;color-scheme:dark}}
.wh *{box-sizing:border-box}
.wh h3{font-size:16px;font-weight:650;margin:0 0 2px}
.wh .sub{color:var(--ink2);font-size:13px;margin:0 0 12px}
.wh .muted{color:var(--muted)} .wh .ink2{color:var(--ink2)}
.wh .row{display:flex;flex-wrap:wrap;gap:12px;align-items:stretch}
.wh .tile{background:var(--plane);border-radius:10px;padding:10px 14px;min-width:120px;flex:1}
.wh .tile .lbl{font-size:12px;color:var(--ink2)} .wh .tile .val{font-size:22px;font-weight:650}
.wh .tile .note{font-size:12px;color:var(--muted)}
.wh .bar{height:6px;border-radius:3px;background:var(--grid);overflow:hidden}
.wh .bar>i{display:block;height:100%;background:var(--accent);border-radius:3px}
.wh .badge{white-space:nowrap;display:inline-flex;align-items:center;gap:4px;font-size:12px;font-weight:600;
padding:2px 8px;border-radius:999px;background:var(--plane)}
.wh .badge.ok{color:var(--good-ink)} .wh .badge.no{color:var(--bad)}
.wh .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:0}
.wh table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;background:transparent;color:inherit}
.wh tr,.wh tr:nth-child(odd),.wh tr:nth-child(even),.wh tr:hover{background:transparent!important}
.wh th{font-weight:600;color:var(--ink2);text-align:left;padding:4px 10px;border-bottom:1px solid var(--grid);background:transparent}
.wh td{padding:4px 10px;border-bottom:1px solid var(--grid)}
.wh button,.wh select{font:inherit;font-size:13px;color:var(--ink);background:var(--plane);
border:1px solid var(--border);border-radius:8px;padding:4px 10px;cursor:pointer}
.wh button.on{background:var(--accent);color:#fff;border-color:transparent}
.wh button:focus-visible,.wh select:focus-visible,.wh input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.wh input[type=range]{accent-color:var(--accent)}
.wh .panel{display:flex;flex-direction:column;gap:4px;align-items:center}
.wh .panel img{width:100%;max-width:230px;border-radius:6px;image-rendering:pixelated;display:block}
.wh .cap{font-size:12px;color:var(--ink2);text-align:center}
.wh .strip{display:flex;gap:4px;overflow-x:auto;padding:4px 0}
.wh .strip img{width:44px;height:44px;border-radius:4px;cursor:pointer;opacity:.75;border:2px solid transparent}
.wh .strip img.sel{opacity:1;border-color:var(--accent)}
.wh .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.wh .grid figure{margin:0;font-size:11px;color:var(--ink2);text-align:center}
.wh .grid img{width:100%;border-radius:6px;display:block;image-rendering:pixelated}
.wh svg text{fill:var(--muted);font-size:11px;font-family:inherit}
.wh svg .ink{fill:var(--ink)} .wh svg .ink2{fill:var(--ink2)}
.wh svg .gridline{stroke:var(--grid);stroke-width:1} .wh svg .axis{stroke:var(--axis);stroke-width:1}
.wh .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:4px 0}
.wh .sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-2px}
"""

_RESIZE_JS = """
(function(){try{if(window.google&&google.colab&&google.colab.output&&google.colab.output.setIframeHeight){
google.colab.output.setIframeHeight(0,true,{maxHeight:4000});}}catch(e){}})();
"""


def _js_data(obj):
    """
    JSON for embedding directly in a <script>: some front-ends (Colab in Safari) drop or rewrite
    separate JSON <script> data blocks, so widget data lives in the executable code itself.
    """
    return (json.dumps(obj).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _uid():
    return "wh" + uuid.uuid4().hex[:10]


def _guard(uid, script):
    """
    Run a widget script robustly: wait until the widget's HTML (and its JSON
    data) is attached, and show any error inside the widget instead of failing
    silently. Some front-ends (e.g. Colab output frames) may run a script before
    the markup around it is fully in the document.
    """
    return ("(function(){var U=%s,tries=0;"
            "function ready(){return !!document.getElementById(U)}"
            "function fail(e){var r=document.getElementById(U);if(!r)return;var d=document.createElement('pre');"
            "d.style.cssText='color:#d03b3b;white-space:pre-wrap;font-size:12px;margin-top:8px';"
            "d.textContent='Widget error: '+String(e)+(e&&e.stack?'\\n'+e.stack:'');r.appendChild(d)}"
            "function run(){if(!ready()){if(tries++<100){return setTimeout(run,50)}return fail('widget markup never appeared')}"
            "try{(function(){%s})()}catch(e){fail(e)}}run()})();") % (json.dumps(uid), script)


def _wrap(body, uid=None, script=""):
    uid = uid or _uid()
    js = f"<script>{_RESIZE_JS}{_guard(uid, script)}</script>" if script else f"<script>{_RESIZE_JS}</script>"
    return f"<style>{_CSS}</style><div class='wh' id='{uid}'>{body}</div>{js}"


def _show(markup, max_mb=4.0):
    mb = len(markup) / 1e6
    if mb > max_mb:
        print(f"⚠️ widget is {mb:.1f} MB — consider lowering max_items to keep the notebook small")
    display(HTML(markup))


def _esc(s):
    return html.escape(str(s))


# ── Image helpers ─────────────────────────────────────────────────────────────

def _to_uint8(arr):
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if arr.dtype != np.uint8:
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr


def encode_thumb(arr, size=128, quality=80, fmt="JPEG"):
    """numpy image (H, W) or (H, W, 3) -> data URI."""
    im = Image.fromarray(_to_uint8(arr))
    if size and im.size[0] != size:
        im = im.resize((size, size), Image.NEAREST if im.mode == "L" and fmt == "PNG" else Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=quality) if fmt == "JPEG" else im.save(buf, format=fmt, optimize=True)
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def _resize_bool(mask, size):
    if mask.shape[0] == size:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127


def _resize_cls(target, size):
    """Nearest-neighbour resize of a class map (H, W) uint8."""
    target = np.asarray(target).astype(np.uint8)
    if target.shape[0] == size:
        return target
    return np.asarray(Image.fromarray(target).resize((size, size), Image.NEAREST))


def class_rgb(target, size=None):
    """Colour a class map: belt = dark grey, usable = off-white, damaged = red."""
    target = _resize_cls(target, size) if size else np.asarray(target).astype(np.uint8)
    return CLASS_RGB[np.clip(target, 0, 2)]


def _usable(target):
    tile = (target != BACKGROUND).sum()
    return float((target == USABLE).sum() / tile) if tile else 0.0


def error_overlay(image, gt, pred, size=None):
    """
    RGB composite of a prediction against the truth (both class maps):
      correct usable = photo · correct damaged = photo tinted red · belt = dimmed photo
      red   = predicted usable, actually damaged  (the damage would be sold)
      blue  = predicted damaged, actually usable  (good surface wasted)
      amber = tile and belt confused
    """
    size = size or np.asarray(pred).shape[0]
    img = np.asarray(Image.fromarray(_to_uint8(image)).resize((size, size), Image.BILINEAR)).astype(np.float32)
    gt, pred = _resize_cls(gt, size), _resize_cls(pred, size)
    rgb = np.repeat(img[..., None], 3, axis=2)
    ok = gt == pred
    rgb[ok & (gt == BACKGROUND)] *= 0.35
    dmg_ok = ok & (gt == DAMAGED)
    rgb[dmg_ok] = 0.6 * rgb[dmg_ok] + 0.4 * np.array((208, 59, 59), np.float32)
    for m, col in (((pred == USABLE) & (gt == DAMAGED), FP_RGB), ((pred == DAMAGED) & (gt == USABLE), FN_RGB),
                   (((pred == BACKGROUND) != (gt == BACKGROUND)), EDGE_RGB)):
        rgb[m] = 0.2 * rgb[m] + 0.8 * np.array(col, np.float32)
    return rgb.clip(0, 255).astype(np.uint8)


def _error_legend():
    return ("<div class='legend'><span><span class='sw' style='background:#8a8a86'></span>correct</span>"
            "<span><span class='sw' style='background:rgba(208,59,59,.45)'></span>damaged, correctly found</span>"
            "<span><span class='sw' style='background:var(--fp)'></span>missed damage (would be sold)</span>"
            "<span><span class='sw' style='background:var(--fn)'></span>good surface marked damaged (wasted)</span>"
            "<span><span class='sw' style='background:#eda100'></span>tile / belt confused</span></div>")


# ── SVG line chart ────────────────────────────────────────────────────────────

def _nice_ticks(lo, hi, n=4):
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = np.floor(lo / step) * step
    ticks = np.arange(start, hi + step * 0.5, step)
    return ticks, step


def _fmt(v, step):
    if step >= 1:
        return f"{v:,.0f}"
    decimals = int(max(0, -np.floor(np.log10(step))))
    return f"{v:.{decimals}f}"


def line_chart(series, width=500, height=210, title="", y_label="", ref=None, x_max=None, y_min=None, y_max=None):
    """
    series: list of (name, values, color). ref: optional (value, label) reference line.
    Returns an SVG string. One y-axis; legend for >= 2 series; end labels; native tooltips.
    """
    pad_l, pad_r, pad_t, pad_b = 46, 70, 10, 22
    vals = [v for _, ys, _ in series for v in ys if v is not None and np.isfinite(v)]
    n = max((len(ys) for _, ys, _ in series), default=1)
    x_max = max(x_max or n, 2)
    lo = min(vals + ([ref[0]] if ref else [])) if vals else 0
    hi = max(vals + ([ref[0]] if ref else [])) if vals else 1
    if y_min is not None: lo = y_min
    if y_max is not None: hi = y_max
    if hi - lo < 1e-9:
        hi = lo + 1
    ticks, step = _nice_ticks(lo, hi)
    lo, hi = ticks[0], ticks[-1]
    W, H = width - pad_l - pad_r, height - pad_t - pad_b
    X = lambda i: pad_l + W * i / (x_max - 1)
    Y = lambda v: pad_t + H * (1 - (v - lo) / (hi - lo))

    parts = [f"<svg viewBox='0 0 {width} {height}' width='100%' style='max-width:{width}px' role='img' aria-label='{_esc(title)}'>"]
    for t in ticks:
        parts.append(f"<line class='gridline' x1='{pad_l}' x2='{pad_l + W}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>"
                     f"<text x='{pad_l - 6}' y='{Y(t) + 4:.1f}' text-anchor='end'>{_fmt(t, step)}</text>")
    parts.append(f"<line class='axis' x1='{pad_l}' x2='{pad_l + W}' y1='{pad_t + H}' y2='{pad_t + H}'/>")
    for e in sorted({1, x_max} | {int(round(x_max * f)) for f in (0.25, 0.5, 0.75)}):
        parts.append(f"<text x='{X(e - 1):.1f}' y='{height - 6}' text-anchor='middle'>{e}</text>")
    if ref:
        parts.append(f"<line x1='{pad_l}' x2='{pad_l + W}' y1='{Y(ref[0]):.1f}' y2='{Y(ref[0]):.1f}' "
                     f"stroke='var(--ink2)' stroke-width='1' opacity='.6'/>"
                     f"<text x='{pad_l + W + 6}' y='{Y(ref[0]) + 4:.1f}' class='ink2'>{_esc(ref[1])}</text>")
    end_labels = []   # (y, text): placed after all lines so they can be spread apart
    for name, ys, color in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(ys) if v is not None and np.isfinite(v)]
        if not pts:
            continue
        d = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}" for k, (x, y) in enumerate(pts))
        parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round' stroke-linecap='round'/>")
        for i, (x, y) in enumerate(pts):
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='7' fill='transparent'><title>{_esc(name)} · epoch {i + 1}: {ys[i]:.4g}</title></circle>")
        x, y = pts[-1]
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}' stroke='var(--surface)' stroke-width='2'/>")
        if not ref or abs(y - Y(ref[0])) > 10:
            end_labels.append([y + 4, x + 8, f"{ys[-1]:.3g}"])
    end_labels.sort()
    for k in range(1, len(end_labels)):            # push labels apart vertically (min 13 px)
        if end_labels[k][0] - end_labels[k - 1][0] < 13:
            end_labels[k][0] = end_labels[k - 1][0] + 13
    for ly, lx, text in end_labels:
        parts.append(f"<text x='{lx:.1f}' y='{ly:.1f}' class='ink'>{text}</text>")
    parts.append("</svg>")
    head = f"<div class='ink2' style='font-size:12px;font-weight:600'>{_esc(title)}"
    head += f" <span class='muted' style='font-weight:400'>by {_esc(y_label)}</span></div>" if y_label else "</div>"
    if len(series) > 1:
        head += "<div class='legend'>" + "".join(
            f"<span><span class='dot' style='background:{c}'></span>{_esc(nm)}</span>" for nm, _, c in series) + "</div>"
    return head + "".join(parts)


# ── Training dashboard ────────────────────────────────────────────────────────

class TrainingDashboard:
    """
    Live training view. Create it, pass it to a training function (or call
    .update() yourself inside a loop), and the output cell redraws every epoch.

        dash = TrainingDashboard("Simple U-Net", epochs=10)
        for ep in range(10):
            ...
            dash.update(ep + 1, 10, train_loss=avg)

    Pure HTML/SVG, no JavaScript — it keeps working in saved notebooks.
    """

    LABELS = {"train_loss": "Train loss", "val_loss": "Val loss", "val_acc": "Val accuracy",
              "val_iou": "Val mean IoU", "val_mae": "Val usable-share error"}

    def __init__(self, title, epochs=None, target_acc=None):
        self.title, self.epochs, self.target = title, epochs, target_acc
        self.hist, self.times, self.t0 = {}, [], time.time()
        self.handle = display(HTML(self._render()), display_id=True)

    def update(self, epoch, epochs=None, secs=None, **metrics):
        self.epochs = epochs or self.epochs
        now = time.time()
        self.times.append(secs if secs is not None else now - self.t0 - sum(self.times))
        for k, v in metrics.items():
            self.hist.setdefault(k, []).append(float(v))
        self.epoch = epoch
        self.handle.update(HTML(self._render()))

    def _render(self):
        ep = getattr(self, "epoch", 0)
        total = self.epochs or max(ep, 1)
        frac = min(ep / total, 1.0) if total else 0
        per = np.mean(self.times[-5:]) if self.times else 0
        eta = per * (total - ep)
        status = "Done" if ep >= total and ep else (f"Epoch {ep}/{total}" if ep else "Starting…")
        eta_txt = f" · {per:.1f}s/epoch · ~{eta / 60:.1f} min left" if ep and ep < total else (
            f" · {sum(self.times) / 60:.1f} min total" if ep else "")

        tiles = []
        for k in ("train_loss", "val_loss", "val_acc", "val_iou", "val_mae"):
            if k not in self.hist:
                continue
            v = self.hist[k][-1]
            unit = "%" if k == "val_acc" else (" pp" if k == "val_mae" else "")
            better = max if k in ("val_acc", "val_iou") else min
            best = better(self.hist[k])
            best_ep = self.hist[k].index(best) + 1
            val = f"{v:.1f}{unit}" if k in ("val_acc", "val_mae") else f"{v:.4f}"
            tiles.append(f"<div class='tile'><div class='lbl'>{self.LABELS[k]}</div><div class='val'>{val}</div>"
                         f"<div class='note'>best {best:.4g}{unit} @ epoch {best_ep}</div></div>")

        charts = []
        loss_series = [(self.LABELS[k], self.hist[k], SERIES[i]) for i, k in enumerate(("train_loss", "val_loss")) if k in self.hist]
        if loss_series:
            charts.append(line_chart(loss_series, title="Loss", x_max=total, y_label="epoch"))
        if "val_acc" in self.hist:
            ref = (self.target, f"target {self.target}%") if self.target else None
            charts.append(line_chart([("Val accuracy", self.hist["val_acc"], SERIES[0])],
                                     title="Validation accuracy (%)", ref=ref, x_max=total, y_label="epoch"))
        elif "val_iou" in self.hist:
            charts.append(line_chart([("Val mean IoU", self.hist["val_iou"], SERIES[0])], title="Validation mean IoU",
                                     x_max=total, y_label="epoch"))
        chart_html = "".join(f"<div style='flex:1;min-width:300px'>{c}</div>" for c in charts)
        body = (f"<h3>{_esc(self.title)}</h3><p class='sub'>{status}{eta_txt}</p>"
                f"<div class='bar' style='margin-bottom:12px'><i style='width:{100 * frac:.1f}%'></i></div>"
                f"<div class='row' style='margin-bottom:10px'>{''.join(tiles)}</div>"
                f"<div class='row'>{chart_html}</div>")
        return _wrap(body)


# ── Scoreboard ────────────────────────────────────────────────────────────────

DEFAULT_TARGETS = {"Threshold": 96.0, "FNN": 78.0, "CNN": 92.0, "Regression": 90.0}   # from a full run of the solution


def _default_store():
    base = "/content/cache" if os.path.isdir("/content") else "cache"
    return os.path.join(base, "attempts.json")


class Scoreboard:
    """
    Accuracy of every method against its target, plus a history of attempts.

        board = Scoreboard()
        board.record(results_basic)        # an EvalResult from evaluate_models
        board.record(results_improved)

    The history is saved to cache/attempts.json so it survives re-running cells.
    """

    def __init__(self, targets=None, iou_target=0.90, mae_target=2.5, store=None):
        self.targets = dict(DEFAULT_TARGETS if targets is None else targets)
        self.iou_target, self.mae_target = iou_target, mae_target
        self.store = store or _default_store()
        self.attempts = []
        if os.path.exists(self.store):
            try:
                with open(self.store) as f:
                    self.attempts = json.load(f)
            except (OSError, ValueError):
                self.attempts = []

    def record(self, res, name=None, show=True):
        entry = {"name": name or res.label, "time": time.strftime("%H:%M"),
                 "acc": {k: round(v, 2) for k, v in res.acc.items()},
                 "iou": round(res.iou, 4), "mae": round(res.frac_mae, 3)}
        self.attempts = [a for a in self.attempts if a["name"] != entry["name"]] + [entry]
        try:
            os.makedirs(os.path.dirname(self.store) or ".", exist_ok=True)
            with open(self.store, "w") as f:
                json.dump(self.attempts, f, indent=1)
        except OSError:
            pass
        if show:
            self.show(entry["name"])

    def reset(self):
        self.attempts = []
        if os.path.exists(self.store):
            os.remove(self.store)

    def show(self, name=None):
        _show(self._render(name))

    def _card(self, label, value_txt, frac, target_txt, ok, gap_txt):
        badge = (f"<span class='badge ok'>✓ target met</span>" if ok
                 else f"<span class='badge no'>✕ {gap_txt} to go</span>")
        return (f"<div class='tile' style='min-width:170px'><div class='lbl'>{_esc(label)}</div>"
                f"<div class='val'>{value_txt}</div>"
                f"<div class='bar' style='margin:6px 0'><i style='width:{100 * max(0, min(frac, 1)):.1f}%'></i></div>"
                f"<div class='note'>target {target_txt}</div><div style='margin-top:4px'>{badge}</div></div>")

    def _render(self, name=None):
        if not self.attempts:
            return _wrap("<h3>Scoreboard</h3><p class='sub'>No attempts recorded yet.</p>")
        cur = next((a for a in self.attempts if a["name"] == name), self.attempts[-1])
        cards = []
        for m, acc in cur["acc"].items():
            t = self.targets.get(m)
            if t is None:
                cards.append(self._card(m, f"{acc:.1f}%", acc / 100, "—", True, ""))
            else:
                cards.append(self._card(m, f"{acc:.1f}%", (acc - 50) / (t - 50) if t > 50 else acc / 100,
                                        f"{t:.0f}%", acc >= t, f"{t - acc:.1f} pp"))
        cards.append(self._card("Mean IoU (3 classes)", f"{cur['iou']:.3f}", cur["iou"] / self.iou_target,
                                f"{self.iou_target:.2f}", cur["iou"] >= self.iou_target, f"{self.iou_target - cur['iou']:.3f}"))
        cards.append(self._card("Usable-share error (MAE)", f"{cur['mae']:.2f} pp",
                                self.mae_target / max(cur["mae"], 1e-6), f"≤ {self.mae_target} pp",
                                cur["mae"] <= self.mae_target, f"{cur['mae'] - self.mae_target:.2f} pp"))
        met = sum(1 for m, a in cur["acc"].items() if m in self.targets and a >= self.targets[m])
        n_t = sum(1 for m in cur["acc"] if m in self.targets)

        methods = []
        for a in self.attempts:
            methods += [m for m in a["acc"] if m not in methods]
        base = self.attempts[0]
        head = "".join(f"<th style='text-align:right'>{_esc(m)}</th>" for m in methods)
        rows = []
        for a in self.attempts:
            cells = []
            for m in methods:
                v = a["acc"].get(m)
                if v is None:
                    cells.append("<td style='text-align:right' class='muted'>—</td>")
                    continue
                d = v - base["acc"].get(m, v) if a is not base else None
                delta = f" <span class='muted'>({d:+.1f})</span>" if d is not None and m in base["acc"] else ""
                cells.append(f"<td style='text-align:right'>{v:.1f}%{delta}</td>")
            style = " style='font-weight:600'" if a is cur else ""
            rows.append(f"<tr{style}><td>{_esc(a['name'])}</td><td class='muted'>{a['time']}</td>{''.join(cells)}"
                        f"<td style='text-align:right'>{a['iou']:.3f}</td><td style='text-align:right'>{a['mae']:.2f}</td></tr>")
        table = (f"<div style='overflow-x:auto;margin-top:14px'><table><tr><th>Attempt</th><th>Time</th>{head}"
                 f"<th style='text-align:right'>IoU</th><th style='text-align:right'>MAE pp</th></tr>{''.join(rows)}</table></div>")
        body = (f"<h3>Scoreboard — {_esc(cur['name'])}</h3>"
                f"<p class='sub'>{met} of {n_t} accuracy targets met on the held-out test set</p>"
                f"<div class='row'>{''.join(cards)}</div>{table}")
        return _wrap(body)


# ── Test-set inspector ────────────────────────────────────────────────────────

def _pick(df, methods, max_items, seed=0):
    """Errors first, then borderline tiles, then random ones."""
    rng = np.random.default_rng(seed)
    wrong = np.zeros(len(df), bool)
    for m in methods:
        wrong |= (df[m] != df["Actual"]).to_numpy()
    border = ((df["true_frac"] - USABLE_THRESHOLD).abs() < 0.05).to_numpy()
    order = [np.flatnonzero(wrong), np.flatnonzero(border & ~wrong), np.flatnonzero(~border & ~wrong)]
    chosen = []
    for group, quota in zip(order, (max_items // 2, max_items // 4, max_items)):
        take = rng.permutation(group)[:max(0, min(quota, max_items - len(chosen)))]
        chosen += take.tolist()
    return sorted(chosen, key=lambda r: df["true_frac"].iat[r])


def test_inspector(res, max_items=80, size=128, title=None):
    """
    Step through test tiles: photo | true class map | predicted class map | error map,
    with every model's verdict. Filter by errors or borderline tiles.
    """
    df, cache, methods = res.df, res.cache, res.methods
    rows = _pick(df, methods, max_items)
    items = []
    for r in rows:
        idx = int(df["index"].iat[r])
        pred = res.pred_masks[r]
        gt = cache.target(idx)
        verdicts = {}
        for m in methods:
            p = {"FNN": df["fnn_prob"].iat[r], "CNN": df["cnn_prob"].iat[r]}.get(m)
            extra = f"p={p:.2f}" if p is not None else (
                f"{100 * df['reg_frac'].iat[r]:.1f}% usable" if m == "Regression" else f"{100 * df['pred_frac'].iat[r]:.1f}% usable")
            verdicts[m] = [df[m].iat[r], extra]
        items.append({
            "f": df["filename"].iat[r], "t": round(100 * df["true_frac"].iat[r], 1),
            "p": round(100 * df["pred_frac"].iat[r], 1), "iou": round(df["iou_damaged"].iat[r], 3),
            "a": df["Actual"].iat[r], "v": verdicts,
            "wrong": any(v[0] != df["Actual"].iat[r] for v in verdicts.values()),
            "border": bool(abs(df["true_frac"].iat[r] - USABLE_THRESHOLD) < 0.05),
            "img": encode_thumb(cache.image(idx), size), "gt": encode_thumb(class_rgb(gt, size), size, fmt="PNG"),
            "pr": encode_thumb(class_rgb(pred, size), size, fmt="PNG"),
            "err": encode_thumb(error_overlay(cache.image(idx), gt, pred, size), size),
        })
    uid = _uid()
    n_wrong = sum(i["wrong"] for i in items)
    first = items[0] if items else None
    panels = "".join(
        f"<div class='panel' style='flex:1;min-width:140px'><img data-k='{k}' src='{first[k] if first else ''}' alt='{lbl}'>"
        f"<div class='cap'>{lbl}</div></div>"
        for k, lbl in (("img", "Photo"), ("gt", "Truth (white usable · red damaged)"),
                       ("pr", "U-Net prediction"), ("err", "Error map")))
    body = f"""
<h3>{_esc(title or 'Test-set inspector — ' + res.label)}</h3>
<p class='sub'>{len(items)} of {len(df)} test tiles ({n_wrong} with at least one wrong verdict), sorted by true usable share.
Use ← → or click the strip.</p>
<div class='row' style='align-items:center;margin-bottom:10px'>
  <span class='ink2' style='font-size:13px'>Show</span>
  <button data-f='all' class='on'>All</button><button data-f='wrong'>Wrong verdicts</button>
  <button data-f='border'>Borderline ({100 * USABLE_THRESHOLD - 5:.0f}–{100 * USABLE_THRESHOLD + 5:.0f}%)</button>
  <input type='range' min='0' max='{max(len(items) - 1, 0)}' value='0' style='flex:1;min-width:160px' aria-label='image index'>
  <button data-nav='-1' aria-label='previous'>←</button><button data-nav='1' aria-label='next'>→</button>
</div>
<div class='row' style='align-items:flex-start'>
  <div class='row' style='flex:3;min-width:320px'>{panels}</div>
  <div style='flex:1;min-width:220px' class='info'></div>
</div>
{_error_legend()}
<div class='strip'></div>
"""
    script = """
(function(){const root=document.getElementById('%s');const all=__DATA__;
let view=all.slice(),pos=0;const rng=root.querySelector('input[type=range]'),info=root.querySelector('.info'),strip=root.querySelector('.strip');
function badge(v,a){const ok=v===a;return `<span class="badge ${ok?'ok':'no'}">${ok?'✓':'✕'} ${v}</span>`}
function draw(){if(!view.length){info.innerHTML='<p class="muted">Nothing matches this filter.</p>';return}
const it=view[pos];root.querySelectorAll('img[data-k]').forEach(im=>im.src=it[im.dataset.k]);
let rows='';for(const [m,[v,x]] of Object.entries(it.v)){rows+=`<tr><td>${m}</td><td class="muted" style="text-align:right">${x}</td><td>${badge(v,it.a)}</td></tr>`}
info.innerHTML=`<div style="font-weight:600;margin-bottom:4px">${it.f}</div>
<div class="ink2" style="font-size:13px">True usable share <b>${it.t}%%</b> → <b>${it.a}</b><br>U-Net usable share ${it.p}%% · damaged-zone IoU ${it.iou}</div>
<table style="margin-top:8px"><tr><th>Method</th><th></th><th>Verdict</th></tr>${rows}</table>`;
rng.max=Math.max(view.length-1,0);rng.value=pos;strip.querySelectorAll('img').forEach((im,i)=>im.classList.toggle('sel',i===pos));
const s=strip.querySelectorAll('img')[pos];if(s)s.scrollIntoView({block:'nearest',inline:'nearest'})}
function build(){strip.innerHTML='';view.forEach((it,i)=>{const im=document.createElement('img');im.src=it.err;im.title=it.f+' · '+it.t+'%%';
im.onclick=()=>{pos=i;draw()};strip.appendChild(im)});pos=0;draw()}
root.querySelectorAll('button[data-f]').forEach(b=>b.onclick=()=>{root.querySelectorAll('button[data-f]').forEach(x=>x.classList.remove('on'));
b.classList.add('on');const f=b.dataset.f;view=all.filter(it=>f==='all'||it[f]);build()});
root.querySelectorAll('button[data-nav]').forEach(b=>b.onclick=()=>{pos=Math.min(Math.max(pos+ +b.dataset.nav,0),view.length-1);draw()});
rng.oninput=()=>{pos=+rng.value;draw()};root.tabIndex=0;root.onkeydown=e=>{if(e.key==='ArrowRight'){pos=Math.min(pos+1,view.length-1);draw()}
if(e.key==='ArrowLeft'){pos=Math.max(pos-1,0);draw()}};build()})();""" % uid
    _show(_wrap(body, uid, script.replace('__DATA__', _js_data(items))))


def dataset_preview(cache, indices, n=8, size=128, title="Sample tiles from the test set"):
    """Static grid: photo + class map, with usable share and verdict."""
    figs = []
    for idx in list(indices)[:n]:
        frac = 100 * cache.usable[idx]
        verdict = "APPROVE" if cache.labels[idx] else "REJECT"
        cls = "ok" if cache.labels[idx] else "no"
        figs.append(f"<figure><img src='{encode_thumb(cache.image(idx), size)}' alt='{cache.files[idx]}'>"
                    f"<img src='{encode_thumb(class_rgb(cache.target(idx), size), size, fmt='PNG')}' style='margin-top:4px' alt='class map'>"
                    f"<figcaption>{_esc(cache.files[idx])}<br>{frac:.1f}% usable "
                    f"<span class='badge {cls}'>{'✓' if cache.labels[idx] else '✕'} {verdict}</span></figcaption></figure>")
    approve = int(cache.labels.sum())
    body = (f"<h3>{_esc(title)}</h3><p class='sub'>{len(cache):,} tiles · {approve:,} APPROVE "
            f"({100 * approve / len(cache):.1f}%) · {len(cache) - approve:,} REJECT. Top: photo from the inspection camera, "
            f"bottom: class map (grey = belt, white = usable surface, red = damaged zone to cut away). "
            f"A tile is approved when at least {100 * USABLE_THRESHOLD:.0f}% of its surface is usable.</p>"
            f"<div class='grid' style='grid-template-columns:repeat(auto-fill,minmax(110px,1fr))'>{''.join(figs)}</div>")
    _show(_wrap(body))


def pipeline_card(image, gt_map, pred_map, verdicts, true_usable, pred_usable, title="Full pipeline on one tile", size=200):
    """
    One tile through the whole pipeline: photo -> truth -> prediction -> errors,
    plus a verdict per method.  verdicts: {"Threshold": ("APPROVE", "91.3% usable"), ...}
    """
    image, gt_map, pred_map = (np.asarray(a).squeeze() for a in (image, gt_map, pred_map))
    actual = "APPROVE" if true_usable >= USABLE_THRESHOLD else "REJECT"
    panels = [("Photo (model input)", encode_thumb(image, size)),
              (f"Truth · {100 * true_usable:.1f}% usable", encode_thumb(class_rgb(gt_map, size), size, fmt="PNG")),
              (f"U-Net · {100 * pred_usable:.1f}% usable", encode_thumb(class_rgb(pred_map, size), size, fmt="PNG")),
              ("Error map", encode_thumb(error_overlay(image, gt_map, pred_map, size), size))]
    figs = "".join(f"<div class='panel' style='flex:1;min-width:130px'><img src='{src}' alt='{_esc(c)}'><div class='cap'>{_esc(c)}</div></div>"
                   for c, src in panels)
    rows = "".join(
        f"<tr><td>{_esc(m)}</td><td class='muted' style='text-align:right'>{_esc(d)}</td>"
        f"<td><span class='badge {'ok' if v == actual else 'no'}'>{'✓' if v == actual else '✕'} {v}</span></td></tr>"
        for m, (v, d) in verdicts.items())
    body = (f"<h3>{_esc(title)}</h3><p class='sub'>Correct decision: <b>{actual}</b> "
            f"({100 * true_usable:.1f}% of the tile is usable vs. the {100 * USABLE_THRESHOLD:.0f}% rule)</p>"
            f"<div class='row' style='align-items:flex-start'><div class='row' style='flex:3;min-width:320px'>{figs}</div>"
            f"<div style='flex:1;min-width:220px'><table><tr><th>Method</th><th>Evidence</th><th>Verdict</th></tr>{rows}</table></div></div>"
            f"{_error_legend()}")
    _show(_wrap(body))


# ── Confusion explorer ────────────────────────────────────────────────────────

def confusion_explorer(res, methods=None, per_cell=18, size=112):
    """Clickable confusion matrix per method; click a cell to see those tiles."""
    df, cache = res.df, res.cache
    methods = methods or res.methods
    thumbs, data = {}, {}
    rng = np.random.default_rng(0)
    for m in methods:
        cells = {}
        for actual in ("APPROVE", "REJECT"):
            for pred in ("APPROVE", "REJECT"):
                rows = np.flatnonzero(((df["Actual"] == actual) & (df[m] == pred)).to_numpy())
                cap = per_cell * 2 if actual != pred else per_cell // 2
                shown = rng.permutation(rows)[:cap]
                shown = sorted(shown, key=lambda r: -abs(df["true_frac"].iat[r] - df["pred_frac"].iat[r]))
                for r in shown:
                    if r not in thumbs:
                        idx = int(df["index"].iat[r])
                        thumbs[int(r)] = {"src": encode_thumb(error_overlay(cache.image(idx), cache.target(idx), res.pred_masks[r], size), size),
                                          "f": df["filename"].iat[r], "t": round(100 * df["true_frac"].iat[r], 1),
                                          "p": round(100 * df["pred_frac"].iat[r], 1)}
                cells[f"{actual}|{pred}"] = {"n": int(len(rows)), "show": [int(r) for r in shown]}
        data[m] = {"cells": cells, "acc": round(res.acc[m], 1)}
    uid = _uid()
    tabs = "".join(f"<button data-m='{_esc(m)}' class='{'on' if i == 0 else ''}'>{_esc(m)}</button>" for i, m in enumerate(methods))
    body = f"""
<h3>Where do the models go wrong? — {_esc(res.label)}</h3>
<p class='sub'>Rows are the truth, columns the model's verdict. Click a cell to see those tiles.
<b>False approvals</b> ship damaged tiles to customers; <b>false rejections</b> send sellable tiles to the recycling bin.</p>
<div class='row' style='margin-bottom:10px'>{tabs}</div>
<div class='row' style='align-items:flex-start'>
 <div class='cm' style='min-width:300px'></div>
 <div style='flex:1;min-width:300px'><div class='gtitle ink2' style='font-size:13px;margin-bottom:6px'></div><div class='grid'></div></div>
</div>{_error_legend()}
"""
    script = """
(function(){const root=document.getElementById('%s');const D=__DATA__;
let m=Object.keys(D.m)[0],sel='REJECT|APPROVE';const cm=root.querySelector('.cm'),grid=root.querySelector('.grid'),gt=root.querySelector('.gtitle');
const NAME={'APPROVE|APPROVE':'Correct approvals','REJECT|REJECT':'Correct rejections','REJECT|APPROVE':'False approvals (damaged tile sold)','APPROVE|REJECT':'False rejections (good tile recycled)'};
function draw(){const c=D.m[m].cells;const max=Math.max(...Object.values(c).map(x=>x.n),1);
let h=`<div class="ink2" style="font-size:13px;margin-bottom:6px">${m}: <b>${D.m[m].acc}%%</b> accuracy</div><table style="font-size:14px"><tr><th></th><th style="text-align:center">→ APPROVE</th><th style="text-align:center">→ REJECT</th></tr>`;
for(const a of ['APPROVE','REJECT']){h+=`<tr><th>${a==='APPROVE'?'Truly sellable':'Truly too damaged'}</th>`;for(const p of ['APPROVE','REJECT']){const k=a+'|'+p,x=c[k],err=a!==p;
const alpha=(0.08+0.55*x.n/max).toFixed(2);const bg=err?`rgba(227,73,72,${alpha})`:`rgba(42,120,214,${alpha})`;
h+=`<td data-k="${k}" style="cursor:pointer;text-align:center;padding:16px 22px;background:${bg};border:2px solid ${k===sel?'var(--ink)':'var(--surface)'};border-radius:8px"><div style="font-size:22px;font-weight:650">${x.n}</div><div style="font-size:11px;color:var(--ink2)">${err?'error':'correct'}</div></td>`}h+='</tr>'}
cm.innerHTML=h+'</table>';cm.querySelectorAll('td[data-k]').forEach(td=>td.onclick=()=>{sel=td.dataset.k;draw()});
const x=c[sel];gt.innerHTML=`<b>${NAME[sel]}</b> — ${x.n} tiles${x.n>x.show.length?`, showing ${x.show.length} (largest usable-share error first)`:''}`;
grid.innerHTML=x.show.map(r=>{const t=D.t[r];return `<figure><img src="${t.src}" alt="${t.f}"><figcaption>${t.f}<br>true ${t.t}%% · pred ${t.p}%% usable</figcaption></figure>`}).join('')||'<p class="muted">No tiles in this cell 🎉</p>'}
root.querySelectorAll('button[data-m]').forEach(b=>b.onclick=()=>{root.querySelectorAll('button[data-m]').forEach(x=>x.classList.remove('on'));b.classList.add('on');m=b.dataset.m;draw()});draw()})();""" % uid
    _show(_wrap(body, uid, script.replace('__DATA__', _js_data({"m": data, "t": thumbs}))))


# ── Threshold & business-cost explorer ───────────────────────────────────────

def threshold_explorer(res, cost_false_approve=5.0, cost_false_reject=1.0, source="U-Net"):
    """
    Scatter of true vs predicted usable share with a draggable decision threshold.
    Shows accuracy, false approvals/rejections and total business cost live.
    source: "U-Net" (usable share of the predicted mask) or "Regression" (regression head).
    """
    df = res.df
    sources = {"U-Net": df["pred_frac"].to_numpy()}
    if "Regression" in res.methods:
        sources["Regression"] = df["reg_frac"].to_numpy()
    data = {"t": np.round(df["true_frac"].to_numpy(), 4).tolist(), "thr": USABLE_THRESHOLD,
            "s": {k: np.round(v, 4).tolist() for k, v in sources.items()},
            "f": df["filename"].tolist(), "cfa": cost_false_approve, "cfr": cost_false_reject,
            "src": source if source in sources else "U-Net"}
    uid = _uid()
    src_btns = "".join(f"<button data-s='{k}' class='{'on' if k == data['src'] else ''}'>{k} usable share</button>" for k in sources)
    body = f"""
<h3>Decision threshold & business cost — {_esc(res.label)}</h3>
<p class='sub'>Each dot is a test tile. The business rule is fixed at {100 * USABLE_THRESHOLD:.0f}% usable (vertical line);
<b>you</b> choose where to cut the <i>predicted</i> usable share (horizontal line — drag it or use the slider).
If selling a damaged tile costs more than recycling a good one, should the cut stay at {100 * USABLE_THRESHOLD:.0f}%?</p>
<div class='row' style='margin-bottom:8px'>{src_btns}</div>
<div class='row' style='align-items:flex-start'>
 <div style='flex:1.3;min-width:320px'><svg class='sc' viewBox='0 0 420 400' width='100%' style='max-width:460px;touch-action:none'></svg></div>
 <div style='flex:1;min-width:260px'>
  <label class='ink2' style='font-size:13px'>Decision threshold on the predicted usable share: <b class='thv'></b></label>
  <input class='th' type='range' min='{100 * USABLE_THRESHOLD - 25:.0f}' max='{min(100, 100 * USABLE_THRESHOLD + 12):.0f}' step='0.5' value='{100 * USABLE_THRESHOLD:.0f}' style='width:100%'>
  <label class='ink2' style='font-size:13px'>Cost of a false approval (damaged tile sold, returned): <b class='fav'></b></label>
  <input class='fa' type='range' min='0' max='20' step='0.5' value='{cost_false_approve}' style='width:100%'>
  <label class='ink2' style='font-size:13px'>Cost of a false rejection (sellable tile recycled): <b class='frv'></b></label>
  <input class='fr' type='range' min='0' max='20' step='0.5' value='{cost_false_reject}' style='width:100%'>
  <div class='row stats' style='margin:10px 0'></div>
  <button class='best'>Jump to the cheapest threshold</button>
  <div class='cost' style='margin-top:10px'></div>
 </div>
</div>
<div class='legend'><span><span class='dot' style='background:#2a78d6'></span>correct</span>
<span><span class='dot' style='background:#e34948'></span>false approval</span><span><span class='dot' style='background:#eda100'></span>false rejection</span></div>
"""
    script = """
(function(){const root=document.getElementById('%s');const D=__DATA__;
const svg=root.querySelector('.sc'),th=root.querySelector('.th'),fa=root.querySelector('.fa'),fr=root.querySelector('.fr');
const L=46,T=12,W=360,H=340,NS='http://www.w3.org/2000/svg';let src=D.src;
const X=v=>L+W*v,Y=v=>T+H*(1-v);
function el(n,a,p){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);(p||svg).appendChild(e);return e}
const LO=+th.min,HI=+th.max;
function stats(t){const s=D.s[src];let fa_=0,fr_=0;for(let i=0;i<s.length;i++){const good=D.t[i]>=D.thr,ok=s[i]>=t;if(ok&&!good)fa_++;if(!ok&&good)fr_++}
return {fa:fa_,fr:fr_,acc:100*(1-(fa_+fr_)/s.length),cost:fa_*+fa.value+fr_*+fr.value}}
function axes(){svg.innerHTML='';for(let v=0;v<=1.0001;v+=0.2){el('line',{x1:L,x2:L+W,y1:Y(v),y2:Y(v),class:'gridline'});el('line',{y1:T,y2:T+H,x1:X(v),x2:X(v),class:'gridline'});
el('text',{x:L-6,y:Y(v)+4,'text-anchor':'end'}).textContent=Math.round(v*100)+'%%';el('text',{x:X(v),y:T+H+16,'text-anchor':'middle'}).textContent=Math.round(v*100)+'%%'}
el('text',{x:L+W,y:T+H+32,'text-anchor':'end'}).textContent='true usable share →';
const yl=el('text',{x:L+6,y:T+H-6,'text-anchor':'start'});yl.textContent='↑ predicted usable share';
el('line',{x1:X(D.thr),x2:X(D.thr),y1:T,y2:T+H,stroke:'var(--ink2)','stroke-width':1});
el('text',{x:X(D.thr)-4,y:T+H-6,class:'ink2','text-anchor':'end'}).textContent='rule: '+Math.round(D.thr*100)+'%%'}
let dots=[],line,lbl;
function build(){axes();const s=D.s[src];dots=s.map((v,i)=>{const c=el('circle',{cx:X(D.t[i]),cy:Y(Math.min(Math.max(v,0),1)),r:3.2,stroke:'var(--surface)','stroke-width':1});
const tt=el('title',{},c);tt.textContent=`${D.f[i]} · true ${(100*D.t[i]).toFixed(1)}%% · predicted ${(100*v).toFixed(1)}%%`;return c});
line=el('line',{x1:L,x2:L+W,stroke:'var(--ink)','stroke-width':2,style:'cursor:ns-resize'});
lbl=el('text',{x:L+6,class:'ink','style':'font-size:12px;font-weight:600'});
const hit=el('rect',{x:L,width:W,height:14,fill:'transparent',style:'cursor:ns-resize'});line.hit=hit;draw()}
function draw(){const t=th.value/100,s=D.s[src];root.querySelector('.thv').textContent=(+th.value).toFixed(1)+'%%';
root.querySelector('.fav').textContent=fa.value;root.querySelector('.frv').textContent=fr.value;
dots.forEach((c,i)=>{const good=D.t[i]>=D.thr,ok=s[i]>=t;c.setAttribute('fill',ok&&!good?'#e34948':(!ok&&good?'#eda100':'#2a78d6'));c.setAttribute('opacity',ok===good?0.45:0.95)});
line.setAttribute('y1',Y(t));line.setAttribute('y2',Y(t));line.hit.setAttribute('y',Y(t)-7);lbl.setAttribute('y',Y(t)-6);lbl.textContent='cut: '+(+th.value).toFixed(1)+'%%';
const r=stats(t);root.querySelector('.stats').innerHTML=[['Accuracy',r.acc.toFixed(1)+'%%'],['False approvals',r.fa],['False rejections',r.fr],['Total cost',r.cost.toFixed(1)]]
.map(([k,v])=>`<div class="tile" style="min-width:90px"><div class="lbl">${k}</div><div class="val" style="font-size:18px">${v}</div></div>`).join('');curve(t)}
function curve(t){let pts=[],best=null;for(let v=LO;v<=HI;v+=0.5){const r=stats(v/100);pts.push([v,r.cost]);if(!best||r.cost<best[1])best=[v,r.cost]}
const max=Math.max(...pts.map(p=>p[1]),1),w=300,h=90,x=v=>8+(w-16)*(v-LO)/(HI-LO),y=c=>h-14-(h-24)*c/max;
const path=pts.map((p,i)=>(i?'L':'M')+x(p[0]).toFixed(1)+','+y(p[1]).toFixed(1)).join(' ');
root.querySelector('.cost').innerHTML=`<div class="ink2" style="font-size:12px;font-weight:600">Total cost vs threshold — cheapest at ${best[0].toFixed(1)}%%</div>
<svg viewBox="0 0 ${w} ${h}" width="100%%" style="max-width:${w}px"><line class="axis" x1="8" x2="${w-8}" y1="${h-14}" y2="${h-14}"/>
<path d="${path}" fill="none" stroke="#2a78d6" stroke-width="2"/><circle cx="${x(th.value)}" cy="${y(stats(t).cost)}" r="4" fill="#2a78d6" stroke="var(--surface)" stroke-width="2"/>
<text x="8" y="${h-2}">${LO}%%</text><text x="${x(D.thr*100)}" y="${h-2}" text-anchor="middle">${Math.round(D.thr*100)}%%</text><text x="${w-8}" y="${h-2}" text-anchor="end">${HI}%%</text></svg>`;root._best=best[0]}
let drag=false;svg.addEventListener('pointerdown',e=>{drag=true;move(e)});window.addEventListener('pointerup',()=>drag=false);
svg.addEventListener('pointermove',e=>{if(drag)move(e)});
function move(e){const r=svg.getBoundingClientRect(),yy=(e.clientY-r.top)*400/r.height;const v=Math.min(Math.max((1-(yy-T)/H)*100,LO),HI);th.value=Math.round(v*2)/2;draw()}
[th,fa,fr].forEach(i=>i.oninput=draw);root.querySelector('.best').onclick=()=>{th.value=root._best;draw()};
root.querySelectorAll('button[data-s]').forEach(b=>b.onclick=()=>{root.querySelectorAll('button[data-s]').forEach(x=>x.classList.remove('on'));b.classList.add('on');src=b.dataset.s;build()});
build()})();""" % uid
    _show(_wrap(body, uid, script.replace('__DATA__', _js_data(data))))


# ── Receptive-field visualiser ───────────────────────────────────────────────

def rf_visualizer(cache, idx, rfs, tile_sizes=None, scale=2, sizes_label="Damaged-zone sizes"):
    """
    Draw each model's receptive field as a square on a real tile photo (click the
    image to move it) next to a histogram of object sizes (e.g. damaged zones).
    Below the photo, bars show what each square contains at the clicked spot
    (belt / usable surface / damaged zone), counted from the tile's true class map.
    rfs: {"model name": side_px_at_model_resolution}
    """
    size = cache.size
    uid = _uid()
    img = encode_thumb(cache.image(idx), size * scale)
    names = list(rfs)
    colors = {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}
    legend = "".join(f"<span><span class='sw' style='background:{colors[n]}'></span>{_esc(n)}: {rfs[n]} px</span>" for n in names)
    hist = ""
    if tile_sizes is not None and len(tile_sizes):
        bins = np.arange(0, max(130, int(tile_sizes.max()) + 10), 8)
        counts, edges = np.histogram(tile_sizes, bins=bins)
        w, h, pl, pb = 420, 220, 40, 30
        xmax = max(edges[-1], max(rfs.values()) + 10)
        X = lambda v: pl + (w - pl - 10) * v / xmax
        Y = lambda c: h - pb - (h - pb - 16) * c / max(counts.max(), 1)
        bars = "".join(f"<rect x='{X(edges[i]) + 1:.1f}' y='{Y(c):.1f}' width='{max(X(edges[i + 1]) - X(edges[i]) - 2, 1):.1f}' "
                       f"height='{h - pb - Y(c):.1f}' rx='2' fill='var(--axis)'><title>{edges[i]:.0f}–{edges[i + 1]:.0f} px: {c}</title></rect>"
                       for i, c in enumerate(counts))
        lines = "".join(f"<line x1='{X(v):.1f}' x2='{X(v):.1f}' y1='12' y2='{h - pb}' stroke='{colors[n]}' stroke-width='2'>"
                        f"<title>{_esc(n)}: {v} px</title></line>" for n, v in rfs.items())
        ticks = "".join(f"<text x='{X(v):.1f}' y='{h - pb + 14}' text-anchor='middle'>{v}</text>" for v in range(0, int(xmax) + 1, 25))
        pct = {n: 100 * (tile_sizes <= v).mean() for n, v in rfs.items()}
        cover = " · ".join(f"{_esc(n)}: {pct[n]:.0f}%" for n in names)
        cover = "Zones smaller than each field — " + cover + ". Most zones are larger than every field: a unit never sees a whole zone, only the context around its pixel"
        hist = (f"<div class='ink2' style='font-size:12px;font-weight:600'>{_esc(sizes_label)} (px at {size}×{size}) "
                f"<span class='muted' style='font-weight:400'>· whole dataset, stays fixed</span></div>"
                f"<svg viewBox='0 0 {w} {h}' width='100%' style='max-width:{w}px'><line class='axis' x1='{pl}' x2='{w - 10}' y1='{h - pb}' y2='{h - pb}'/>"
                f"{bars}{lines}{ticks}<text x='{w - 10}' y='{h - 2}' text-anchor='end'>side length (px)</text></svg>"
                f"<p class='sub' style='margin-top:4px'>{cover}.</p>")
    body = f"""
<h3>How much of the tile can one neuron see?</h3>
<p class='sub'>Squares show the receptive field of the <b>bottleneck</b> (the input area that can influence one bottleneck unit).
To tell a crack edge from a stain, a marble vein or a scratch, a unit needs enough surface around its pixel.
<b>Click the photo</b> to move the squares — the bars below show what each unit would see there.</p>
<div class='legend'>{legend}</div>
<div class='row' style='align-items:flex-start'>
 <div style='position:relative;flex:0 1 {size * scale}px;min-width:260px'>
  <img src='{img}' style='width:100%;border-radius:8px;display:block;cursor:crosshair' alt='{_esc(cache.files[idx])}'>
  <svg class='ov' viewBox='0 0 {size} {size}' style='position:absolute;inset:0;width:100%;height:100%;pointer-events:none'></svg>
  <div class='rfinfo' style='margin-top:10px'></div>
 </div>
 <div style='flex:1;min-width:280px'>{hist}</div>
</div>
"""
    script = """
(function(){const root=document.getElementById('%s');const D=__DATA__;
const ov=root.querySelector('.ov'),img=root.querySelector('img'),info=root.querySelector('.rfinfo');let cx=D.x,cy=D.y;
const NAMES=['belt','usable','damaged'],COLS=['#3b3b38','#e8e8e4','#d03b3b'];
function contents(s){const n=D.n,x0=Math.max(0,Math.floor(cx-s/2)),x1=Math.min(n,Math.ceil(cx+s/2)),y0=Math.max(0,Math.floor(cy-s/2)),y1=Math.min(n,Math.ceil(cy+s/2));
const c=[0,0,0];for(let y=y0;y<y1;y++){const row=y*n;for(let x=x0;x<x1;x++)c[D.cm.charCodeAt(row+x)-48]++}const t=c[0]+c[1]+c[2]||1;return c.map(v=>100*v/t)}
function table(){const here=NAMES[D.cm.charCodeAt(Math.min(D.n-1,Math.floor(cy))*D.n+Math.min(D.n-1,Math.floor(cx)))-48];
let h=`<div class="ink2" style="font-size:12px;font-weight:600;margin-bottom:6px">What each unit sees here <span class="muted" style="font-weight:400">· the white dot is on <b>${here==='damaged'?'the damaged zone':here==='usable'?'usable surface':'the belt'}</b></span></div>`;
for(const [n,s] of Object.entries(D.rf)){const p=contents(s);
h+=`<div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px"><span class="dot" style="background:${D.c[n]}"></span><span style="min-width:122px;white-space:nowrap">${n}</span>`+
`<div style="flex:1;display:flex;height:12px;border-radius:3px;overflow:hidden;min-width:70px">${p.map((v,i)=>`<i style="width:${v}%%;background:${COLS[i]}" title="${NAMES[i]} ${v.toFixed(0)}%%"></i>`).join('')}</div>`+
`<span class="muted" style="white-space:nowrap;font-size:11px;font-variant-numeric:tabular-nums">${p[2].toFixed(0)}%% damaged · ${p[1].toFixed(0)}%% usable · ${p[0].toFixed(0)}%% belt</span></div>`}
info.innerHTML=h+`<div class="legend" style="margin-top:6px">${NAMES.map((m,i)=>`<span><span class="sw" style="background:${COLS[i]}"></span>${m}</span>`).join('')}</div>`}
function draw(){table();ov.innerHTML=Object.entries(D.rf).map(([n,s])=>`<rect x="${cx-s/2}" y="${cy-s/2}" width="${s}" height="${s}" fill="none" stroke="#fff" stroke-width="2.4"/><rect x="${cx-s/2}" y="${cy-s/2}" width="${s}" height="${s}" fill="none" stroke="${D.c[n]}" stroke-width="1.4"/>`).join('')+`<circle cx="${cx}" cy="${cy}" r="2.5" fill="#fff" stroke="#000" stroke-width=".8"/>`}
img.onclick=e=>{const r=img.getBoundingClientRect();cx=(e.clientX-r.left)*D.n/r.width;cy=(e.clientY-r.top)*D.n/r.height;draw()};draw()})();""" % uid
    target = cache.target(idx)
    dmg = np.argwhere(target == DAMAGED)          # start on the damaged zone nearest the centre
    y0, x0 = (dmg[np.argmin(((dmg - size / 2) ** 2).sum(1))] if len(dmg) else (size // 2, size // 2))
    data = {"rf": rfs, "c": colors, "n": size, "x": int(x0) + 0.5, "y": int(y0) + 0.5,
            "cm": "".join(map(str, target.astype(np.uint8).ravel().tolist()))}
    _show(_wrap(body, uid, script.replace('__DATA__', _js_data(data))))


# ── Side-by-side gallery for several models ──────────────────────────────────

def compare_gallery(results, n=6, size=128, title="Same tiles, different models", names=None):
    """
    Error maps of several U-Nets on the tiles where the FIRST one misses the
    most damage (predicts usable where the tile is damaged) — its hardest cases.
    Only the segmentation (and the threshold verdict computed from it) is shown;
    the FNN/CNN/regression heads play no part. `names` labels the rows (default:
    each result's label).
    """
    names = list(names) if names is not None else [res.label for res in results]
    base = results[0]
    cache = base.cache
    gts = np.stack([_resize_cls(cache.target(int(i)), base.pred_masks.shape[1]) for i in base.df["index"]])
    missed = ((base.pred_masks == USABLE) & (gts == DAMAGED)).mean(axis=(1, 2))
    rows = np.argsort(-missed)[:n]
    head = "<tr><th></th>" + "".join(f"<th style='text-align:center'>{_esc(cache.files[int(base.df['index'].iat[r])])}"
                                    f"<br><span class='muted'>true {100 * base.df['true_frac'].iat[r]:.1f}% usable</span></th>" for r in rows) + "</tr>"
    body_rows = []
    for res, name in zip(results, names):
        cells = []
        for r in rows:
            idx = int(res.df["index"].iat[r])
            src = encode_thumb(error_overlay(cache.image(idx), cache.target(idx), res.pred_masks[r], size), size)
            ok = res.df["Threshold"].iat[r] == res.df["Actual"].iat[r]
            cells.append(f"<td style='text-align:center'><img src='{src}' style='width:{size}px;max-width:100%;border-radius:6px;display:block;margin:auto'>"
                         f"<span class='muted' style='font-size:11px'>pred {100 * res.df['pred_frac'].iat[r]:.1f}%</span> "
                         f"<span class='badge {'ok' if ok else 'no'}'>{'✓' if ok else '✕'}</span></td>")
        body_rows.append(f"<tr><th>{_esc(name)}<br><span class='muted' style='font-weight:400'>mIoU {res.iou:.3f} · "
                         f"threshold acc. {res.acc['Threshold']:.1f}%</span></th>{''.join(cells)}</tr>")
    body = (f"<h3>{_esc(title)}</h3><p class='sub'>The {n} test tiles where <b>{_esc(names[0])}</b> misses the most damage. "
            f"Each row is one U-Net's segmentation of the same tile, coloured against the ground truth "
            f"(red = damaged surface predicted usable). <b>pred</b> = usable share measured on that mask; "
            f"✓/✕ = whether the threshold rule (APPROVE if ≥ {100 * USABLE_THRESHOLD:.0f}%) gives the true verdict. "
            f"The FNN, CNN and regression heads are not part of this view.</p><div style='overflow-x:auto'><table>{head}{''.join(body_rows)}</table></div>{_error_legend()}")
    _show(_wrap(body))


# ── Architecture diagram ─────────────────────────────────────────────────────

def architecture_diagram(model, input_size=256, title=None):
    """Blocks for every top-level module in execution order, with output shapes."""
    import torch
    import torch.nn as nn
    records, hooks = [], []
    targets = []
    for name, mod in model.named_children():
        if isinstance(mod, (nn.ModuleList, nn.ModuleDict)):
            targets += [(f"{name}.{k}", sub) for k, sub in mod.named_children()]
        else:
            targets.append((name, mod))
    for name, mod in targets:
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, name=name: records.append((name, type(m).__name__, tuple(o.shape) if hasattr(o, "shape") else None))))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    x = torch.zeros(1, 1, input_size, input_size, device=device)
    try:
        with torch.no_grad():
            out = model(x)
    finally:
        for h in hooks:
            h.remove()
    with torch.no_grad():
        b, skips = model.encode(x)
    model.train(was_training)
    blocks = records
    max_c = max((s[1] for _, _, s in blocks if s and len(s) == 4), default=1)
    items = []
    for name, kind, shape in blocks:
        if not shape or len(shape) != 4:
            continue
        c, h = shape[1], shape[2]
        height = 40 + 110 * (h / input_size)
        width = 16 + 40 * (np.log2(c + 1) / np.log2(max_c + 1))
        is_b = shape[1:] == tuple(b.shape[1:]) and "bottleneck" in name.lower()
        color = "var(--accent)" if is_b else ("var(--axis)" if "down" in name or "pool" in name or "up" in name else "var(--ink2)")
        items.append(f"<div style='display:flex;flex-direction:column;align-items:center;gap:3px;min-width:{max(width + 16, 74):.0f}px'>"
                     f"<div title='{_esc(kind)}' style='width:{width:.0f}px;height:{height:.0f}px;border-radius:5px;background:{color};opacity:{1 if is_b else .55}'></div>"
                     f"<div style='font-size:11px;font-weight:600;white-space:nowrap'>{_esc(name)}</div>"
                     f"<div class='muted' style='font-size:10px;white-space:nowrap'>{c} × {h}²</div></div>")
    from week3.project1.utils import count_parameters
    body = (f"<h3>{_esc(title or type(model).__name__)}</h3>"
            f"<p class='sub'>{count_parameters(model):,} parameters · bottleneck {tuple(b.shape[1:])} · "
            f"{len(skips)} skip connection{'s' if len(skips) != 1 else ''} · output {tuple(out.shape[1:])}. "
            f"Block height = spatial size, width = channels (log scale); labels show channels × height². The bottleneck is highlighted.</p>"
            f"<div style='display:flex;align-items:flex-end;gap:4px;overflow-x:auto;padding-bottom:6px'>{''.join(items)}</div>")
    _show(_wrap(body))


# ── Dataset self-check ───────────────────────────────────────────────────────

def check_dataset(student_ds, cache, n=5):
    """Compare a hand-written SegmentationDataset with the reference cache."""
    import torch
    rows, ok_all = [], True
    ok_mark, no_mark = "<span class='badge ok'>✓</span>", "<span class='badge no'>✕</span>"
    for i in np.linspace(0, len(cache) - 1, n).astype(int):
        try:
            img, target = student_ds[i][:2]
            ref_img = cache.images[i].float() / 255.0
            ref_target = cache.targets[i].long()
            target = torch.as_tensor(target)
            checks = {
                "shape": tuple(img.shape) == tuple(ref_img.shape) and tuple(target.squeeze().shape) == tuple(ref_target.shape),
                "image values": torch.allclose(img.float(), ref_img, atol=1 / 255 + 1e-6),
                "classes 0/1/2": bool(torch.isin(target.unique(), torch.tensor([0, 1, 2])).all()),
                "class map matches": float((target.squeeze().long() == ref_target).float().mean()) > 0.999,
            }
        except Exception as e:  # noqa: BLE001 — show the student what failed
            rows.append(f"<tr><td>{cache.files[i]}</td><td colspan='4' style='color:var(--bad)'>{_esc(type(e).__name__)}: {_esc(e)}</td></tr>")
            ok_all = False
            continue
        ok_all &= all(checks.values())
        cells = "".join(f"<td>{ok_mark if v else no_mark}</td>" for v in checks.values())
        rows.append(f"<tr><td>{cache.files[i]}</td>{cells}</tr>")
    head = "<tr><th>File</th><th>Shape</th><th>Image values</th><th>Classes 0/1/2</th><th>Class map matches</th></tr>"
    verdict = ("<span class='badge ok'>✓ Your dataset matches the reference loader</span>" if ok_all
               else "<span class='badge no'>✕ Something differs — check the rows below</span>")
    _show(_wrap(f"<h3>Dataset check</h3><p class='sub'>{verdict}</p><table>{head}{''.join(rows)}</table>"))
    return ok_all


# ── Intro animation: the assembly line ───────────────────────────────────────
#
# Timeline: the belt moves in steps. Every _AL_S seconds it advances one pitch
# in _AL_MV seconds, then pauses while the camera scans the tile at stop 2 and
# the diverter handles the tile at stop 3. All tiles share the belt, so they
# all move and pause together. _AL_K tile "slots" follow the same tracks, each
# one step apart; a slot's loop lasts _AL_K steps:
#   step 0 intake -> stop 1 · step 1 -> camera · step 2 -> diverter (badge,
#   rejects pushed into the bin) · step 3 -> stop 4 · step 4 -> resale crate ·
#   step 5 hidden behind the intake, where JS swaps in a new random tile.
# Stations are placed ON the stop grid (camera = stop 2, diverter/bin = stop 3).
# Positions are SVG transform attributes set by JS from the tracks below (user
# units in every browser/zoom); CSS keyframes only animate unitless things
# (opacity, scale, rotation about a local origin) and provide the clock.

_AL_S, _AL_MV, _AL_K, _AL_T0 = 2.8, 0.75, 6, 1.45
_AL_PITCH = 200                                   # belt step (a multiple of the 25-unit slat spacing)
_AL_X = tuple(30 + k * _AL_PITCH for k in range(5))  # stop k tile centre = belt start + tile/2 + k*pitch
_AL_CAM, _AL_DIV, _AL_END = _AL_X[2], _AL_X[3], _AL_X[4]
_AL_Y = 245                                       # tile centre y on the belt
_AL_PUSH = 58                                     # pusher stroke
_AL_EASE = {"io": (.45, 0, .35, 1), "in": (.45, 0, 1, 1), "out": (.2, .3, .6, 1), "push": (.5, 0, 1, .7)}

# (face colour, pattern) — tiles are drawn on a 70x70 face centred at 0
_AL_TILES = [
    ("#c8643b", "plain"), ("#f1e8d6", "checker"), ("#efe6d2", "encaustic"),
    ("#4f6f99", "star"), ("#e3cf9f", "bowtie"), ("#86a882", "border"), ("#d9a33a", "argyle"),
]
_AL_CRACKS = ("M-35,-9L-22,-3L-14,-15L-3,-1L7,-9L15,5L25,1L35,13M-3,-1L-7,11L1,20",
              "M-12,-35L-6,-20L-15,-8L-4,4L-9,17L2,35M-4,4L10,9L18,2")
_AL_CHIP = "-35,-35 5,-35 10,-27 18,-25 21,-17 28,-13 35,-3 35,35 -35,35"
_AL_CHIP_EDGE = "M5,-35L10,-27L18,-25L21,-17L28,-13L35,-3"


def _al_shade(hex_, f):
    r, g, b = (int(hex_[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, c * f))) for c in (r, g, b))


def _al_face(base, pattern):
    q = [f"<rect x='-35' y='-35' width='70' height='70' rx='3' fill='{base}'/>"]
    ink, teal, terra, gold, cream = "#34373d", "#2f8f8a", "#c8643b", "#d9a33a", "#f4ecdc"
    if pattern == "plain":
        q.append("<rect x='-29' y='-29' width='58' height='58' rx='2' fill='none' stroke='#fff' stroke-opacity='.22' stroke-width='2'/>")
    elif pattern == "checker":
        q += [f"<rect x='{-35 + 17.5 * c}' y='{-35 + 17.5 * r}' width='17.5' height='17.5' fill='{ink}'/>"
              for r in range(4) for c in range(4) if (r + c) % 2]
    elif pattern == "encaustic":
        q += [f"<path d='M{sx * 35},{sy * 35}h{-sx * 21}a21,21 0 0 {1 if sx * sy > 0 else 0} {sx * 21},{-sy * 21}z' fill='{teal}'/>"
              for sx in (-1, 1) for sy in (-1, 1)]
        q.append(f"<path d='M0,-19L19,0L0,19L-19,0z' fill='{terra}'/><circle r='6' fill='{gold}'/>")
    elif pattern == "star":
        q.append(f"<rect x='-16' y='-16' width='32' height='32' fill='{cream}'/>"
                 f"<rect x='-16' y='-16' width='32' height='32' fill='{cream}' transform='rotate(45)'/>"
                 f"<circle r='8' fill='{gold}'/><rect x='-31' y='-31' width='62' height='62' fill='none' stroke='{cream}' stroke-width='1.5' stroke-opacity='.7'/>")
    elif pattern == "bowtie":
        q.append(f"<path d='M-35,-35L0,0L-35,35z M35,-35L0,0L35,35z' fill='{teal}'/><circle r='5' fill='{cream}'/>")
    elif pattern == "border":
        q.append(f"<rect x='-27' y='-27' width='54' height='54' fill='none' stroke='{cream}' stroke-width='3'/>"
                 f"<rect x='-19' y='-19' width='38' height='38' fill='none' stroke='{cream}' stroke-width='1.5' stroke-opacity='.8'/>")
    elif pattern == "argyle":
        q.append(f"<path d='M0,-35L17.5,0L0,35L-17.5,0z M-35,-35L-17.5,0L-35,35z M35,-35L17.5,0L35,35z' fill='{cream}'/>"
                 f"<path d='M-35,0H35M0,-35V35' stroke='{terra}' stroke-width='1.2' stroke-opacity='.6'/>")
    return "".join(q)


def _al_tile_defs(uid):
    """One <g> per tile look: intact ('i'), cracked ('c') and chipped ('h'), plus the model overlays."""
    out, good, bad = [], [], []
    for k, (base, pattern) in enumerate(_AL_TILES):
        face = _al_face(base, pattern)
        for dmg in "ich":
            tid = f"t{k}{dmg}"
            (good if dmg == "i" else bad).append(tid)
            clip = f" clip-path='url(#{uid}-chip)'" if dmg == "h" else ""
            extra = ""
            if dmg == "c":
                d = _AL_CRACKS[k % 2]
                extra = (f"<path d='{d}' fill='none' stroke='#fff' stroke-opacity='.6' stroke-width='5' stroke-linejoin='round' stroke-linecap='round'/>"
                         f"<path d='{d}' fill='none' stroke='#1d1a18' stroke-width='2.2' stroke-linejoin='round' stroke-linecap='round'/>")
            elif dmg == "h":
                extra = f"<path d='{_AL_CHIP_EDGE}' fill='none' stroke='#1d1a18' stroke-opacity='.45' stroke-width='2' stroke-linejoin='round'/>"
            edge = "M-35,27H35V33Q35,36 32,36H-32Q-35,36 -35,33z"
            sil = f"<polygon points='{_AL_CHIP}'/>" if dmg == "h" else "<rect x='-35' y='-35' width='70' height='70' rx='3'/>"
            out.append(f"<g id='{uid}-{tid}'><g transform='translate(3,4)' opacity='.28'><path d='{edge}'/>"
                       f"<g transform='scale(1,.8)'>{sil}</g></g><path d='{edge}' fill='{_al_shade(base, .72)}'/>"
                       f"<g transform='scale(1,.8)'><g{clip}>{face}{extra}</g></g></g>")
    # what the model "sees": damage highlighted in red, intact tiles framed in green
    hl = "fill='none' stroke='var(--bad)' stroke-linejoin='round' stroke-linecap='round'"
    for j, d in enumerate(_AL_CRACKS):
        out.append(f"<g id='{uid}-mc{j}'><g transform='scale(1,.8)'><path d='{d}' {hl} stroke-width='9' stroke-opacity='.45'/>"
                   f"<path d='{d}' {hl} stroke-width='2'/></g></g>")
    out.append(f"<g id='{uid}-mh'><g transform='scale(1,.8)'><path d='M3,-38L38,-38L38,-1L35,-3L28,-13L21,-17L18,-25L10,-27z' "
               f"fill='var(--bad)' fill-opacity='.3' stroke='var(--bad)' stroke-width='2' stroke-dasharray='4 3'/></g></g>")
    out.append(f"<g id='{uid}-mi'><path d='M-39,-18V-32H-25M25,-32H39V-18M39,26V40H25M-25,40H-39V26' fill='none' "
               f"stroke='var(--good)' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/></g>")
    return "".join(out), good, bad


def _al_mask(tid):
    """Overlay id for a tile id such as 't3c'."""
    return {"i": "mi", "h": "mh"}.get(tid[-1], f"mc{int(tid[1:-1]) % 2}")


def _al_tracks():
    """Position tracks (SVG user units), evaluated identically in Python (first frame) and JS.
    Each stop: (time_s, values, easing to the next stop or None for linear)."""
    S, M, P, X, Y, D = _AL_S, _AL_MV, _AL_S * _AL_K, _AL_X, _AL_Y, _AL_PUSH
    t2, crate = 2 * S, _AL_END + 105
    ride = []                                     # common ride: intake -> diverter stop
    for k in range(1, 4):
        ride += [((k - 1) * S, (X[k - 1], Y), "io"), ((k - 1) * S + M, (X[k], Y), None)]
    approve = ride + [(3 * S, (X[3], Y), "io"), (3 * S + M, (X[4], Y), None), (4 * S, (X[4], Y), "in"),
                      (4 * S + .42, (_AL_END + 48, Y + 3), "out"), (4 * S + .75, (crate, 336), None),
                      (5 * S - .05, (crate, 336), None), (5 * S - .02, (X[0], Y), None), (P, (X[0], Y), None)]
    reject = ride + [(5 * S - .02, (X[3], Y), None), (5 * S - .01, (X[0], Y), None), (P, (X[0], Y), None)]
    push = [(0, (0, 0, 1), None), (t2 + 1.2, (0, 0, 1), "io"), (t2 + 1.55, (D, 0, 1), "push"),
            (t2 + 1.95, (D + 50, 10, .72), None), (5 * S + .02, (D + 50, 10, .72), None), (5 * S + .03, (0, 0, 1), None),
            (P, (0, 0, 1), None)]
    rod = [(0, (0,), None), (t2 + 1.2, (0,), "io"), (t2 + 1.55, (D,), None), (t2 + 1.75, (D,), "io"),
           (t2 + 2.15, (0,), None), (P, (0,), None)]
    belt = [(0, (0,), "io"), (M, (_AL_PITCH,), None), (S, (_AL_PITCH,), None)]
    return {"a": (P, approve), "r": (P, reject), "p": (P, push), "d": (P, rod), "b": (S, belt)}


def _al_bezier(x1, y1, x2, y2, x):
    bx = lambda t: 3 * x1 * t * (1 - t) ** 2 + 3 * x2 * t * t * (1 - t) + t ** 3
    lo, hi = 0.0, 1.0
    for _ in range(22):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if bx(mid) < x else (lo, mid)
    t = (lo + hi) / 2
    return 3 * y1 * t * (1 - t) ** 2 + 3 * y2 * t * t * (1 - t) + t ** 3


def _al_eval(track, u):
    period, stops = track
    u %= period
    for (t0, v0, e), (t1, v1, _) in zip(stops, stops[1:]):
        if t0 <= u < t1:
            f = (u - t0) / (t1 - t0)
            f = _al_bezier(*_AL_EASE[e], f) if e else f
            return [a + (b - a) * f for a, b in zip(v0, v1)]
    return list(stops[-1][1])


def _al_tf(kind, v):
    """SVG transform attribute for a track value (same format as the JS side)."""
    r = lambda x: round(x, 2)
    if kind in "ar":
        return f"translate({r(v[0])},{r(v[1])})"
    if kind == "p":
        return f"translate(0,{r(v[0])}) rotate({r(v[1])}) scale({r(v[2])})"
    return f"translate({r(v[0])},0)" if kind == "b" else f"translate(0,{r(v[0])})"


def _al_kf(name, period, stops):
    """CSS keyframes. stops: (time_s, css declarations, easing to the next stop or None)."""
    body = "".join(f"{100 * t / period:.3f}%{{{d}{';animation-timing-function:' + e if e else ''}}}" for t, d, e in stops)
    return f"@keyframes {name}{{{body}}}"


def _al_keyframes():
    """Unitless CSS animations only: opacity, scale and rotation about a local origin."""
    import math
    S, M, K = _AL_S, _AL_MV, _AL_K
    P, t2, E = S * K, 2 * _AL_S, "cubic-bezier(.45,0,.35,1)"
    op = lambda *pts: [(t, f"opacity:{o}", None) for t, o in pts]
    tile_o = op((0, 1), (5 * S - .06, 1), (5 * S - .05, 0), (5 * S + .05, 0), (5 * S + .1, 1), (P, 1))
    push_o = op((0, 1), (t2 + 1.95, 1), (t2 + 2.0, 0), (5 * S + .05, 0), (5 * S + .1, 1), (P, 1))
    rod_o = op((0, 0), (t2 - .12, 0), (t2 - .1, 1), (3 * S + .1, 1), (3 * S + .12, 0), (P, 0))
    ov = op((0, 0), (S + 1.5, 0), (S + 1.8, 1), (2 * S + .25, 1), (2 * S + .6, 0), (P, 0))
    pop = "cubic-bezier(.34,1.7,.64,1)"
    bs = lambda sc, o: f"transform:scale({sc});opacity:{o}"
    bda = [(0, bs(.3, 0), None), (t2 + .85, bs(.3, 0), pop), (t2 + 1.2, bs(1, 1), None),
           (4 * S + .05, bs(1, 1), None), (4 * S + .4, bs(1, 0), None), (P, bs(.3, 0), None)]
    bdr = [(0, bs(.3, 0), None), (t2 + .85, bs(.3, 0), pop), (t2 + 1.2, bs(1, 1), None),
           (3 * S - .1, bs(1, 1), None), (3 * S + .3, bs(1, 0), None), (P, bs(.3, 0), None)]
    turn = 90 * round(math.degrees(_AL_PITCH / 6.06) / 90)   # roller turn per step, a multiple of 90 deg
    fan = f"{math.degrees(math.atan(32 / 110)):.2f}"          # scan fan swing: +-32 units at the tile
    per = [
        _al_kf("whal-roll", S, [(0, "transform:rotate(0deg)", E), (M, f"transform:rotate({turn}deg)", None), (S, f"transform:rotate({turn}deg)", None)]),
        _al_kf("whal-beam", S, op((0, 0), (.8, 0), (.95, 1), (2.0, 1), (2.2, 0), (S, 0))),
        _al_kf("whal-fan", S, [(0, f"transform:rotate({fan}deg)", None), (.95, f"transform:rotate({fan}deg)", E),
                               (1.5, f"transform:rotate(-{fan}deg)", E), (2.02, f"transform:rotate({fan}deg)", None), (S, f"transform:rotate({fan}deg)", None)]),
        _al_kf("whal-scan", S, op((0, 0), (.74, 0), (.76, 1), (2.04, 1), (2.06, 0), (S, 0))),
        _al_kf("whal-res", S, op((0, 1), (.74, 1), (.76, 0), (2.04, 0), (2.06, 1), (S, 1))),
        _al_kf("whal-prog", S, [(0, "transform:scaleX(0)", None), (.8, "transform:scaleX(0)", None), (2.04, "transform:scaleX(1)", None),
                                (S, "transform:scaleX(1)", None)]),
        _al_kf("whal-led", S, op((0, .25), (.8, .25), (.85, 1), (1.2, .35), (1.55, 1), (1.9, .35), (2.05, 1), (2.2, .25), (S, .25))),
    ]
    slot = [_al_kf("whal-tile", P, tile_o), _al_kf("whal-psh", P, push_o), _al_kf("whal-rod", P, rod_o),
            _al_kf("whal-bda", P, bda), _al_kf("whal-bdr", P, bdr), _al_kf("whal-ov", P, ov)]
    return "".join(per + slot), turn


def _al_css():
    S, P = _AL_S, _AL_S * _AL_K
    kf, _ = _al_keyframes()
    a = lambda cls, name, per: f".wh .al {cls}{{animation:{name} {per}s linear infinite;animation-delay:var(--d)}}"
    rules = [
        ".wh .al{--al-belt:#43423f;--al-slat:#51504c;--al-rail:#898781;--al-leg:#c3c2b7;--al-metal:#d6d4cc;--al-metal2:#a9a79e;"
        "--al-house:#2a78d6;--al-house2:#1f5fae;--al-bin:#5d6360;--al-bin2:#2f3331;--al-crate:#c99a5b;--al-crate2:#a67a40;--al-hole:#2a2a28;"
        f"--al-floor:#e9e8e2;--al-beam:#46c8ff;--d:{-_AL_T0}s;display:block;width:100%;height:auto;border-radius:10px;background:var(--plane)}}",
        "@media (prefers-color-scheme:dark){.wh .al{--al-belt:#121211;--al-slat:#222220;--al-rail:#5f5e5a;--al-leg:#383835;"
        "--al-metal:#4a4946;--al-metal2:#6a6964;--al-house:#3987e5;--al-house2:#2667b8;--al-bin:#4d5350;--al-bin2:#1c1e1d;"
        "--al-crate:#a47b45;--al-crate2:#7f5c2f;--al-hole:#0e0e0d;--al-floor:#2c2c2a;--al-beam:#5fd4ff}}",
        ".wh .al text{font-family:inherit} .wh .al text.lb{fill:#fff;font-size:12px;font-weight:700;letter-spacing:.08em}",
        ".wh .al text.st{fill:var(--ink2);font-size:13px} .wh .al text.st tspan.n{font-weight:700;fill:var(--ink)}",
        ".wh .al text.scr1{fill:#9c9a93;font-size:10px;letter-spacing:.12em;font-weight:600}",
        ".wh .al text.scr2{fill:#fff;font-size:21px;font-weight:700;font-variant-numeric:tabular-nums}",
        ".wh .al text.scr3{font-size:14px;font-weight:700;letter-spacing:.04em}",
        ".wh .al .vd[data-o=a] .no,.wh .al .vd[data-o=r] .ok{display:none}",
        ".wh .al .vd[data-o=a] text{fill:#2bc02b} .wh .al .vd[data-o=r] text{fill:#f06a6a}",
        ".wh .al text.bd{font-size:13px;font-weight:700;letter-spacing:.03em}",
        ".wh .al [data-o=a] .lr,.wh .al [data-o=r] .la,.wh .al .rd[data-o=a] .rr,.wh .al .rd[data-o=r] .ra{visibility:hidden}",
        a(".la", "whal-tile", P), a(".psh", "whal-psh", P), a(".rd", "whal-rod", P), a(".la .bdg", "whal-bda", P),
        a(".lr .bdg", "whal-bdr", P), a(".ov", "whal-ov", P),
        a(".rl", "whal-roll", S), a(".beam", "whal-beam", S), a(".fan", "whal-fan", S),
        a(".scan", "whal-scan", S), a(".res", "whal-res", S), a(".prg", "whal-prog", S), a(".led", "whal-led", S),
        ".wh.paused .al *{animation-play-state:paused!important}",
        ".wh:not(.js) .al *{animation-play-state:paused!important}",   # no JS: a coherent still frame
        ".wh .al-bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px}",
        ".wh .al-bar .badge{font-size:13px;padding:3px 10px;font-variant-numeric:tabular-nums}",
        ".wh .al-bar .sp{flex:1}",
    ]
    return "".join(rules) + kf


def _al_icon(ok, x, y, r=9):
    """Filled status circle with a white check / cross (never colour alone)."""
    col = "var(--good)" if ok else "var(--bad)"
    s = r * .45
    mark = (f"<path d='M{x - s},{y}l{s * .7},{s * .75}l{s * 1.3},{-s * 1.5}' fill='none' stroke='#fff' stroke-width='{r * .28:.1f}' stroke-linecap='round' stroke-linejoin='round'/>"
            if ok else
            f"<path d='M{x - s},{y - s}l{2 * s},{2 * s}m0,{-2 * s}l{-2 * s},{2 * s}' fill='none' stroke='#fff' stroke-width='{r * .28:.1f}' stroke-linecap='round'/>")
    return f"<circle cx='{x}' cy='{y}' r='{r}' fill='{col}'/>{mark}"


def _al_badge(ok):
    """Pill centred on (0, -44) above the tile centre; scales about its own centre."""
    col = "var(--good-ink)" if ok else "var(--bad)"
    ring = "var(--good)" if ok else "var(--bad)"
    label = "APPROVED" if ok else "REJECTED"
    return (f"<g transform='translate(0,-44)'><g class='bdg'>"
            f"<path d='M-6,14L0,21L6,14z' fill='{ring}'/>"
            f"<rect x='-57' y='-14' width='114' height='28' rx='14' fill='var(--surface)' stroke='{ring}' stroke-width='2'/>"
            f"{_al_icon(ok, -40, 0)}<text class='bd' x='-26' y='4.5' style='fill:{col}'>{label}</text></g></g>")


def assembly_line_intro(height=None):
    """
    Looping intro animation: reclaimed tiles ride a conveyor through a camera
    inspection machine; the vision model approves sellable tiles (-> resale
    crate) and rejects cracked or chipped ones (-> pushed into the recycle bin).
    height: optional maximum height of the scene in pixels (it otherwise scales
    with the notebook width, up to 1000 px wide).
    """
    import math
    uid = _uid()
    S, K, P = _AL_S, _AL_K, _AL_S * _AL_K
    C, V, N = _AL_CAM, _AL_DIV, _AL_END            # station x: camera, diverter/bin, last stop
    tracks = _al_tracks()
    _, turn = _al_keyframes()
    tiles, good, bad = _al_tile_defs(uid)
    # first loop: slot i starts at step i (so at t=0 slot 2 is at the diverter)
    init = [("t0i", 96), ("t3h", 61), ("t2c", 47), ("t1i", 98), ("t6i", 93), ("t4i", 95)]
    init = [{"d": d, "ok": d.endswith("i"), "u": u, "m": _al_mask(d)} for d, u in init]
    tf = lambda kind, off: _al_tf(kind, _al_eval(tracks[kind], off))   # first-frame transforms (JS takes over)

    # ── static scenery ──
    legs = "".join(f"<rect x='{x}' y='300' width='10' height='78' fill='var(--al-leg)'/>" for x in (150, 290, V - 82, V + 82, N + 15))
    slats = "".join(f"<rect x='{x}' y='207' width='4' height='78' rx='2' fill='var(--al-slat)'/>" for x in range(-140, N + 40, 25))
    rr = _AL_PITCH / math.radians(turn)     # roller radius that turns exactly `turn` degrees per step
    rollers = "".join(f"<g transform='translate({x},295)'><g class='rl'><circle r='{rr:.2f}' fill='var(--al-metal2)' stroke='var(--al-belt)' stroke-width='1.5'/>"
                      f"<path d='M-4,0h8M0,-4v8' stroke='var(--al-belt)' stroke-width='1.6'/></g></g>" for x in range(96, N + 36, 40))
    pile = "".join(f"<rect x='{x}' y='{y}' width='44' height='10' rx='2' fill='{c}' transform='rotate({r} {x + 22} {y + 5})'/>"
                   for x, y, c, r in ((8, 112, "#c8643b", -8), (40, 108, "#4f6f99", 6), (66, 113, "#d9a33a", -5), (26, 104, "#86a882", 12)))
    crate_fill = "".join(f"<rect x='{N + x}' y='{y}' width='50' height='12' rx='2' fill='{c}' transform='rotate({r} {N + x + 25} {y + 6})'/>"
                         for x, y, c, r in ((54, 287, "#86a882", -4), (100, 285, "#efe6d2", 5), (75, 280, "#4f6f99", -2)))
    shards = (f"<path d='M{V - 40},318l14,-8l8,9z M{V - 6},316l18,-7l-4,10z M{V + 22},319l10,-9l12,7z' fill='#c8643b' opacity='.9'/>"
              f"<path d='M{V - 24},320l10,-6l6,6z M{V + 10},321l9,-5l7,5z' fill='#efe6d2' opacity='.9'/>")
    belt_w = N + 37 - 70

    back = f"""
<rect x='0' y='376' width='1000' height='40' fill='var(--al-floor)'/>
<rect x='{C - 73}' y='140' width='146' height='70' fill='var(--al-hole)'/>
<rect class='beam' x='{C - 73}' y='140' width='146' height='70' fill='var(--al-beam)' opacity='0' fill-opacity='.18'/>
{legs}
<rect x='{V - 27}' y='160' width='7' height='46' fill='var(--al-leg)'/><rect x='{V + 20}' y='160' width='7' height='46' fill='var(--al-leg)'/>
<rect x='70' y='203' width='{belt_w}' height='86' rx='8' fill='var(--al-belt)'/>
<g clip-path='url(#{uid}-belt)'><g class='slats' transform='{tf("b", _AL_T0)}'>{slats}</g></g>
<rect x='70' y='286' width='{belt_w}' height='18' rx='3' fill='var(--al-rail)'/>
{rollers}
<path d='M{V - 56},308H{V + 56}L{V + 60},324H{V - 60}z' fill='var(--al-bin2)'/>{shards}
<path d='M{N + 44},282H{N + 166}V300H{N + 44}z' fill='var(--al-crate2)'/>{crate_fill}"""

    # ── moving tiles (one group per slot), pusher rods, and the badges (top layer) ──
    rod = (f"<rect x='{V - 6}' y='118' width='12' height='90' rx='2' fill='var(--al-metal2)'/>"
           f"<rect x='{V - 32}' y='205' width='64' height='9' rx='3' fill='var(--al-metal2)' stroke='var(--al-belt)' stroke-width='1'/>")
    rods, slots, badges = [], [], []
    for i, st in enumerate(init):
        o, off = ("a" if st["ok"] else "r"), i * S + _AL_T0
        d = f"--d:{-off:.2f}s"
        rods.append(f"<g class='rd' data-o='{o}' style='{d}'><g class='ra'>{rod}</g><g class='rr' transform='{tf('d', off)}'>{rod}</g></g>")
        use = f"<use href='#{uid}-{st['d']}'/><use class='ov' href='#{uid}-{st['m']}'/>"
        slots.append(f"<g class='sl' data-o='{o}' style='{d}'>"
                     f"<g class='la' transform='{tf('a', off)}'>{use}</g>"
                     f"<g class='lr' transform='{tf('r', off)}'><g class='psh' transform='{tf('p', off)}'>{use}</g></g></g>")
        badges.append(f"<g class='sb' data-o='{o}' style='{d}'><g class='la' transform='{tf('a', off)}'>{_al_badge(True)}</g>"
                      f"<g class='lr' transform='{tf('r', off)}'>{_al_badge(False)}</g></g>")
    pusher = (f"<rect class='dvh' x='{V - 36}' y='110' width='72' height='66' rx='8' fill='var(--al-metal)' stroke='var(--al-metal2)' stroke-width='1.5'/>"
              f"<rect x='{V - 24}' y='124' width='48' height='6' rx='3' fill='var(--al-metal2)'/><rect x='{V - 24}' y='136' width='48' height='6' rx='3' fill='var(--al-metal2)'/>"
              f"<circle cx='{V + 22}' cy='162' r='3.5' fill='var(--bad)'/>"
              f"<text class='st' x='{V}' y='102' text-anchor='middle' style='font-size:11px;letter-spacing:.08em;font-weight:600'>DIVERTER</text>")

    # ── machine front, bin, crate, intake ──
    first = init[1]
    scr_ok = "a" if first["ok"] else "r"
    x0 = C - 69                                   # screen text inset
    front = f"""
<g class='beam' opacity='0'>
 <path d='M{C - 5},168H{C + 5}L{C + 46},281H{C - 46}z' fill='var(--al-beam)' opacity='.1'/>
 <g transform='translate({C},166)'><g class='fan'><path d='M-2,2H2L7,110H-7z' fill='url(#{uid}-bg)'/>
 <rect x='-3' y='48' width='6' height='64' rx='3' fill='var(--al-beam)' filter='url(#{uid}-glow)'/>
 <rect x='-1' y='50' width='2' height='60' rx='1' fill='#fff'/></g></g>
</g>
<path class='house' fill-rule='evenodd' fill='var(--al-house)' d='M{C - 83},8H{C + 83}Q{C + 95},8 {C + 95},20V380H{C - 95}V20Q{C - 95},8 {C - 83},8Z M{C - 73},150H{C + 73}V304H{C - 73}Z'/>
<rect x='{C - 95}' y='304' width='190' height='76' fill='var(--al-house2)'/>
<rect x='{C - 95}' y='150' width='22' height='154' fill='var(--al-house2)' opacity='.55'/>
<text class='lb' x='{C}' y='27' text-anchor='middle' style='font-size:11px'>AI INSPECTION</text>
<rect x='{C - 81}' y='35' width='162' height='100' rx='7' fill='#111211' stroke='#000' stroke-opacity='.3'/>
<text class='scr1' x='{x0}' y='54'>VISION MODEL</text>
<circle class='led' cx='{C + 68}' cy='50' r='4' fill='#f06a6a'/>
<g class='res'>
 <text class='scr2' x='{x0}' y='88'>usable <tspan class='pct'>{first['u']}%</tspan></text>
 <g class='vd' data-o='{scr_ok}'><g class='ok'>{_al_icon(True, x0 + 8, 114, 8)}</g><g class='no'>{_al_icon(False, x0 + 8, 114, 8)}</g>
 <text class='scr3' x='{x0 + 22}' y='119'><tspan class='ok'>APPROVE</tspan><tspan class='no'>REJECT</tspan></text></g>
</g>
<g class='scan' opacity='0'>
 <text class='scr2' x='{x0}' y='88' style='font-size:17px'>scanning…</text>
 <rect x='{x0}' y='108' width='138' height='8' rx='4' fill='#2c2d2c'/>
 <g transform='translate({x0},108)'><rect class='prg' width='138' height='8' rx='4' fill='var(--al-beam)'/></g>
</g>
<rect x='{C - 19}' y='146' width='38' height='15' rx='4' fill='#1c1d1c'/>
<circle cx='{C}' cy='163' r='7' fill='#1c1d1c'/><circle class='cam' cx='{C}' cy='163' r='4' fill='#3a8fe8'/>
<circle cx='{C + 11}' cy='152' r='2' class='led' fill='#f06a6a'/>
{pusher}
<rect class='binf' x='{V - 60}' y='320' width='120' height='60' rx='5' fill='var(--al-bin)'/>
<rect x='{V - 60}' y='320' width='120' height='6' rx='3' fill='#000' opacity='.18'/>
{_al_icon(False, V - 34, 350, 10)}<text class='lb' x='{V - 19}' y='355'>RECYCLE</text>
<rect x='{N + 42}' y='298' width='126' height='82' rx='4' fill='var(--al-crate)'/>
<path d='M{N + 42},323H{N + 168}M{N + 42},348H{N + 168}' stroke='var(--al-crate2)' stroke-width='2'/>
{_al_icon(True, N + 70, 336, 10)}<text class='lb' x='{N + 85}' y='341'>RESALE</text>
{pile}<path d='M-4,120H124L108,172H12z' fill='var(--al-metal2)'/><rect x='-4' y='116' width='128' height='8' rx='3' fill='var(--al-metal2)' stroke='var(--al-leg)' stroke-width='1'/>
<rect x='0' y='170' width='118' height='210' fill='var(--al-metal)'/><path d='M108,196h10v100h-10z' fill='var(--al-hole)'/>
<text class='st' x='56' y='232' text-anchor='middle' style='font-weight:700;letter-spacing:.06em;font-size:12px'>INTAKE</text>
<text class='st' x='56' y='250' text-anchor='middle' style='font-size:11px'>reclaimed</text>
<text class='st' x='56' y='264' text-anchor='middle' style='font-size:11px'>tiles</text>
<text class='st' x='{_AL_X[1]}' y='400' text-anchor='middle'><tspan class='n'>1</tspan> tile arrives</text>
<text class='st' x='{C}' y='400' text-anchor='middle'><tspan class='n'>2</tspan> camera + model score it</text>
<text class='st' x='{V}' y='400' text-anchor='middle'><tspan class='n'>3</tspan> reject → recycle</text>
<text class='st' x='{N + 105}' y='400' text-anchor='middle'><tspan class='n'>4</tspan> approve → resale</text>"""

    defs = f"""<defs>
<clipPath id='{uid}-belt'><rect x='70' y='203' width='{belt_w}' height='86' rx='8'/></clipPath>
<clipPath id='{uid}-chip'><polygon points='{_AL_CHIP}'/></clipPath>
<linearGradient id='{uid}-bg' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='var(--al-beam)' stop-opacity='.9'/>
<stop offset='1' stop-color='var(--al-beam)' stop-opacity='.35'/></linearGradient>
<filter id='{uid}-glow' x='-3' y='-.3' width='7' height='1.6'><feGaussianBlur stdDeviation='3'/></filter>
{tiles}</defs>"""
    aria = ("Animated illustration: floor tiles ride a conveyor belt one at a time into an AI inspection machine. "
            "A camera scans each tile and a screen shows how much of it is usable. Intact tiles get a green check "
            "badge, Approved, and continue to the resale crate; cracked or chipped tiles get a red cross badge, "
            "Rejected, and a pusher diverts them into the recycle bin.")
    # keep the aspect ratio exact (no letterboxing, nothing drawn outside the frame)
    style = f"width:{int(height) * 1000 / 408:.0f}px;max-width:100%;margin:0 auto;" if height else ""
    svg = (f"<svg class='al' viewBox='0 0 1000 408' role='img' aria-label='{_esc(aria)}' style='{style}'>"
           f"{defs}{back}{''.join(rods)}{pusher}{''.join(slots)}{front}{''.join(badges)}</svg>")
    n_ok = sum(s["ok"] for s in init[2:5])
    cfg = {"S": S, "K": K, "P": P, "MV": _AL_MV, "T0": _AL_T0, "good": good, "bad": bad,
           "mask": {t: _al_mask(t) for t in good + bad}, "init": init, "ease": _AL_EASE,
           "tr": {k: [[t, list(v), e] for t, v, e in stops] for k, (_, stops) in tracks.items()}}
    body = f"""<style>{_al_css()}#{uid}{{max-width:1000px}}</style>
<h3>Automated tile inspection: what you will build in this project</h3>
<p class='sub'>Reclaimed tiles ride the belt one at a time. A camera scans each one, your computer-vision model
estimates how much of the tile is usable, and the line approves it for resale or diverts it to recycling.</p>
<div class='al-bar'><span class='badge ok'>✓ Approved <b class='na'>{n_ok}</b></span>
<span class='badge no'>✕ Rejected <b class='nr'>{3 - n_ok}</b></span><span class='sp'></span>
<button class='al-btn' type='button' aria-pressed='false'>❚❚ Pause</button></div>
{svg}
"""
    script = """
(function(){const U='__UID__',R=document.getElementById(U);if(!R)return;
const C=__DATA__,S=C.S,K=C.K,P=C.P,T0=C.T0,st=C.init.map(o=>Object.assign({},o));
const slots=[...R.querySelectorAll('.sl')],rods=[...R.querySelectorAll('.rd')],bdgs=[...R.querySelectorAll('.sb')],slats=R.querySelector('.slats');
const pct=R.querySelector('.pct'),vd=R.querySelector('.vd'),na=R.querySelector('.na'),nr=R.querySelector('.nr'),btn=R.querySelector('.al-btn');
const mod=(a,b)=>((a%b)+b)%b,rnd=a=>a[Math.floor(Math.random()*a.length)];
function bz(p){const X=t=>3*p[0]*t*(1-t)*(1-t)+3*p[2]*t*t*(1-t)+t*t*t;return x=>{let lo=0,hi=1;for(let i=0;i<22;i++){const m=(lo+hi)/2;if(X(m)<x)lo=m;else hi=m}
const t=(lo+hi)/2;return 3*p[1]*t*(1-t)*(1-t)+3*p[3]*t*t*(1-t)+t*t*t}}
const EZ={};for(const k in C.ease)EZ[k]=bz(C.ease[k]);
function ev(k,u){const s=C.tr[k],per=k==='b'?S:P;u=mod(u,per);for(let i=0;i<s.length-1;i++){const a=s[i],b=s[i+1];if(u>=a[0]&&u<b[0]){
let f=(u-a[0])/(b[0]-a[0]);if(a[2])f=EZ[a[2]](f);return a[1].map((v,j)=>v+(b[1][j]-v)*f)}}return s[s.length-1][1]}
const r2=x=>Math.round(x*100)/100;
function tf(k,v){return k==='p'?`translate(0,${r2(v[0])}) rotate(${r2(v[1])}) scale(${r2(v[2])})`:k==='b'?`translate(${r2(v[0])},0)`:
k==='d'?`translate(0,${r2(v[0])})`:`translate(${r2(v[0])},${r2(v[1])})`}
const q=(el,s)=>el.querySelector(s),els=slots.map((s,i)=>({la:[q(s,'.la'),q(bdgs[i],'.la')],lr:[q(s,'.lr'),q(bdgs[i],'.lr')],p:q(s,'.psh'),d:q(rods[i],'.rr')}));
function frame(g){els.forEach((e,i)=>{const u=g+i*S+T0,a=tf('a',ev('a',u)),r=tf('r',ev('r',u));e.la.forEach(x=>x.setAttribute('transform',a));
e.lr.forEach(x=>x.setAttribute('transform',r));e.p.setAttribute('transform',tf('p',ev('p',u)));e.d.setAttribute('transform',tf('d',ev('d',u)))});
slats.setAttribute('transform',tf('b',ev('b',g+T0)))}
let A=+na.textContent,J=+nr.textContent,counted=null,rolled=new Array(K).fill(null),clk=null,streak=0,g0=null,last=null;
function pick(){let bad=Math.random()<0.3;if(bad)streak++;else streak=0;if(streak>2){bad=false;streak=0}
return bad?{d:rnd(C.bad),ok:false,u:38+Math.floor(Math.random()*33)}:{d:rnd(C.good),ok:true,u:88+Math.floor(Math.random()*12)}}
function apply(i){const o=st[i].ok?'a':'r';[slots[i],rods[i],bdgs[i]].forEach(x=>x.setAttribute('data-o',o));
slots[i].querySelectorAll('use').forEach(u=>u.setAttribute('href','#'+U+'-'+(u.classList.contains('ov')?C.mask[st[i].d]:st[i].d)))}
function clock(){if(!clk){const el=R.querySelector('.beam');clk=el&&el.getAnimations?el.getAnimations().find(a=>a.animationName==='whal-beam')||0:0}
if(clk&&clk.currentTime!=null)return clk.currentTime/1000;if(g0===null)g0=performance.now();return R.classList.contains('paused')?last||0:(performance.now()-g0)/1000}
function logic(g){const n=Math.floor((g+T0)/S),tau=g+T0-n*S;
for(let i=0;i<K;i++){if(mod(i+n,K)===K-1&&rolled[i]!==n){rolled[i]=n;st[i]=pick();apply(i)}}
const s=st[mod((tau>=C.MV?1:2)-n,K)],o=s.ok?'a':'r';if(pct.textContent!==s.u+'%')pct.textContent=s.u+'%';
if(vd.getAttribute('data-o')!==o)vd.setAttribute('data-o',o);
if(tau>=0.9&&counted!==n){if(counted!==null){if(st[mod(2-n,K)].ok)A++;else J++;na.textContent=A;nr.textContent=J}counted=n}}
function loop(){if(!R.isConnected)return;const g=clock();if(g!==last){last=g;frame(g);logic(g)}requestAnimationFrame(loop)}
const mq=window.matchMedia?matchMedia('(prefers-reduced-motion: reduce)'):null;let user=!!(mq&&mq.matches),off=false;
function upd(){R.classList.toggle('paused',user||off);btn.textContent=user?'▶ Play':'❚❚ Pause';btn.setAttribute('aria-pressed',String(user))}
btn.onclick=()=>{user=!user;upd()};upd();R.classList.add('js');
if('IntersectionObserver' in window)new IntersectionObserver(e=>{off=!e[e.length-1].isIntersecting;upd()}).observe(R);
loop()})();""".replace("__UID__", uid)
    _show(_wrap(body, uid, script.replace('__DATA__', _js_data(cfg))))
