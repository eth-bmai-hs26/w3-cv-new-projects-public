"""
Model service: loads the student's segmentation model and turns a photo into
a 3-class mask, a usable fraction, a confidence and timings.

Supported checkpoints
    * dill whole-model pickles written by week3.project1.utils.save_model   (*.pt)
    * TorchScript files (torch.jit.save / torch.jit.trace)                   (*.pt, *.ts)
    * a dict {"model": <nn.Module>, "meta": {...}} saved with torch.save + dill

Supported outputs (model(x) with x of shape (B, 1, 256, 256) in [0, 1])
    * (B, 3, H, W) logits, classes 0 = belt, 1 = usable, 2 = damaged zone   <- the course model
    * (B, 1, H, W) logits, legacy: sigmoid > 0.5 = usable; the tile outline is then
      estimated by the demo heuristic (flagged as a warning)
    * a tuple/list whose first element is one of the above (e.g. forward_all)

No checkpoint configured (or it fails to load) -> DEMO MODE, a heuristic.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import heuristic
from .compat import ensure_course_modules

INPUT_SIZE = 256
BELT, USABLE, DAMAGED = 0, 1, 2

_EXPORT_TIP = (" Export the model with export_for_platform(...) in Section 11 of the notebook and upload "
               "platform_model.pt instead: that TorchScript file loads on any computer.")


def _load_hint(e: Exception) -> str:
    """A plain-language next step for the load errors students actually run into."""
    from . import compat
    text = str(e)
    if "no locals found" in text or "code() argument" in text or "bad marshal data" in text:
        return (" (this is a save_model checkpoint written by a different Python version, e.g. Colab's; "
                "it only loads on the Python that saved it)." + _EXPORT_TIP)
    if "Can't get attribute" in text:
        why = (f"the course code could not be imported here: {compat.repo_import_error}" if compat.repo_import_error
               else "the model uses a class from the course code, which is not next to this copy of the platform; "
                    "run the platform from inside the course repository")
        return f" ({why})." + _EXPORT_TIP
    if isinstance(e, ModuleNotFoundError):
        return " (a package the model needs is not installed in the platform's environment)." + _EXPORT_TIP
    return ""


@dataclass
class Prediction:
    classes: np.ndarray                 # (256, 256) uint8 in {0, 1, 2}
    usable_fraction: float | None       # usable / (usable + damaged); None when no tile
    tile_share: float                   # tile pixels / image pixels
    mask_certainty: float               # 0..1
    timings: dict                       # ms: preprocess, inference, postprocess, total
    warnings: list = field(default_factory=list)
    error: str | None = None


def pick_device(pref: str = "auto") -> torch.device:
    pref = (pref or "auto").lower()
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess(img: Image.Image) -> torch.Tensor:
    """Same as the notebook: grayscale -> 256x256 bilinear -> [0, 1]."""
    g = img.convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0)[None, None]


def _is_torchscript(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
        return any("/code/" in n or n.startswith("code/") for n in names) and any(n.endswith("constants.pkl") for n in names)
    except zipfile.BadZipFile:
        return False


def _file_version(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def _sidecar_meta(path: Path) -> dict:
    for cand in (path.with_suffix(".json"), Path(str(path) + ".json"), path.with_suffix(".meta.json")):
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except Exception:
                pass
    return {}


class ModelService:
    def __init__(self, checkpoint: str | None = None, device: str = "auto"):
        self._lock = threading.Lock()
        self.model = None
        self.kind = "demo"
        self.device = pick_device(device)
        self.device_pref = device
        self.info: dict = {}
        self.load(checkpoint)

    # ── loading ──────────────────────────────────────────────────────────────
    @property
    def demo(self) -> bool:
        return self.kind == "demo"

    def _demo_info(self, reason: str | None):
        self.model, self.kind = None, "demo"
        self.info = {
            "mode": "demo", "name": "Demo heuristic", "version": "heuristic-1",
            "kind": "demo", "classes": 3, "checkpoint": None, "device": "cpu",
            "params": 0, "metrics": {}, "warning": reason,
            "loaded_at": datetime.now().isoformat(timespec="seconds"),
            "description": "Colour/texture rules, not a trained model. Load a checkpoint in Settings.",
        }

    def load(self, checkpoint: str | None, device: str | None = None) -> dict:
        """Load a checkpoint (or fall back to demo mode). Never raises; problems land in info['warning']."""
        if device:
            self.device_pref = device
            self.device = pick_device(device)
        with self._lock:
            if not checkpoint:
                self._demo_info("No model file configured.")
                return self.info
            path = Path(checkpoint).expanduser()
            if not path.exists():
                self._demo_info(f"Model file not found: {path}")
                return self.info
            try:
                self._load_file(path)
            except Exception as e:                      # keep the station running
                self._demo_info(f"Could not load {path.name}: {type(e).__name__}: {e}{_load_hint(e)}")
        return self.info

    def _load_file(self, path: Path):
        meta = _sidecar_meta(path)
        if _is_torchscript(path):
            with warnings.catch_warnings():         # torch >= 2.10 flags jit.load as deprecated; it still works
                warnings.simplefilter("ignore", FutureWarning)
                model = torch.jit.load(str(path), map_location=self.device)
            kind = "torchscript"
        else:
            import dill
            ensure_course_modules()
            obj = torch.load(str(path), map_location=self.device, pickle_module=dill, weights_only=False)
            if isinstance(obj, dict) and "model" in obj:
                meta = {**obj.get("meta", {}), **meta}
                obj = obj["model"]
            if isinstance(obj, dict):
                raise ValueError("this file holds only weights (a state_dict). Save the whole model with "
                                 "save_model(model, name) or TorchScript instead.")
            if not isinstance(obj, torch.nn.Module):
                raise ValueError(f"expected a torch.nn.Module, got {type(obj).__name__}")
            model = obj
            kind = "dill"
        model.eval().to(self.device)
        meta = {**getattr(model, "platform_meta", {}), **meta} if not isinstance(model, torch.jit.ScriptModule) else meta

        # probe once: output channels decide how we read the mask
        with torch.no_grad():
            out = self._forward(model, torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=self.device))
        channels = int(out.shape[1])
        if channels not in (1, 3):
            raise ValueError(f"model returns {channels} channels; expected 3 (belt/usable/damaged) or 1 (legacy)")

        params = sum(p.numel() for p in model.parameters())
        st = path.stat()
        self.model, self.kind = model, kind
        self.info = {
            "mode": "model", "kind": kind, "classes": channels,
            "name": meta.get("name") or path.stem,
            "version": meta.get("version") or _file_version(path),
            "architecture": type(model).__name__ if kind == "dill" else getattr(model, "original_name", "TorchScript"),
            "checkpoint": str(path), "size_mb": round(st.st_size / 1e6, 2),
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "device": str(self.device), "params": int(params),
            "metrics": meta.get("metrics", {k: v for k, v in meta.items() if isinstance(v, (int, float))}),
            "trained_on": meta.get("trained_on"),
            "warning": None if channels == 3 else
                "1-channel (legacy) model: it only marks usable pixels, so the tile outline is estimated by a heuristic.",
            "loaded_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def _forward(model, x):
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("logits", out.get("out"))
        if out.dim() == 3:
            out = out[:, None]
        return out.float()

    # ── inference ────────────────────────────────────────────────────────────
    def predict(self, img: Image.Image) -> Prediction:
        t0 = time.perf_counter()
        img = img.convert("RGB")
        warnings = []
        if self.demo:
            rgb = np.asarray(img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))
            t1 = time.perf_counter()
            classes, q = heuristic.segment(rgb)
            t2 = time.perf_counter()
            certainty = 0.9 * self._certainty(q, classes > 0)     # demo never claims full certainty
            warnings.append("Demo mode: heuristic, not a trained model")
        else:
            x = preprocess(img).to(self.device)
            t1 = time.perf_counter()
            with self._lock, torch.no_grad():
                out = self._forward(self.model, x)
                if out.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
                    out = F.interpolate(out, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            t2 = time.perf_counter()
            if out.shape[1] == 3:
                p = torch.softmax(out[0], 0).cpu().numpy()
                classes = p.argmax(0).astype(np.uint8)
                q = p[USABLE] / np.clip(p[USABLE] + p[DAMAGED], 1e-6, None)
            else:
                pu = torch.sigmoid(out[0, 0]).cpu().numpy()
                usable = pu > 0.5
                hull, _ = heuristic.tile_mask(np.asarray(img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)))
                tile = hull | usable
                classes = np.where(usable, USABLE, np.where(tile, DAMAGED, BELT)).astype(np.uint8)
                q = np.where(tile, pu, 0).astype(np.float32)
                warnings.append("Legacy 1-channel model: tile area estimated by heuristic")
            certainty = self._certainty(q, classes > 0)
        t3 = time.perf_counter()

        n_use = int((classes == USABLE).sum())
        n_dmg = int((classes == DAMAGED).sum())
        n_tile = n_use + n_dmg
        min_tile = 0.005 * classes.size
        error, uf = None, None
        if n_tile < min_tile:
            error = "No tile detected"
        else:
            uf = n_use / n_tile
        timings = {
            "preprocess": round(1000 * (t1 - t0), 2),
            "inference": round(1000 * (t2 - t1), 2),
            "postprocess": round(1000 * (t3 - t2), 2),
        }
        timings["total"] = round(sum(timings.values()), 2)
        return Prediction(classes, uf, n_tile / classes.size, round(float(certainty), 4), timings, warnings, error)

    @staticmethod
    def _certainty(q: np.ndarray, tile: np.ndarray) -> float:
        """Mean |2q - 1| over tile pixels: 1 = every pixel clearly usable or clearly damaged."""
        if not tile.any():
            return 0.0
        return float(np.abs(2 * q[tile] - 1).mean())
