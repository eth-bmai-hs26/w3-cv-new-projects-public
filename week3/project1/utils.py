"""
Shared helpers for the tile inspection notebook: losses, the U-Net base class,
provided training loops, batched evaluation and receptive-field tools.

Segmentation is 3-class (see data.py): 0 background, 1 usable, 2 damaged.
Models output logits of shape (B, 3, H, W); a pixel's class is the argmax.
"""

import copy
import os
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from week3.project1.data import BACKGROUND, DAMAGED, N_CLASSES, USABLE, USABLE_THRESHOLD, usable_fraction
import week3._renamed  # noqa: F401,E402  (checkpoints saved as week2.* still load)

CLASS_CMAP = ListedColormap(["#3b3b38", "#e8e8e4", "#d03b3b"])   # background, usable, damaged


# ── Basics ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_loss(logits, target, eps=1e-6):
    """
    Multi-class Dice loss, averaged over the classes.
    logits: (B, C, H, W) raw scores
    target: (B, H, W) class indices
    """
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def predict_classes(logits: torch.Tensor) -> torch.Tensor:
    """Class map (B, H, W) from logits (B, C, H, W)."""
    return logits.argmax(dim=1)


def usable_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Predicted usable share of the tile, shape (B,)."""
    return usable_fraction(predict_classes(logits))


def decision(frac) -> str:
    return "APPROVE" if frac >= USABLE_THRESHOLD else "REJECT"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ── Checkpoints ("resume here") ──────────────────────────────────────────────

_CHECKPOINT_DIR = "cache/checkpoints"


def set_checkpoint_dir(path):
    """
    Where save_model / load_model keep models (use Google Drive to survive Colab restarts).
    Models that were saved to the previous location — e.g. because this was called
    after training had started — are copied over, so nothing gets lost.
    """
    import shutil
    global _CHECKPOINT_DIR
    old, _CHECKPOINT_DIR = _CHECKPOINT_DIR, path
    os.makedirs(path, exist_ok=True)
    if os.path.isdir(old) and os.path.abspath(old) != os.path.abspath(path):
        for f in sorted(os.listdir(old)):
            if f.endswith(".pt") and not os.path.exists(os.path.join(path, f)):
                shutil.copy2(os.path.join(old, f), os.path.join(path, f))
                print(f"📦 moved '{f[:-3]}' (saved before this cell ran) to {path}")


def _checkpoint_path(name):
    return os.path.join(_CHECKPOINT_DIR, f"{name}.pt")


def has_checkpoint(name):
    return os.path.exists(_checkpoint_path(name))


def save_model(model, name):
    """
    Save the WHOLE model — weights and its class code — under `name`.

    Uses dill, so classes you defined in the notebook are stored too: after a
    restart, load_model(name) works without re-running the class cells.
    """
    import dill
    dill.settings["recurse"] = True      # store only the globals the class code uses
    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
    torch.save(model, _checkpoint_path(name), pickle_module=dill)
    size = os.path.getsize(_checkpoint_path(name)) / 1e6
    print(f"💾 saved '{name}' ({size:.1f} MB) → {_checkpoint_path(name)}")


def load_model(name):
    """Load a model stored with save_model, on the current device, in eval mode."""
    import dill
    path = _checkpoint_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model '{name}' in {_CHECKPOINT_DIR}. "
                                f"Run the section that trains it first (it saves the model at the end).")
    model = torch.load(path, map_location=get_device(), pickle_module=dill, weights_only=False)
    model.eval()
    print(f"📂 loaded '{name}' from {path}")
    return model


def class_from_checkpoint(name, class_name=None):
    """
    Recover a class you defined in the notebook from a saved model, e.g. after a
    restart:  ConvBlock = class_from_checkpoint("improved_unet", "ConvBlock").
    Without class_name, returns the class of the saved model itself.
    """
    model = load_model(name)
    if class_name is None:
        return type(model)
    for m in model.modules():
        if type(m).__name__ == class_name:
            return type(m)
    raise LookupError(f"'{name}' contains no module of class {class_name} — re-run the cell that defines it.")


def export_for_platform(model, path, name="Our model", result=None):
    """
    Export a segmentation model for the inspection platform as TorchScript.

    TorchScript stores the compiled network, so the file loads on any computer
    with PyTorch — whatever Python version, and without this notebook's class
    code (a save_model checkpoint only loads on the Python version that saved it).
    A small .json next to it records the name and test metrics for the platform.
    """
    import copy as _copy, json, warnings
    m = _copy.deepcopy(model).cpu().float().eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(m, torch.zeros(1, 1, 256, 256), check_trace=False)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    traced.save(path)
    meta = {"name": name, "version": time.strftime("%Y-%m-%d"), "architecture": type(model).__name__}
    if result is not None:
        meta["metrics"] = {"test_accuracy_pct": round(result.acc["Threshold"], 2), "mean_iou": round(result.iou, 4),
                           "damaged_iou": round(result.iou_classes["damaged"], 4),
                           "usable_share_error_pp": round(result.frac_mae, 3)}
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"🏭 exported {type(model).__name__} for the platform → {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def load_models(*names):
    """load_model for several names at once: unet, fnn = load_models("simple_unet", "simple_fnn")"""
    models = tuple(load_model(n) for n in names)
    return models[0] if len(models) == 1 else models


# ── U-Net base class ─────────────────────────────────────────────────────────

class LatentUNetBase(nn.Module):
    """
    Base class for every segmentation backbone in this notebook.

    Subclasses implement two methods:
        encode(x)          -> (bottleneck, skips)
                              bottleneck: (B, latent_dim, H', W')
                              skips:      list of encoder feature maps (may be empty)
        decode(b, skips)   -> logits (B, n_classes, H, W)  — 3 classes here

    and set `self.latent_dim` (number of bottleneck channels).

    The base class then provides everything the classifier heads need:
        forward(x)         -> segmentation logits (B, 3, H, W)
        get_latent_map(x)  -> spatial bottleneck map   (B, C, H', W')
        get_latent(x)      -> pooled latent vector     (B, C)   via Global Average Pooling
        forward_all(x)     -> (logits, bottleneck map) in ONE encoder pass
    """

    latent_dim: int

    def encode(self, x):
        raise NotImplementedError("Implement encode(x) -> (bottleneck, skips)")

    def decode(self, b, skips):
        raise NotImplementedError("Implement decode(bottleneck, skips) -> logits")

    def forward(self, x):
        b, skips = self.encode(x)
        return self.decode(b, skips)

    def forward_all(self, x):
        b, skips = self.encode(x)
        return self.decode(b, skips), b

    def get_latent_map(self, x):
        return self.encode(x)[0]

    def get_latent(self, x):
        return self.get_latent_map(x).mean(dim=(2, 3))


class DepthUNet(LatentUNetBase):
    """
    Minimal U-Net whose only knob is its DEPTH (number of 2x downsamplings),
    used for the receptive-field experiment. Each block is a single
    Conv3x3 → BatchNorm → ReLU; channels double at every level (capped at 256)
    and skip connections join encoder and decoder at equal resolution.
    Deliberately lightweight — your ImprovedUNet should do better.
    """

    def __init__(self, depth=2, in_ch=1, base_ch=16, n_classes=N_CLASSES):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        chs = [min(base_ch * 2 ** i, 256) for i in range(depth + 1)]
        self.depth = depth
        self.encoders = nn.ModuleList(block(i, o) for i, o in zip([in_ch] + chs[:depth - 1], chs[:depth]))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(chs[depth - 1], chs[depth])
        self.ups = nn.ModuleList(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth)))
        self.decoders = nn.ModuleList(block(2 * chs[i], chs[i]) for i in reversed(range(depth)))
        self.out_conv = nn.Conv2d(chs[0], n_classes, 1)
        self.latent_dim = chs[depth]

    def encode(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        return self.bottleneck(x), skips

    def decode(self, b, skips):
        x = b
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.out_conv(x)


# ── Provided training loops ──────────────────────────────────────────────────

def _loader(ds, batch_size, shuffle, device):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=(device.type == "cuda"), num_workers=0)


def _amp(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def _seg_loss(logits, target, use_dice):
    loss = F.cross_entropy(logits, target)
    if use_dice:
        loss = loss + dice_loss(logits, target)
    return loss


def _iou_counts(pred, target):
    """Per-class intersection and union counts, shape (N_CLASSES,) each."""
    inter = torch.stack([((pred == c) & (target == c)).sum() for c in range(N_CLASSES)]).float()
    union = torch.stack([((pred == c) | (target == c)).sum() for c in range(N_CLASSES)]).float()
    return inter, union


def train_segmentation(model, ds_train, ds_val, epochs=30, lr=1e-3, batch_size=32,
                       use_dice=True, weight_decay=1e-4, scheduler="cosine",
                       amp=True, dashboard=None):
    """
    Train a 3-class segmentation backbone with validation tracking.

    Loss: cross-entropy (+ multi-class Dice if use_dice). Every epoch it records
    the train/val loss, the validation mean IoU over the three classes, and the
    validation accuracy of the usable-area threshold rule. Pass a
    widgets.TrainingDashboard as `dashboard` to watch it live.

    Returns: history dict of per-epoch lists.
    """
    device = get_device()
    model.to(device)
    dl_train = _loader(ds_train, batch_size, True, device)
    dl_val = _loader(ds_val, batch_size, False, device)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if scheduler == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_acc": []}
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        total = 0.0
        for img, target, _ in dl_train:
            img, target = img.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with _amp(device, amp):
                logits = model(img)
            loss = _seg_loss(logits.float(), target, use_dice)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        if sched:
            sched.step()

        val = evaluate_segmentation(model, dl_val, use_dice=use_dice, amp=amp)
        history["train_loss"].append(total / len(dl_train))
        history["val_loss"].append(val["loss"])
        history["val_iou"].append(val["iou"])
        history["val_acc"].append(val["acc"])
        _report(dashboard, "U-Net", ep, epochs, history, time.time() - t0)
    return history


@torch.no_grad()
def evaluate_segmentation(model, dl, use_dice=True, amp=True):
    """Validation loss, mean IoU over the 3 classes and threshold-rule accuracy over a DataLoader."""
    device = next(model.parameters()).device
    model.eval()
    total, correct, n = 0.0, 0, 0
    inter = torch.zeros(N_CLASSES)
    union = torch.zeros(N_CLASSES)
    for img, target, label in dl:
        img, target = img.to(device), target.to(device)
        with _amp(device, amp):
            logits = model(img)
        logits = logits.float()
        total += _seg_loss(logits, target, use_dice).item()
        i, u = _iou_counts(predict_classes(logits), target)
        inter += i.cpu()
        union += u.cpu()
        approve = usable_from_logits(logits) >= USABLE_THRESHOLD
        correct += (approve.cpu().long() == label).sum().item()
        n += len(label)
    iou = (inter / union.clamp(min=1)).mean().item()
    return {"loss": total / len(dl), "iou": iou, "acc": 100 * correct / n}


def train_head(backbone, head, ds_train, ds_val, mode="cnn", epochs=30, lr=1e-3,
               batch_size=32, weight_decay=1e-4, amp=True, dashboard=None):
    """
    Train a classifier / regressor head on top of a FROZEN backbone.

    mode = "fnn": head(get_latent(x))     -> approval probability (B,)   BCE loss
    mode = "cnn": head(get_latent_map(x)) -> approval probability (B,)   BCE loss
    mode = "reg": head(get_latent_map(x)) -> predicted usable share (B,) L1 loss
                  (the usable-area rule is then applied to the prediction)

    Returns: history dict with train_loss, val_loss, val_acc (and val_mae for "reg").
    """
    device = get_device()
    backbone.to(device).eval()
    head.to(device)
    dl_train = _loader(ds_train, batch_size, True, device)
    dl_val = _loader(ds_val, batch_size, False, device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    if mode == "reg":
        history["val_mae"] = []

    def features(img):
        with torch.no_grad(), _amp(device, amp):
            z = backbone.get_latent(img) if mode == "fnn" else backbone.get_latent_map(img)
        return z.float()

    def loss_fn(out, target, label):
        if mode == "reg":
            return F.l1_loss(out, usable_fraction(target))
        return F.binary_cross_entropy(out.clamp(1e-6, 1 - 1e-6), label.float())

    for ep in range(epochs):
        t0 = time.time()
        head.train()
        total = 0.0
        for img, target, label in dl_train:
            img, target, label = img.to(device), target.to(device), label.to(device)
            out = head(features(img))
            loss = loss_fn(out, target, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()

        head.eval()
        vloss, correct, n, abs_err = 0.0, 0, 0, 0.0
        with torch.no_grad():
            for img, target, label in dl_val:
                img, target, label = img.to(device), target.to(device), label.to(device)
                out = head(features(img))
                vloss += loss_fn(out, target, label).item()
                if mode == "reg":
                    pred = out >= USABLE_THRESHOLD
                    abs_err += (out - usable_fraction(target)).abs().sum().item()
                else:
                    pred = out > 0.5
                correct += (pred.long() == label).sum().item()
                n += len(label)
        history["train_loss"].append(total / len(dl_train))
        history["val_loss"].append(vloss / len(dl_val))
        history["val_acc"].append(100 * correct / n)
        if mode == "reg":
            history["val_mae"].append(100 * abs_err / n)
        _report(dashboard, mode.upper(), ep, epochs, history, time.time() - t0)
    return history


def _report(dashboard, name, ep, epochs, history, secs):
    if dashboard is not None:
        dashboard.update(ep + 1, epochs, secs=secs, **{k: v[-1] for k, v in history.items()})
    elif ep % 5 == 0 or ep == epochs - 1:
        parts = " | ".join(f"{k}: {v[-1]:.4f}" for k, v in history.items())
        print(f"[{name}] Epoch {ep + 1:3d}/{epochs} | {parts}")


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Everything the comparison widgets need about one set of models."""
    label: str
    df: pd.DataFrame                 # one row per test image
    acc: dict                        # method -> accuracy in %
    iou: float                       # mean IoU over the 3 classes (pooled over all test pixels)
    iou_classes: dict                # class name -> IoU
    frac_mae: float                  # mean |predicted - true usable share| in %-points
    cache: object = field(repr=False)
    pred_masks: np.ndarray = field(repr=False, default=None)   # (N, H, W) uint8 predicted class maps

    @property
    def methods(self):
        return list(self.acc)

    def __iter__(self):              # allows:  df, accuracies = evaluate_models(...)
        yield self.df
        yield list(self.acc.values())


@torch.no_grad()
def evaluate_models(unet, classifier_fnn=None, classifier_cnn=None, ds_test=None, *,
                    regressor=None, label="MODELS", batch_size=64, verbose=True):
    """
    Evaluate the threshold rule and every head that is given on the test set.

    Threshold:  usable share of the predicted mask >= threshold   (uses only the U-Net)
    FNN / CNN:  classifier probability > 0.5
    Regression: predicted usable share >= threshold

    Runs in batches with a single encoder pass per image.
    Returns an EvalResult (unpacks as `df, accuracies` for older cells).
    """
    device = next(unet.parameters()).device
    for m in (unet, classifier_fnn, classifier_cnn, regressor):
        if m is not None:
            m.to(device).eval()
    cache = ds_test.cache
    dl = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    rows = {k: [] for k in ("true_frac", "pred_frac", "fnn_prob", "cnn_prob", "reg_frac", "label", "iou_usable", "iou_damaged")}
    preds = []
    inter_total = torch.zeros(N_CLASSES)
    union_total = torch.zeros(N_CLASSES)
    for img, target, lab in dl:
        img, target = img.to(device), target.to(device)
        logits, b = unet.forward_all(img)
        pred = predict_classes(logits)
        i, u = _iou_counts(pred, target)
        inter_total += i.cpu()
        union_total += u.cpu()
        for c, key in ((USABLE, "iou_usable"), (DAMAGED, "iou_damaged")):
            inter = ((pred == c) & (target == c)).sum(dim=(1, 2)).float()
            union = ((pred == c) | (target == c)).sum(dim=(1, 2)).float()
            rows[key] += torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).tolist()
        rows["pred_frac"] += usable_fraction(pred).tolist()
        rows["true_frac"] += usable_fraction(target).tolist()
        rows["label"] += lab.tolist()
        nan = [float("nan")] * len(lab)
        rows["fnn_prob"] += classifier_fnn(b.mean(dim=(2, 3))).tolist() if classifier_fnn else nan
        rows["cnn_prob"] += classifier_cnn(b).tolist() if classifier_cnn else nan
        rows["reg_frac"] += regressor(b).tolist() if regressor else nan
        preds.append(pred.to(torch.uint8).cpu().numpy())

    df = pd.DataFrame(rows)
    df.insert(0, "index", ds_test.indices)
    df.insert(1, "filename", [cache.files[i] for i in ds_test.indices])
    df["Actual"] = np.where(df["label"] == 1, "APPROVE", "REJECT")
    to_verdict = lambda ok: np.where(ok, "APPROVE", "REJECT")
    df["Threshold"] = to_verdict(df["pred_frac"] >= USABLE_THRESHOLD)
    methods = ["Threshold"]
    if classifier_fnn:
        df["FNN"] = to_verdict(df["fnn_prob"] > 0.5); methods.append("FNN")
    if classifier_cnn:
        df["CNN"] = to_verdict(df["cnn_prob"] > 0.5); methods.append("CNN")
    if regressor:
        df["Regression"] = to_verdict(df["reg_frac"] >= USABLE_THRESHOLD); methods.append("Regression")

    acc = {m: 100 * (df[m] == df["Actual"]).mean() for m in methods}
    per_class = (inter_total / union_total.clamp(min=1)).tolist()
    res = EvalResult(label=label, df=df, acc=acc, iou=float(np.mean(per_class)),
                     iou_classes={"background": per_class[BACKGROUND], "usable": per_class[USABLE],
                                  "damaged": per_class[DAMAGED]},
                     frac_mae=float(100 * (df["pred_frac"] - df["true_frac"]).abs().mean()),
                     cache=cache, pred_masks=np.concatenate(preds))
    if verbose:
        print(f"{label} — {len(df)} test tiles | mean IoU {res.iou:.3f} "
              f"(usable {res.iou_classes['usable']:.3f}, damaged {res.iou_classes['damaged']:.3f}) | "
              f"usable-share error {res.frac_mae:.2f} pp | " + " | ".join(f"{m} {a:.1f}%" for m, a in acc.items()))
    return res


def predictions_table(res: EvalResult, n=20):
    """Compact per-tile table for display."""
    cols = ["filename", "true_frac", "pred_frac", "Actual"] + res.methods
    out = res.df[cols].head(n).copy()
    out["true_frac"] = (100 * out["true_frac"]).map("{:.1f}%".format)
    out["pred_frac"] = (100 * out["pred_frac"]).map("{:.1f}%".format)
    return out.rename(columns={"true_frac": "True usable", "pred_frac": "U-Net usable"}).set_index("filename")


# ── Receptive field ──────────────────────────────────────────────────────────

def _center_grad(model, target, size, x=None):
    """|d(center unit)/d(input)| for the bottleneck or output of a U-Net."""
    device = next(model.parameters()).device
    if x is None:
        x = torch.ones(1, 1, size, size, device=device)
    x = x.clone().requires_grad_(True)
    out = model.get_latent_map(x) if target == "bottleneck" else model(x)
    h, w = out.shape[-2:]
    out[..., h // 2, w // 2].sum().backward()
    return x.grad.detach().abs().sum(dim=(0, 1)).cpu().numpy()


def theoretical_receptive_field(model, target="bottleneck", size=256):
    """
    Side length (pixels) of the input region that can influence the centre
    unit. Computed by back-propagating through a copy of the network whose
    weights are all positive, so no gradient cancels out and every pixel that
    is connected to the centre unit gets a non-zero gradient.
    """
    probe = copy.deepcopy(model).cpu().float().eval()
    # max-pooling routes the gradient to a single pixel; average pooling
    # connects the same input window to the output, but to every pixel in it
    for parent in list(probe.modules()):
        for name, child in parent.named_children():
            if isinstance(child, nn.MaxPool2d):
                setattr(parent, name, nn.AvgPool2d(child.kernel_size, child.stride, child.padding))
    for m in probe.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            # 1/fan-in keeps activations ~1 however deep the network is
            nn.init.constant_(m.weight, 1.0 / m.weight[0].numel())
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d,)):
            m.reset_parameters(); m.reset_running_stats()
    g = _center_grad(probe, target, size)
    ys, xs = np.nonzero(g > 0)
    return int(max(np.ptp(ys), np.ptp(xs)) + 1)


def effective_receptive_field(model, images, target="bottleneck", mass=0.9):
    """
    Side length of the centred square that holds `mass` of the input-gradient
    magnitude of the centre unit, averaged over real images. This is the part
    of the theoretical field the trained network actually pays attention to.
    Returns (side_px, gradient_map).
    """
    model.eval()
    device = next(model.parameters()).device
    g = sum(_center_grad(model, target, None, img.unsqueeze(0).to(device)) for img in images)
    g = g / g.sum()
    c = np.array(g.shape) // 2
    for r in range(0, g.shape[0] // 2 + 1):
        if g[c[0] - r:c[0] + r + 1, c[1] - r:c[1] + r + 1].sum() >= mass:
            return 2 * r + 1, g
    return g.shape[0], g


def defect_sizes(cache, every=5):
    """Side length (px, at model resolution) of each connected damaged zone, via connected components."""
    from scipy.ndimage import find_objects, label
    sizes = []
    for i in range(0, len(cache), every):
        lab, _ = label(cache.target(i) == DAMAGED)
        for sl in find_objects(lab):
            sizes.append(max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start))
    return np.array(sizes)


def tile_sizes(cache, every=5):
    """Longer side (px, at model resolution) of the tile in each image."""
    sizes = []
    for i in range(0, len(cache), every):
        ys, xs = np.nonzero(cache.target(i) != BACKGROUND)
        if len(ys):
            sizes.append(max(np.ptp(ys), np.ptp(xs)) + 1)
    return np.array(sizes)


# ── Static matplotlib fallbacks ──────────────────────────────────────────────

def plot_segmentation_results(unet, ds_test, n=8, title="U-Net predictions on the TEST SET"):
    """3-row grid: photo | true class map | predicted class map for the first n test tiles."""
    unet.eval()
    device = next(unet.parameters()).device
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7))
    for col in range(n):
        img, target, _ = ds_test[col]
        with torch.no_grad():
            pred = predict_classes(unet(img.unsqueeze(0).to(device)))[0].cpu()
        idx = ds_test.indices[col]
        axes[0, col].imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(ds_test.cache.files[idx], fontsize=8)
        for row, (cm, name) in enumerate([(target, f"truth {100 * usable_fraction(target):.0f}% usable"),
                                          (pred, f"pred {100 * usable_fraction(pred):.0f}% usable")], start=1):
            axes[row, col].imshow(cm, cmap=CLASS_CMAP, vmin=0, vmax=N_CLASSES - 1, interpolation="nearest")
            axes[row, col].set_title(name, fontsize=8)
        for row in range(3):
            axes[row, col].axis("off")
    plt.suptitle(f"{title}   (grey = belt · white = usable · red = damaged)", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_history(history, title="Training"):
    """Loss curves + (if present) validation accuracy, one chart each."""
    has_acc = "val_acc" in history
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(13 if has_acc else 7, 4))
    axes = np.atleast_1d(axes)
    axes[0].plot(history["train_loss"], label="Train loss", color="#2a78d6", lw=2)
    axes[0].plot(history["val_loss"], label="Val loss", color="#eb6834", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_title(f"{title} — loss"); axes[0].legend()
    if has_acc:
        axes[1].plot(history["val_acc"], color="#2a78d6", lw=2)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("%"); axes[1].set_title(f"{title} — val accuracy")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
