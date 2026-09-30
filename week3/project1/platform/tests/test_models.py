"""The model service loads the checkpoint formats students produce and reads 3-class and legacy outputs."""

import torch
import torch.nn as nn

from backend.model_service import ModelService


class ThreeClassNet(nn.Module):
    """Bright pixels -> usable, dark belt -> background, top-left quarter -> damaged."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        usable = x * self.scale - 2.5
        bg = torch.zeros_like(x)
        corner = torch.zeros_like(x)
        corner[..., :64, :64] = 1.0
        dmg = torch.where((corner > 0) & (usable > 0), usable + 1.0, torch.full_like(x, -5.0))
        return torch.cat([bg, usable, dmg], dim=1)


class OneChannelNet(nn.Module):
    def forward(self, x):
        return x * 10 - 2.5


NOTEBOOK_CELL = """
import sys, dill, torch, torch.nn as nn
sys.path.insert(0, {platform!r})
from backend.compat import ensure_course_modules
ensure_course_modules()
from week3.project1.utils import LatentUNetBase

class TinyUNet(LatentUNetBase):          # defined in the notebook, i.e. in __main__
    latent_dim = 4
    def __init__(self):
        super().__init__()
        self.enc = nn.Conv2d(1, 4, 3, padding=1)
        self.head = nn.Conv2d(4, 3, 1)
    def encode(self, x):
        return self.enc(x), []
    def decode(self, b, skips):
        return self.head(b)

m = TinyUNet()
m.platform_meta = {{"version": "v7", "metrics": {{"dice": 0.93}}}}
dill.settings["recurse"] = True
torch.save(m, {path!r}, pickle_module=dill)
"""


def _save_like_notebook(path):
    """Run save_model's logic in a separate interpreter so the class lives in __main__, as in Colab."""
    import subprocess
    import sys
    from pathlib import Path
    code = NOTEBOOK_CELL.format(platform=str(Path(__file__).resolve().parent.parent), path=str(path))
    subprocess.run([sys.executable, "-c", code], check=True)


def _photo():
    from PIL import Image
    img = Image.new("RGB", (256, 256), (40, 42, 41))
    img.paste((200, 190, 170), (32, 32, 224, 224))
    return img


def test_torchscript_three_class(tmp_path):
    p = tmp_path / "scripted.pt"
    torch.jit.script(ThreeClassNet()).save(str(p))
    svc = ModelService(str(p))
    assert svc.info["mode"] == "model" and svc.info["kind"] == "torchscript" and svc.info["classes"] == 3
    pred = svc.predict(_photo())
    assert pred.error is None and pred.warnings == []
    # tile = 192x192 square; damaged = its part inside the top-left 64x64 quarter = 32x32
    assert abs(pred.usable_fraction - (1 - 32 * 32 / (192 * 192))) < 0.02
    assert set(pred.classes.ravel().tolist()) <= {0, 1, 2}
    assert pred.timings["total"] > 0


def test_dill_whole_model_like_save_model(tmp_path):
    p = tmp_path / "platform_model.pt"
    _save_like_notebook(p)
    svc = ModelService(str(p))
    assert svc.info["kind"] == "dill" and svc.info["architecture"] == "TinyUNet"
    assert svc.info["version"] == "v7" and svc.info["metrics"]["dice"] == 0.93 and svc.info["classes"] == 3
    pred = svc.predict(_photo())                 # untrained weights: any valid 3-class answer
    assert set(pred.classes.ravel().tolist()) <= {0, 1, 2}


def test_sidecar_json_metrics(tmp_path):
    p = tmp_path / "m.pt"
    torch.jit.script(ThreeClassNet()).save(str(p))
    (tmp_path / "m.json").write_text('{"name": "improved_unet", "metrics": {"val_dice": 0.91}}')
    info = ModelService(str(p)).info
    assert info["name"] == "improved_unet" and info["metrics"] == {"val_dice": 0.91}


def test_legacy_one_channel_model_warns(tmp_path):
    p = tmp_path / "legacy.pt"
    torch.jit.script(OneChannelNet()).save(str(p))
    svc = ModelService(str(p))
    assert svc.info["classes"] == 1 and "legacy" in svc.info["warning"].lower()
    pred = svc.predict(_photo())
    assert any("legacy" in w.lower() for w in pred.warnings)
    assert pred.usable_fraction is not None


def test_state_dict_is_explained_not_crashing(tmp_path):
    p = tmp_path / "weights.pt"
    torch.save(ThreeClassNet().state_dict(), p)
    svc = ModelService(str(p))
    assert svc.demo and "state_dict" in svc.info["warning"]


def test_upload_model_via_api(client, tmp_path, monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "UPLOADED_MODELS_DIR", tmp_path / "uploads")
    p = tmp_path / "student.pt"
    torch.jit.script(ThreeClassNet()).save(str(p))
    with open(p, "rb") as f:
        r = client.post("/api/model/upload", files={"file": ("student.pt", f, "application/octet-stream")}, data={"actor": "tester"})
    assert r.status_code == 200 and r.json()["mode"] == "model"
    assert client.get("/api/settings").json()["model"]["checkpoint"].endswith("student.pt")
    assert client.post("/api/model/upload", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 422


COURSE_CLASS_CELL = """
import sys, dill, torch
sys.path.insert(0, {repo!r})
from week3.project1.utils import DepthUNet     # a class from the course code, pickled by reference
m = DepthUNet(2).eval()
dill.settings["recurse"] = True
torch.save(m, {path!r}, pickle_module=dill)
"""


def test_course_class_checkpoint_loads(tmp_path):
    """save_model files of course classes (DepthUNet, the ResNet) need the real course code and its packages."""
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[4]
    p = tmp_path / "depth_unet_2.pt"
    subprocess.run([sys.executable, "-c", COURSE_CLASS_CELL.format(repo=str(repo), path=str(p))], check=True)
    svc = ModelService(str(p))
    assert svc.info["mode"] == "model", svc.info.get("warning")
    assert svc.info["kind"] == "dill" and svc.info["classes"] == 3


def test_load_hints_explain_the_fix():
    from backend.model_service import _load_hint
    version = _load_hint(SystemError("no locals found when setting up annotations"))
    assert "different Python version" in version and "export_for_platform" in version
    missing = _load_hint(AttributeError("Can't get attribute 'DepthUNet' on <module 'week3.project1.utils'>"))
    assert "course" in missing and "export_for_platform" in missing
    assert _load_hint(ValueError("something else")) == ""


def test_stored_paths_follow_a_renamed_or_moved_folder(tmp_path):
    """A database written before week2/ became week3/ (or before the repo moved) still finds the model and photos."""
    from backend import config
    from backend.core import Platform
    plat = Platform(data_dir=tmp_path, load_model=False)
    old = "/somewhere/else/w3-repo/week2/project1"
    plat.db.set_setting("model", {"checkpoint": f"{old}/platform/models/default_model.pt", "device": "auto"})
    plat.db.set_setting("image_source", f"{old}/dataset_new/original")
    assert plat.model_settings()["checkpoint"] == str(config.DEFAULT_CHECKPOINT)
    assert plat.image_source() == str(config.PLATFORM_DIR.parent / "dataset_new" / "original")
    assert config.relocate("/nowhere/model.pt") == "/nowhere/model.pt"          # unrelated paths are left alone
    assert config.relocate(r"C:\Users\x\week2\project1\platform\models\default_model.pt") == str(config.DEFAULT_CHECKPOINT)
