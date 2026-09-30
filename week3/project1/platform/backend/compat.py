"""
Make student checkpoints loadable outside the notebook.

`save_model` pickles the WHOLE model with dill. Classes the student defined in
the notebook travel inside the file, but the base class `LatentUNetBase` is
pickled *by reference* to `week3.project1.utils` (`week2.project1.utils` in
files saved before the course folder was renamed), so that module has to be
importable when the platform loads the file.

1. Running from the course repo: the repo root is put on sys.path and the real
   module is used.
2. Platform copied somewhere else: a minimal stand-in module with the same
   name and the same `LatentUNetBase` is registered instead.
"""

from __future__ import annotations

import sys
import types

import torch.nn as nn

from .config import REPO_ROOT


class LatentUNetBase(nn.Module):
    """Stand-in with the same interface as week3.project1.utils.LatentUNetBase."""

    latent_dim: int

    def encode(self, x):
        raise NotImplementedError

    def decode(self, b, skips):
        raise NotImplementedError

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


_done = False
repo_import_error: str | None = None      # why the real course code was not used (shown when a file needs it)


def ensure_course_modules() -> str:
    """Returns 'repo' if the real course code is used, 'stub' if the stand-in was registered."""
    global _done, repo_import_error
    if (REPO_ROOT / "week3" / "project1" / "utils.py").exists() and str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    try:
        import week3.project1.utils  # noqa: F401
        _done = True
        return "repo"
    except Exception as e:
        if repo_import_error is None and (REPO_ROOT / "week3" / "project1" / "utils.py").exists():
            repo_import_error = f"{type(e).__name__}: {e}"
    if not _done:
        utils = types.ModuleType("week3.project1.utils")
        utils.LatentUNetBase = LatentUNetBase
        for top in ("week3", "week2"):                  # week2: files saved before the folder was renamed
            for name in (top, f"{top}.project1"):
                if name not in sys.modules:
                    mod = types.ModuleType(name)
                    mod.__path__ = []
                    sys.modules[name] = mod
            sys.modules[f"{top}.project1.utils"] = utils
            sys.modules[f"{top}.project1"].utils = utils
        _done = True
    return "stub"
