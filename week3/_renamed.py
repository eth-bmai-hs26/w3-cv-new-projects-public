"""
The course folder used to be called `week2/`. Checkpoints written with dill
before the rename refer to their classes as `week2.project1.utils.LatentUNetBase`
(or `week2.project2...`), so unpickling them imports `week2.*`.

Importing this module (the project helpers do) makes every `week2.<x>` import
resolve to the already-renamed `week3.<x>` module, so those files still load.
"""

import importlib
import importlib.abc
import importlib.util
import sys

OLD, NEW = "week2", "week3"


class _OldPackageName(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == OLD or fullname.startswith(OLD + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(NEW + spec.name[len(OLD):])   # the real module, under its old name too

    def exec_module(self, module):
        pass                                                        # already executed as week3.<x>


# First in line: otherwise the normal path finder would find week3/project1/utils.py
# through the aliased parent package and load a second, separate copy of it.
if not any(isinstance(f, _OldPackageName) for f in sys.meta_path):
    sys.meta_path.insert(0, _OldPackageName())
