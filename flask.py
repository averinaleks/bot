from __future__ import annotations

import importlib.util
import os
import sys


def _load_real_flask():
    for path in sys.path[1:]:
        candidate = os.path.join(path, "flask", "__init__.py")
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("flask", candidate)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[__name__] = module
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                return module
    return None


_real = _load_real_flask()
if _real is not None:
    sys.modules[__name__] = _real
    globals().update(_real.__dict__)
else:
    from flask_stub import *  # type: ignore
