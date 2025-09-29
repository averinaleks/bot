"""Compatibility shim exporting the top-level ``utils`` module."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from .utils_loader import require_utils


def _load_utils() -> ModuleType:
    try:
        module = importlib.import_module("utils")
    except ModuleNotFoundError:
        module = require_utils("configure_logging", "logger", "retry", "suppress_tf_logs")
    else:
        if not all(
            hasattr(module, name)
            for name in ("configure_logging", "logger", "retry", "suppress_tf_logs")
        ):
            module = require_utils("configure_logging", "logger", "retry", "suppress_tf_logs")

    if module.__name__ != "utils":
        proxy = ModuleType("utils")
        proxy.__dict__.update(module.__dict__)
        proxy.__name__ = "utils"
        module = proxy
    module.__spec__ = importlib.machinery.ModuleSpec("utils", loader=None)

    sys.modules["utils"] = module
    return module


module = _load_utils()
sys.modules[__name__] = module

