"""Utility loader that tolerates lightweight test stubs."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

_UTILS_CACHE: ModuleType | None = None


def _load_from_source() -> ModuleType:
    """Load the real ``utils`` module directly from its source file."""

    global _UTILS_CACHE
    if _UTILS_CACHE is not None:
        return _UTILS_CACHE

    project_root = Path(__file__).resolve().parent.parent
    utils_path = project_root / "utils.py"
    spec = importlib.util.spec_from_file_location("_bot_real_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load utils module from {utils_path!s}")

    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)  # type: ignore[arg-type]
    sys.modules.setdefault("_bot_real_utils", module)
    _UTILS_CACHE = module
    return module


def _get_candidate(name: str) -> ModuleType | None:
    module = sys.modules.get(name)
    if module is not None:
        return module
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def require_utils(*required_names: str) -> ModuleType:
    """Return a utils-like module containing the requested attributes."""

    names: Iterable[str] = required_names or ()

    for candidate_name in ("utils", "bot.utils"):
        candidate = _get_candidate(candidate_name)
        if candidate is None:
            continue
        if all(hasattr(candidate, attr) for attr in names):
            return candidate

    module = _load_from_source()
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ImportError(f"utils module missing required attributes: {joined}")
    return module


__all__ = ["require_utils"]
