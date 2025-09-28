"""Utility loader that tolerates lightweight test stubs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

_UTILS_CACHE: ModuleType | None = None
_ALLOWED_MODULE_NAMES = ("utils", "bot.utils")


def _load_from_source() -> ModuleType:
    """Load the real ``utils`` module directly from its source file."""

    global _UTILS_CACHE
    if _UTILS_CACHE is not None:
        return _UTILS_CACHE

    project_root = Path(__file__).resolve().parent.parent
    utils_path = project_root / "utils.py"
    spec = importlib.util.spec_from_file_location("_bot_real_utils", utils_path)
    if spec is None:
        raise ImportError(f"Unable to load utils module from {utils_path!s}")

    loader = spec.loader
    if loader is None:
        raise ImportError(f"Unable to load utils module from {utils_path!s}")

    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)  # type: ignore[arg-type]
    sys.modules.setdefault("_bot_real_utils", module)
    _UTILS_CACHE = module
    return module


def _get_candidate(name: str) -> ModuleType | None:
    """Return the utils module referenced by ``name`` if it is allowed."""

    module = sys.modules.get(name)
    if module is not None:
        return module

    try:
        if name == "utils":
            import utils as utils_module  # type: ignore[import-not-found]

            module = utils_module
        elif name == "bot.utils":
            from bot import utils as bot_utils  # type: ignore[import-not-found]

            module = bot_utils
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Refusing to import unexpected utils module {name!r}")
    except Exception:
        return None

    sys.modules.setdefault(name, module)
    return module


def require_utils(*required_names: str) -> ModuleType:
    """Return a utils-like module containing the requested attributes."""

    names: Iterable[str] = required_names or ()

    for candidate_name in _ALLOWED_MODULE_NAMES:
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
