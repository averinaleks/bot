"""Utility loader that tolerates lightweight test stubs."""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

_UTILS_CACHE: ModuleType | None = None
_ALLOWED_MODULE_NAMES = ("utils", "bot.utils")


def _project_root() -> Path:
    """Return the repository root used for resolving ``utils.py``."""

    return Path(__file__).resolve().parent.parent


def _resolve_utils_path(base: Path | None = None) -> Path:
    """Return a secure absolute path to ``utils.py`` within *base*.

    The helper verifies that the target exists, is a regular file, and resides
    inside the expected repository directory.  Symlinks are explicitly rejected
    to avoid time-of-check/time-of-use attacks where an attacker could swap in a
    different module between checks.  ``ImportError`` is raised whenever the
    invariants cannot be satisfied so callers can surface a clear failure
    message to operators.
    """

    root = base if base is not None else _project_root()
    try:
        resolved_root = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ImportError(
            f"Project root {root} is unavailable"
        ) from exc
    except OSError as exc:
        raise ImportError(f"Unable to resolve project root {root}: {exc}") from exc

    candidate = root / "utils.py"
    try:
        if candidate.is_symlink():
            raise ImportError("Refusing to load utils.py through a symlink")
    except OSError as exc:
        raise ImportError(f"Unable to inspect utils.py at {candidate}: {exc}") from exc

    try:
        resolved_candidate = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ImportError(f"utils.py not found at {candidate}") from exc
    except OSError as exc:
        raise ImportError(f"Unable to resolve utils.py at {candidate}: {exc}") from exc

    try:
        stat_info = resolved_candidate.stat()
    except OSError as exc:
        raise ImportError(f"Unable to stat utils.py at {resolved_candidate}: {exc}") from exc

    if not stat.S_ISREG(stat_info.st_mode):
        raise ImportError("utils.py must be a regular file")

    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ImportError(
            f"Resolved utils.py {resolved_candidate} escapes project root {resolved_root}"
        ) from exc

    return resolved_candidate


def _load_from_source() -> ModuleType:
    """Load the real ``utils`` module directly from its source file."""

    global _UTILS_CACHE
    if _UTILS_CACHE is not None:
        return _UTILS_CACHE

    utils_path = _resolve_utils_path()
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
