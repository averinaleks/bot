"""Compatibility facade for :mod:`bot.trade_manager`.

This repository historically exposed a top-level :mod:`trade_manager`
module.  Several tests – and, by extension, downstream tooling – import that
module directly to verify import order hooks.  The project structure has since
been reorganised around :mod:`bot.trade_manager`, which means the legacy import
stopped working and the tests started failing with ``ModuleNotFoundError``.

To preserve backwards compatibility we re-export the public interface from the
current package.  The implementation is intentionally lightweight: it imports
``bot.trade_manager`` lazily and then mirrors the attributes that package makes
available via ``__all__``.  Any new public attribute added to
``bot.trade_manager`` will automatically propagate here without further
maintenance.
"""

from __future__ import annotations

import importlib
import sys
from importlib import import_module
from types import ModuleType
from typing import Any, List


def _load_target() -> ModuleType:
    """Import and return the canonical trade manager package."""

    existing = sys.modules.get("bot.trade_manager")
    # Avoid reloading this compatibility shim when it's mistakenly registered
    # as ``bot.trade_manager`` in ``sys.modules``.
    if existing is not None and existing is sys.modules.get(__name__):
        existing = None

    if existing is not None:
        return importlib.reload(existing)
    return import_module("bot.trade_manager")


_TARGET = _load_target()


def __getattr__(name: str) -> Any:  # pragma: no cover - thin forwarding logic
    return getattr(_TARGET, name)


def __dir__() -> List[str]:  # pragma: no cover - mirrors target metadata
    public = getattr(_TARGET, "__all__", None)
    if public:
        return sorted(set(public))
    return sorted(attr for attr in dir(_TARGET) if not attr.startswith("_"))


for _name in getattr(_TARGET, "__all__", []):
    globals()[_name] = getattr(_TARGET, _name)

