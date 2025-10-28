"""Utilities for loading ``resolve_github_path`` with graceful fallbacks."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib.abc import Loader
from pathlib import Path
from typing import Protocol

from scripts.github_path_resolver_fallback import (
    resolve_github_path as _fallback_resolve_github_path,
)


class _Resolver(Protocol):
    def __call__(
        self,
        raw_path: str | None,
        *,
        allow_missing: bool = False,
        description: str = "GitHub provided path",
    ) -> Path | None:
        ...


def _load_resolver() -> _Resolver:
    existing = sys.modules.get("scripts.github_paths")
    if existing is not None:
        candidate = getattr(existing, "resolve_github_path", None)
        if callable(candidate):
            return candidate

    spec = importlib.util.find_spec("scripts.github_paths")
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            loader = spec.loader
            if isinstance(loader, Loader):
                loader.exec_module(module)
            else:
                raise TypeError("spec.loader does not implement exec_module")
        except Exception as exc:
            sys.modules.pop(spec.name, None)
            print(
                f"::notice::Failed to load scripts.github_paths: {exc}. Using fallback.",
                file=sys.stderr,
            )
        else:
            candidate = getattr(module, "resolve_github_path", None)
            if callable(candidate):
                return candidate

    return _fallback_resolve_github_path


_RESOLVER: _Resolver = _load_resolver()


def resolve_github_path(
    raw_path: str | None,
    *,
    allow_missing: bool = False,
    description: str = "GitHub provided path",
) -> Path | None:
    global _RESOLVER

    if _RESOLVER is _fallback_resolve_github_path:
        candidate = _load_resolver()
        if candidate is not _fallback_resolve_github_path:
            _RESOLVER = candidate

    return _RESOLVER(
        raw_path,
        allow_missing=allow_missing,
        description=description,
    )


__all__ = ["resolve_github_path"]

