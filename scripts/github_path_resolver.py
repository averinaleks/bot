"""Utilities for loading ``resolve_github_path`` with graceful fallbacks."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

from scripts.github_path_resolver_fallback import (
    resolve_github_path as _fallback_resolve_github_path,
)


def _load_resolver() -> Callable[[str | None], Path | None]:
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
            spec.loader.exec_module(module)  # type: ignore[arg-type]
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


_RESOLVER: Callable[[str | None], Path | None] = _load_resolver()


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

