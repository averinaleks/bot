"""Utilities for loading ``resolve_github_path`` with graceful fallbacks."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable


def _deduplicate(bases: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        unique.append(base)
    return unique


def _fallback_allowed_directories() -> list[Path]:
    allowed: list[Path] = []

    workspace_env = os.getenv("GITHUB_WORKSPACE")
    if workspace_env:
        try:
            workspace = Path(workspace_env).resolve(strict=True)
        except OSError as exc:
            print(
                f"::warning::Invalid GITHUB_WORKSPACE '{workspace_env}': {exc}",
                file=sys.stderr,
            )
        else:
            allowed.append(workspace)
            parent = workspace.parent
            allowed.append(parent)
            grandparent = parent.parent
            if grandparent != parent:
                allowed.append(grandparent)

    runner_temp = os.getenv("RUNNER_TEMP")
    if runner_temp:
        try:
            allowed.append(Path(runner_temp).resolve(strict=True))
        except OSError:
            pass

    try:
        allowed.append(Path.cwd().resolve(strict=True))
    except OSError as exc:
        print(
            f"::warning::Unable to resolve current working directory: {exc}",
            file=sys.stderr,
        )

    try:
        allowed.append(Path(tempfile.gettempdir()).resolve(strict=False))
    except (FileNotFoundError, RuntimeError) as exc:
        print(
            f"::warning::Unable to resolve temporary directory: {exc}",
            file=sys.stderr,
        )

    return _deduplicate(allowed)


def _fallback_resolve_github_path(
    raw_path: str | None,
    *,
    allow_missing: bool = False,
    description: str = "GitHub provided path",
) -> Path | None:
    if not raw_path:
        return None

    try:
        candidate = Path(raw_path).resolve(strict=not allow_missing)
    except OSError as exc:
        print(
            f"::warning::Unable to resolve {description} '{raw_path}': {exc}",
            file=sys.stderr,
        )
        return None

    for base in _fallback_allowed_directories():
        try:
            candidate.relative_to(base)
        except ValueError:
            continue
        return candidate

    print(
        f"::warning::Ignoring {description} outside trusted directories",
        file=sys.stderr,
    )
    return None


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

