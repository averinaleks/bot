"""Fallback helpers for resolving GitHub-provided filesystem paths.

These helpers intentionally avoid any optional dependencies so they can be
imported even when the rest of the helper package is only partially
available.  They mirror the logic that ships with
``scripts.github_path_resolver`` but live in a standalone module so other
scripts can gracefully degrade if the primary resolver cannot be imported.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable


def _deduplicate(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _allowed_directories() -> list[Path]:
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


def resolve_github_path(
    raw_path: str | None,
    *,
    allow_missing: bool = False,
    description: str = "GitHub provided path",
) -> Path | None:
    """Best effort resolver used when the primary helper is unavailable."""

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

    for base in _allowed_directories():
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


__all__ = ["resolve_github_path"]

