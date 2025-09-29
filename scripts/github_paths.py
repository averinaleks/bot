"""Helpers for validating GitHub-provided filesystem paths."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _deduplicate(bases: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        unique.append(base)
    return unique


def allowed_github_directories() -> list[Path]:
    """Return directories considered safe for GitHub-provided file paths."""

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

    try:
        allowed.append(Path.cwd().resolve(strict=True))
    except OSError:
        pass

    try:
        allowed.append(Path(tempfile.gettempdir()).resolve(strict=False))
    except (FileNotFoundError, RuntimeError):
        pass

    return _deduplicate(allowed)


def resolve_github_path(
    raw_path: str | None,
    *,
    allow_missing: bool = False,
    description: str = "GitHub provided path",
) -> Path | None:
    """Validate *raw_path* and return a resolved :class:`Path` if safe."""

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

    for base in allowed_github_directories():
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


__all__ = ["allowed_github_directories", "resolve_github_path"]

