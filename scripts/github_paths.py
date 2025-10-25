"""Helpers for validating GitHub-provided filesystem paths."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _has_control_characters(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


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

    runner_temp = os.getenv("RUNNER_TEMP")
    if runner_temp:
        try:
            allowed.append(Path(runner_temp).resolve(strict=True))
        except OSError:
            pass

    try:
        allowed.append(Path.cwd().resolve(strict=True))
    except OSError as exc:
        print(f"::warning::Unable to resolve current working directory: {exc}", file=sys.stderr)

    try:
        allowed.append(Path(tempfile.gettempdir()).resolve(strict=False))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"::warning::Unable to resolve temporary directory: {exc}", file=sys.stderr)

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
        decoded = os.fsdecode(raw_path)
    except (TypeError, ValueError) as exc:
        print(
            f"::warning::Unable to decode {description} '{raw_path}': {exc}",
            file=sys.stderr,
        )
        return None

    if not decoded:
        return None

    if _has_control_characters(decoded):
        print(
            f"::warning::Ignoring {description} containing control characters",
            file=sys.stderr,
        )
        return None

    try:
        candidate = Path(decoded).resolve(strict=not allow_missing)
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

