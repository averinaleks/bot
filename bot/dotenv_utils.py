"""Utilities for loading .env configuration.

This module wraps :func:`dotenv.dotenv_values` to provide a resilient helper
that mirrors the expectations in :mod:`config`.
"""

from __future__ import annotations

import os
import sys
from os import PathLike
from typing import IO, Dict

try:  # pragma: no cover - exercised indirectly via tests
    from dotenv import dotenv_values as _dotenv_values
    from dotenv import load_dotenv as _load_dotenv

    DOTENV_AVAILABLE = True
    DOTENV_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - handled via fallback
    DOTENV_AVAILABLE = False
    DOTENV_ERROR = str(exc)

    # Fallback parser for environments without ``python-dotenv`` installed.
    # It supports simple ``KEY=VALUE`` pairs and ignores blank lines and
    # comments starting with ``#``. Values retain surrounding whitespace to
    # avoid surprising behaviour compared to the optional dependency.
    def _dotenv_values(
        dotenv_path: str | PathLike[str] | None = None,
        stream: IO[str] | None = None,
        verbose: bool = False,
        interpolate: bool = True,
        encoding: str | None = "utf-8",
    ) -> Dict[str, str | None]:
        path = dotenv_path or ".env"
        if not os.path.exists(path):
            return {}

        values: Dict[str, str | None] = {}
        try:
            with open(path, "r", encoding="utf-8") as env_file:
                for line in env_file:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue

                    if "=" not in stripped:
                        continue

                    key, value = stripped.split("=", 1)
                    key = key.strip()
                    if not key:
                        continue

                    values[key] = value
        except OSError:
            return {}

        return values

    def _load_dotenv(
        dotenv_path: str | PathLike[str] | None = None,
        stream: IO[str] | None = None,
        verbose: bool = False,
        override: bool = False,
        interpolate: bool = True,
        encoding: str | None = "utf-8",
    ) -> bool:
        """Fallback no-op when :mod:`python-dotenv` is unavailable."""

        return False


def dotenv_values() -> Dict[str, str]:
    """Load variables from a ``.env`` file, filtering out ``None`` entries."""

    try:
        values = _dotenv_values()
    except Exception:
        return {}

    filtered: Dict[str, str] = {}
    for key, value in values.items():
        if value is not None:
            filtered[key] = value

    return filtered


def load_dotenv() -> None:
    """Load variables from a ``.env`` file into :mod:`os.environ`.

    Existing environment variables are preserved to mirror the default
    behaviour of :func:`dotenv.load_dotenv`.
    """

    if (
        "pytest" in sys.modules
        or os.getenv("PYTEST_CURRENT_TEST") is not None
    ) and os.getenv("FORCE_DOTENV_IN_TESTS") != "1":
        return

    if DOTENV_AVAILABLE:
        # Defer to the real implementation when available to mirror
        # :func:`dotenv.load_dotenv` semantics (e.g. handling of quoted values).
        try:
            _load_dotenv(override=False)
        except Exception:
            return
        return

    try:
        values = _dotenv_values()
    except Exception:
        return

    for key, value in values.items():
        if value is not None and key not in os.environ:
            os.environ[key] = value
