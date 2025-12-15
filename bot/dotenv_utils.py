"""Utilities for loading .env configuration.

This module wraps :func:`dotenv.dotenv_values` to provide a resilient helper
that mirrors the expectations in :mod:`config`.
"""

from __future__ import annotations

import os
from typing import Dict

try:
    from dotenv import dotenv_values as _dotenv_values
except ModuleNotFoundError:
    # Fallback parser for environments without ``python-dotenv`` installed.
    # It supports simple ``KEY=VALUE`` pairs and ignores blank lines and
    # comments starting with ``#``. Values retain surrounding whitespace to
    # avoid surprising behaviour compared to the optional dependency.
    def _dotenv_values(path: str = ".env") -> Dict[str, str]:
        if not os.path.exists(path):
            return {}

        values: Dict[str, str] = {}
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


def dotenv_values() -> Dict[str, str]:
    """Load variables from a ``.env`` file, filtering out ``None`` entries."""

    try:
        values = _dotenv_values()
    except Exception:
        return {}

    return {key: value for key, value in values.items() if value is not None}


def load_dotenv() -> None:
    """Load variables from a ``.env`` file into :mod:`os.environ`.

    Existing environment variables are preserved to mirror the default
    behaviour of :func:`dotenv.load_dotenv`.
    """

    try:
        values = _dotenv_values()
    except Exception:
        return

    for key, value in values.items():
        if value is not None and key not in os.environ:
            os.environ[key] = value
