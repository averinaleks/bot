"""Utilities for loading .env configuration.

This module wraps :func:`dotenv.dotenv_values` to provide a resilient helper
that mirrors the expectations in :mod:`config`.
"""

from __future__ import annotations

from typing import Dict

from dotenv import dotenv_values as _dotenv_values


def dotenv_values() -> Dict[str, str]:
    """Load variables from a ``.env`` file, filtering out ``None`` entries."""

    try:
        values = _dotenv_values()
    except Exception:
        return {}

    return {key: value for key, value in values.items() if value is not None}
