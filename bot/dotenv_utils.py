"""Utilities for loading .env configuration.

This module wraps :func:`dotenv.dotenv_values` to provide a resilient helper
that mirrors the expectations in :mod:`config`.
"""

from __future__ import annotations

import os
from typing import Dict

from dotenv import dotenv_values as _dotenv_values


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
