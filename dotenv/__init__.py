"""Minimal stub of python-dotenv for tests.

This stub provides :func:`dotenv_values` expected by ``pydantic-settings``.
It returns an empty mapping and does not load any environment files.
"""

from __future__ import annotations

from typing import Mapping


def dotenv_values(*args, **kwargs) -> Mapping[str, str]:
    """Return an empty mapping of environment variables.

    The real ``python-dotenv`` package loads environment variables from files.
    For the purposes of these tests, an empty mapping is sufficient.
    """

    return {}


def load_dotenv(*args, **kwargs) -> None:
    """A no-op replacement for :func:`python_dotenv.load_dotenv`."""
    return None

__all__ = ["dotenv_values", "load_dotenv"]

