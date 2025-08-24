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


def load_dotenv(*args, **kwargs) -> bool:
    """No-op stand-in for :func:`python_dotenv.load_dotenv`.

    Returns ``True`` to mirror the real function which indicates whether a
    dotenv file was loaded. In this stub, no files are read.
    """

    return True


__all__ = ["dotenv_values", "load_dotenv"]

