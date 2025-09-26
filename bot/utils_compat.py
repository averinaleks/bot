"""Helpers to access :mod:`bot.utils` attributes with safe fallbacks."""

from __future__ import annotations

from importlib import import_module

from . import _core_utils


def get(name: str):
    """Return an attribute from :mod:`bot.utils` falling back to core utils."""

    utils_module = import_module("bot.utils")
    return getattr(utils_module, name, getattr(_core_utils, name))
