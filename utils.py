"""Compatibility shim exposing :mod:`bot.utils` at the legacy top-level."""
from bot.utils import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
