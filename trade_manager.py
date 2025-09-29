"""Backwards compatible entry point for :mod:`bot.trade_manager`."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _load_trade_manager() -> ModuleType:
    sys.modules.pop("bot.trade_manager", None)
    return importlib.import_module("bot.trade_manager")


_MODULE = _load_trade_manager()
globals().update(_MODULE.__dict__)
sys.modules[__name__] = _MODULE


