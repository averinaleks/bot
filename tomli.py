"""Minimal TOML loader using the stdlib when available."""

try:  # Python 3.11+
    import tomllib as _tomllib  # type: ignore
except ModuleNotFoundError:  # Python <3.11
    import importlib
    import os
    import sys

    _path = sys.path.copy()
    _module = sys.modules.pop(__name__, None)
    try:
        sys.path = [p for p in sys.path if p not in ("", ".", os.path.dirname(__file__))]
        _tomllib = importlib.import_module("tomli")  # type: ignore
    finally:
        if _module is not None:
            sys.modules[__name__] = _module
        sys.path = _path

loads = _tomllib.loads

__all__ = ["loads"]
