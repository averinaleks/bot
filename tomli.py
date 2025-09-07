"""Minimal tomli stub using built-in tomllib."""

import tomllib as _tomllib

loads = _tomllib.loads

__all__ = ["loads"]
