"""Minimal stub of pip_audit for testing."""
from importlib import metadata

__all__ = ["__version__"]

try:
    __version__ = metadata.version("pip-audit")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback when not installed
    __version__ = "0.0.0-stub"
