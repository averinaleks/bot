"""Tests for CodeQL and CI specific behaviour in :mod:`sitecustomize`."""

from __future__ import annotations

import importlib
from types import ModuleType


def _reload_sitecustomize() -> ModuleType:
    """Reload :mod:`sitecustomize` to pick up environment changes."""

    import sitecustomize  # type: ignore[import-untyped]

    return importlib.reload(sitecustomize)


def test_ensure_packages_skips_in_github_actions(monkeypatch):
    """Auto-install hooks should be disabled on GitHub Actions runners."""

    sitecustomize = importlib.import_module("sitecustomize")  # type: ignore[import-untyped]

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("BOT_AUTO_INSTALL_DISABLED", raising=False)
    monkeypatch.delenv("CODEQL_DIST", raising=False)

    # ``sitecustomize`` caches the CodeQL detection helper at import time, so
    # ensure the reload sees the adjusted environment variables.
    sitecustomize = _reload_sitecustomize()

    calls: list[str] = []
    monkeypatch.setattr(sitecustomize, "_run_pip_install", calls.append)

    sitecustomize._ensure_packages([("definitely_missing", "package>=1.0")])

    assert calls == []

