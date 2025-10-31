"""Tests for CodeQL and CI specific behaviour in :mod:`sitecustomize`."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _reload_sitecustomize() -> ModuleType:
    """Reload :mod:`sitecustomize` to pick up environment changes."""

    module_name = "sitecustomize"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)  # type: ignore[import-untyped]



def test_ensure_packages_skips_in_github_actions(monkeypatch):
    """Auto-install hooks should be disabled on GitHub Actions runners."""


    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("BOT_AUTO_INSTALL_DISABLED", raising=False)
    monkeypatch.delenv("CODEQL_DIST", raising=False)

    # ``sitecustomize`` caches the CodeQL detection helper at import time, so
    # ensure the reload sees the adjusted environment variables.

    sitecustomize_module = _reload_sitecustomize()

    calls: list[str] = []
    monkeypatch.setattr(sitecustomize_module, "_run_pip_install", calls.append)

    sitecustomize_module._ensure_packages([("definitely_missing", "package>=1.0")])

    assert calls == []

