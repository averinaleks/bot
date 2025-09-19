"""Tests for the Werkzeug fallback used in :mod:`services.model_builder_service`."""

from __future__ import annotations

import builtins
import importlib
import sys


def test_secure_filename_fallback_sanitizes(monkeypatch):
    """Ensure the fallback ``secure_filename`` strips dangerous characters."""

    # Ensure the module under test is re-imported without Werkzeug available.
    monkeypatch.delitem(sys.modules, "services.model_builder_service", raising=False)
    monkeypatch.delitem(sys.modules, "werkzeug.utils", raising=False)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
        if name == "werkzeug.utils":
            raise ImportError("werkzeug not installed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module = importlib.import_module("services.model_builder_service")

    malicious = "../..//evil\\model.pkl"
    sanitized = module.secure_filename(malicious)

    assert sanitized == module._fallback_secure_filename(malicious)
    assert ".." not in sanitized
    assert "/" not in sanitized
    assert "\\" not in sanitized
