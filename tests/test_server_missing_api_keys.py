import importlib
import sys
import os
import pytest


def test_missing_api_keys_causes_startup_failure(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    sys.modules.pop("server", None)
    with pytest.raises(RuntimeError, match="API_KEYS environment variable is required"):
        importlib.import_module("server")
    sys.modules.pop("server", None)
    os.environ["API_KEYS"] = "testkey"
