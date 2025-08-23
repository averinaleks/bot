import importlib
import os
import sys


def test_import_does_not_set_tf_log_level(monkeypatch):
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)
    monkeypatch.delitem(sys.modules, "bot.utils", raising=False)
    importlib.import_module("bot.utils")
    assert "TF_CPP_MIN_LOG_LEVEL" not in os.environ
