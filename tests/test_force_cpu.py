import os
import importlib
import sys
from bot import utils


def test_force_cpu(monkeypatch):
    monkeypatch.setenv("FORCE_CPU", "1")
    sys.modules["utils"] = utils
    if getattr(utils, "__spec__", None) is None:
        import importlib.machinery
        utils.__spec__ = importlib.machinery.ModuleSpec("bot.utils", None)
    importlib.reload(utils)
    assert utils.is_cuda_available() is False
