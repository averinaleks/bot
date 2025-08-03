import importlib
import os
import sys
from bot import utils


def test_force_cpu(monkeypatch):
    monkeypatch.setenv("FORCE_CPU", "1")
    sys.modules["utils"] = utils
    sys.modules["bot.utils"] = utils
    if getattr(utils, "__spec__", None) is None:
        from importlib.machinery import ModuleSpec
        utils.__spec__ = ModuleSpec("bot.utils", None)
    importlib.reload(utils)
    assert utils.is_cuda_available() is False
