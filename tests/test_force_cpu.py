import os
import importlib
import sys
import utils


def test_force_cpu(monkeypatch):
    monkeypatch.setenv("FORCE_CPU", "1")
    sys.modules["utils"] = utils
    importlib.reload(utils)
    assert utils.is_cuda_available() is False
