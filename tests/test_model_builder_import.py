import importlib
import sys
import pytest


def test_model_builder_requires_gymnasium(monkeypatch):
    sys.modules.pop('model_builder', None)
    sys.modules.pop('utils', None)
    monkeypatch.setitem(sys.modules, 'gymnasium', None)
    with pytest.raises(ImportError, match='gymnasium package is required'):
        importlib.import_module('model_builder')
