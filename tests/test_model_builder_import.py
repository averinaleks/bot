import importlib
import sys
import types
import pytest


def test_model_builder_requires_gymnasium(monkeypatch):
    sys.modules.pop('model_builder', None)
    sys.modules.pop('utils', None)
    if 'torch' not in sys.modules:
        torch_stub = types.ModuleType('torch')
        nn_stub = types.ModuleType('torch.nn')
        utils_stub = types.ModuleType('torch.utils')
        data_stub = types.ModuleType('torch.utils.data')
        # minimal attributes used during import
        data_stub.DataLoader = object()
        data_stub.TensorDataset = object()
        nn_stub.Module = object
        torch_stub.nn = nn_stub
        torch_stub.utils = utils_stub
        utils_stub.data = data_stub
        sys.modules['torch'] = torch_stub
        sys.modules['torch.nn'] = nn_stub
        sys.modules['torch.utils'] = utils_stub
        sys.modules['torch.utils.data'] = data_stub
    monkeypatch.setitem(sys.modules, 'gymnasium', None)
    with pytest.raises(ImportError, match='gymnasium package is required'):
        importlib.import_module('model_builder')


def test_model_builder_imports_without_mlflow(monkeypatch):
    sys.modules.pop('model_builder', None)
    sys.modules.pop('utils', None)
    if 'torch' not in sys.modules:
        torch_stub = types.ModuleType('torch')
        nn_stub = types.ModuleType('torch.nn')
        utils_stub = types.ModuleType('torch.utils')
        data_stub = types.ModuleType('torch.utils.data')
        data_stub.DataLoader = object()
        data_stub.TensorDataset = object()
        nn_stub.Module = object
        torch_stub.nn = nn_stub
        torch_stub.utils = utils_stub
        utils_stub.data = data_stub
        sys.modules['torch'] = torch_stub
        sys.modules['torch.nn'] = nn_stub
        sys.modules['torch.utils'] = utils_stub
        sys.modules['torch.utils.data'] = data_stub
    gym_stub = types.ModuleType('gymnasium')
    gym_stub.Env = object
    gym_stub.spaces = types.ModuleType('spaces')
    monkeypatch.setitem(sys.modules, 'gymnasium', gym_stub)
    monkeypatch.setitem(sys.modules, 'mlflow', None)
    importlib.import_module('model_builder')
