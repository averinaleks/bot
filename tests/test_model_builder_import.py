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
        monkeypatch.setitem(sys.modules, 'torch', torch_stub)
        monkeypatch.setitem(sys.modules, 'torch.nn', nn_stub)
        monkeypatch.setitem(sys.modules, 'torch.utils', utils_stub)
        monkeypatch.setitem(sys.modules, 'torch.utils.data', data_stub)
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
        monkeypatch.setitem(sys.modules, 'torch', torch_stub)
        monkeypatch.setitem(sys.modules, 'torch.nn', nn_stub)
        monkeypatch.setitem(sys.modules, 'torch.utils', utils_stub)
        monkeypatch.setitem(sys.modules, 'torch.utils.data', data_stub)
    gym_stub = types.ModuleType('gymnasium')
    gym_stub.Env = object
    gym_stub.spaces = types.ModuleType('spaces')
    monkeypatch.setitem(sys.modules, 'gymnasium', gym_stub)
    monkeypatch.setitem(sys.modules, 'mlflow', None)
    sk_mod = types.ModuleType('sklearn')
    preproc = types.ModuleType('sklearn.preprocessing')
    preproc.StandardScaler = object
    sk_mod.preprocessing = preproc
    linear_mod = types.ModuleType('sklearn.linear_model')
    linear_mod.LogisticRegression = object
    metrics_mod = types.ModuleType('sklearn.metrics')
    metrics_mod.brier_score_loss = lambda *a, **k: 0.0
    calib_mod = types.ModuleType('sklearn.calibration')
    calib_mod.calibration_curve = lambda *a, **k: ([], [])
    sk_mod.linear_model = linear_mod
    sk_mod.metrics = metrics_mod
    sk_mod.calibration = calib_mod
    monkeypatch.setitem(sys.modules, 'sklearn', sk_mod)
    monkeypatch.setitem(sys.modules, 'sklearn.preprocessing', preproc)
    monkeypatch.setitem(sys.modules, 'sklearn.linear_model', linear_mod)
    monkeypatch.setitem(sys.modules, 'sklearn.metrics', metrics_mod)
    monkeypatch.setitem(sys.modules, 'sklearn.calibration', calib_mod)
    importlib.import_module('model_builder')
