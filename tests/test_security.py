"""Тесты для модуля security."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import os

import pytest

from security import (
    _MLFLOW_DISABLED_ATTRS,
    apply_ray_security_defaults,
    ensure_minimum_ray_version,
    harden_mlflow,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Очистить переменные окружения, которые изменяют функции безопасности."""

    for key in ("RAY_DISABLE_DASHBOARD", "RAY_JOB_ALLOWLIST", "MLFLOW_ENABLE_MODEL_LOADING"):
        monkeypatch.delenv(key, raising=False)


def _build_mlflow_stub() -> ModuleType:
    """Создать имитацию mlflow со всеми уязвимыми точками входа."""

    mlflow_stub = ModuleType("mlflow")

    def _ensure_path(module: ModuleType, path: tuple[str, ...]) -> ModuleType:
        parent = module
        accumulated: list[str] = []
        for part in path:
            accumulated.append(part)
            try:
                child = getattr(parent, part)
            except AttributeError:
                child = ModuleType("mlflow." + ".".join(accumulated))
                setattr(parent, part, child)
            parent = child
        return parent

    def _dummy_loader(*_args: Any, **_kwargs: Any) -> str:
        return "loaded"

    for path, attr in _MLFLOW_DISABLED_ATTRS:
        target_module = _ensure_path(mlflow_stub, path)
        setattr(target_module, attr, _dummy_loader)

    models_module = ModuleType("mlflow.models")

    class _Model:
        @staticmethod
        def load(*_args: Any, **_kwargs: Any) -> str:
            return "loaded"

    models_module.Model = _Model
    mlflow_stub.models = models_module

    recipes_module = getattr(mlflow_stub, "recipes", ModuleType("mlflow.recipes"))

    class _Recipe:
        @staticmethod
        def load(*_args: Any, **_kwargs: Any) -> str:
            return "loaded"

    recipes_module.Recipe = _Recipe
    setattr(mlflow_stub, "recipes", recipes_module)

    return mlflow_stub


def test_apply_ray_security_defaults_sets_safe_values() -> None:
    params = {"num_cpus": 4}
    hardened = apply_ray_security_defaults(params)

    assert hardened is not params
    assert hardened["num_cpus"] == 4
    assert hardened["include_dashboard"] is False
    assert hardened["dashboard_host"] == "127.0.0.1"
    assert os.environ["RAY_DISABLE_DASHBOARD"] == "1"
    assert os.environ["RAY_JOB_ALLOWLIST"] == ""

    custom = apply_ray_security_defaults({"include_dashboard": True, "dashboard_host": "0.0.0.0"})
    assert custom["include_dashboard"] is True
    assert custom["dashboard_host"] == "0.0.0.0"


def test_harden_mlflow_disables_all_loaders() -> None:
    mlflow_stub = _build_mlflow_stub()

    harden_mlflow(mlflow_stub)

    for path, attr in _MLFLOW_DISABLED_ATTRS:
        target = mlflow_stub
        for name in path:
            target = getattr(target, name)
        loader = getattr(target, attr)
        with pytest.raises(RuntimeError) as exc:
            loader()
        assert "MLflow" in str(exc.value)

    with pytest.raises(RuntimeError):
        mlflow_stub.models.Model.load()

    recipe_cls = mlflow_stub.recipes.Recipe
    with pytest.raises(RuntimeError):
        recipe_cls.load()

    assert os.environ["MLFLOW_ENABLE_MODEL_LOADING"] == "false"


def test_ensure_minimum_ray_version_accepts_safe_release() -> None:
    ray_stub = ModuleType("ray")
    ray_stub.__version__ = "2.49.2"

    ensure_minimum_ray_version(ray_stub)


def test_ensure_minimum_ray_version_rejects_legacy_release() -> None:
    ray_stub = ModuleType("ray")
    ray_stub.__version__ = "2.49.1"

    with pytest.raises(RuntimeError) as exc:
        ensure_minimum_ray_version(ray_stub)

    assert "Ray" in str(exc.value)
