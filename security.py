"""Security hardening helpers for optional third-party integrations.

This module centralises mitigations for known vulnerabilities reported by
Trivy so that the codebase remains safe without dropping optional features.
"""
from __future__ import annotations

import logging
import os
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


def apply_ray_security_defaults(params: dict[str, Any]) -> dict[str, Any]:
    """Return Ray initialisation kwargs hardened against CVE-2023-48022.

    The ShadowRay advisory (CVE-2023-48022) abuses the Ray dashboard job
    submission API which is enabled by default.  We explicitly disable the
    dashboard and ensure the process only binds to the loopback interface.
    """

    os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    os.environ.setdefault("RAY_JOB_ALLOWLIST", "")
    hardened = dict(params)
    hardened.setdefault("include_dashboard", False)
    hardened.setdefault("dashboard_host", "127.0.0.1")
    return hardened


_MLFLOW_DISABLED_ATTRS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("pyfunc",), "load_model"),
    (("sklearn",), "load_model"),
    (("pytorch",), "load_model"),
    (("tensorflow",), "load_model"),
    (("lightgbm",), "load_model"),
    (("xgboost",), "load_model"),
    (("catboost",), "load_model"),
    (("pmdarima",), "load_model"),
    (("recipes",), "load_recipe"),
)


def _disable_callable(target: Any, qualname: str) -> Any:
    message = (
        "Загрузка моделей MLflow отключена по соображениям безопасности. "
        "Подробности см. в CVE-2024-37052…CVE-2024-37060."
    )

    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(message)

    if callable(target):
        logger.info("Отключена уязвимая функция MLflow: %s", qualname)
        return _blocked
    return None


def harden_mlflow(mlflow_module: ModuleType) -> None:
    """Disable model-loading entry points that lead to remote code execution.

    MLflow 3.4.0 содержит серию RCE-уязвимостей (CVE-2024-37052…37060). Мы не
    загружаем сторонние модели внутри проекта, поэтому безопасно блокировать
    соответствующие API, оставив журналы и экспорт в рабочем состоянии.
    """

    for path, attr in _MLFLOW_DISABLED_ATTRS:
        parent: Any = mlflow_module
        qualname = ["mlflow"]
        for name in path:
            try:
                parent = getattr(parent, name)
            except AttributeError:
                break
            except Exception as exc:  # pragma: no cover - optional deps may be missing
                logger.debug("Пропуск hardening для mlflow.%s: %s", ".".join(qualname[1:] + [name]), exc)
                break
            else:
                qualname.append(name)
        else:
            try:
                candidate = getattr(parent, attr)
            except AttributeError:
                continue
            except Exception as exc:  # pragma: no cover
                logger.debug("Не удалось получить %s.%s: %s", ".".join(qualname), attr, exc)
                continue
            qualname.append(attr)
            blocked = _disable_callable(candidate, ".".join(qualname))
            if blocked is not None:
                setattr(parent, attr, blocked)

    models = getattr(mlflow_module, "models", None)
    if models is not None:
        model_class = getattr(models, "Model", None)
        if model_class is not None and hasattr(model_class, "load"):
            blocked = _disable_callable(model_class.load, "mlflow.models.Model.load")
            if blocked is not None:
                model_class.load = staticmethod(blocked)

    recipes = getattr(mlflow_module, "recipes", None)
    if recipes is not None:
        recipe_class = getattr(recipes, "Recipe", None)
        if recipe_class is not None and hasattr(recipe_class, "load"):
            blocked = _disable_callable(recipe_class.load, "mlflow.recipes.Recipe.load")
            if blocked is not None:
                recipe_class.load = staticmethod(blocked)

    os.environ.setdefault("MLFLOW_ENABLE_MODEL_LOADING", "false")
    logger.info("MLflow hardened: загрузка моделей запрещена для предотвращения RCE")
