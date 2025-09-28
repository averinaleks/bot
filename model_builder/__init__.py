"""Backward compatible facade for the :mod:`model_builder` package."""

from __future__ import annotations

import os
import sys
import types

_ALLOW_GYM_STUB = os.getenv("ALLOW_GYM_STUB", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

if not _ALLOW_GYM_STUB:
    if "gymnasium" in sys.modules and sys.modules["gymnasium"] is None:
        raise ImportError("gymnasium package is required")

from . import core as _core_module
from .core import (
    IS_RAY_STUB,
    DQN,
    DummyVecEnv,
    ModelBuilder,
    PPO,
    RLAgent,
    SB3_AVAILABLE,
    TradingEnv,
    _freeze_keras_base_layers,
    _freeze_torch_base_layers,
    _get_torch_modules,
    _train_model_keras,
    _train_model_lightning,
    _train_model_remote,
    check_dataframe_empty,
    ensure_writable_directory,
    fit_scaler,
    generate_time_series_splits,
    gym,
    is_cuda_available,
    logger,
    prepare_features,
    ray,
    shap,
    spaces,
    validate_host,
)

from . import api as _api
from .storage import (
    JOBLIB_AVAILABLE,
    MODEL_DIR,
    MODEL_FILE,
    joblib,
    save_artifacts,
    _is_within_directory,
    _resolve_model_artifact,
    _safe_model_file_path,
)

api_app = _api.api_app
configure_logging = _api.configure_logging
api_main = _api.main
ping = _api.ping
predict_route = _api.predict_route
train_route = _api.train_route


def _load_model() -> None:
    """Reload the cached scikit-learn model via :mod:`model_builder.api`."""

    _api._load_model()


class _ModelBuilderModule(types.ModuleType):
    """Module wrapper syncing ``_model`` with :mod:`model_builder.api`."""

    def __getattr__(self, name: str):  # type: ignore[override]
        if name == "_model":
            return _api._model
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if hasattr(_core_module, name):
                return getattr(_core_module, name)
            raise

    def __setattr__(self, name: str, value) -> None:  # type: ignore[override]
        if name == "_model":
            _api._model = value
            return

        super().__setattr__(name, value)

        if hasattr(_core_module, name):
            # ``_core_module`` is a plain :class:`types.ModuleType`.  Using
            # :func:`setattr` would re-enter this method via the monkeypatching
            # helper used in the tests which triggered an infinite recursion.
            # Updating ``__dict__`` writes the attribute directly without
            # invoking any custom descriptors.
            _core_module.__dict__[name] = value


sys.modules[__name__].__class__ = _ModelBuilderModule

__all__ = [
    "IS_RAY_STUB",
    "DQN",
    "DummyVecEnv",
    "ModelBuilder",
    "PPO",
    "RLAgent",
    "SB3_AVAILABLE",
    "TradingEnv",
    "_freeze_keras_base_layers",
    "_freeze_torch_base_layers",
    "_get_torch_modules",
    "_train_model_keras",
    "_train_model_lightning",
    "_train_model_remote",
    "check_dataframe_empty",
    "ensure_writable_directory",
    "fit_scaler",
    "generate_time_series_splits",
    "gym",
    "is_cuda_available",
    "logger",
    "prepare_features",
    "ray",
    "shap",
    "spaces",
    "validate_host",
    "JOBLIB_AVAILABLE",
    "MODEL_DIR",
    "MODEL_FILE",
    "joblib",
    "save_artifacts",
    "_is_within_directory",
    "_resolve_model_artifact",
    "_safe_model_file_path",
    "api_app",
    "configure_logging",
    "api_main",
    "ping",
    "predict_route",
    "train_route",
    "_load_model",
    "_model",
]
