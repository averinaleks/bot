"""Backward compatible facade for the :mod:`model_builder` package."""

from __future__ import annotations

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
import sys
import types

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

_core_module.ensure_gym_available()

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

        core_dict = getattr(_core_module, "__dict__", None)
        if core_dict is not None:
            core_dict[name] = value

    def __delattr__(self, name: str) -> None:  # type: ignore[override]
        if name == "_model":
            raise AttributeError("_model attribute cannot be deleted")

        super().__delattr__(name)

        core_dict = getattr(_core_module, "__dict__", None)
        if core_dict is not None and name in core_dict:
            del core_dict[name]


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
