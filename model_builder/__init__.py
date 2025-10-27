"""Backward compatible facade for the :mod:`model_builder` package."""

from __future__ import annotations

import logging
import os
import sys
import traceback
import types

from bot import config as bot_config

from .offline import OfflineModelBuilder

_CORE_AVAILABLE = False
_CORE_IMPORT_ERROR: ImportError | None = None
_CORE_IMPORT_TRACEBACK = ""
_core_module: types.ModuleType | None = None
_model: object | None = None

_OFFLINE_REQUESTED = bool(getattr(bot_config, "OFFLINE_MODE", False))

if not _OFFLINE_REQUESTED:
    try:
        from . import core as _core_module  # type: ignore[assignment]
    except ImportError as exc:
        _CORE_IMPORT_ERROR = exc
        _CORE_IMPORT_TRACEBACK = traceback.format_exc()
        _core_module = None
        _OFFLINE_REQUESTED = True
    else:
        _CORE_AVAILABLE = True

if _CORE_AVAILABLE:
    from .core import (
        IS_RAY_STUB,
        DQN,
        DummyVecEnv,
        ModelBuilder,
        PPO,
        RLAgent,
        SB3_AVAILABLE,
        TradingEnv,
        KERAS_FRAMEWORKS,
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
else:
    logger = logging.getLogger("TradingBot")
    ModelBuilder = OfflineModelBuilder

    if _CORE_IMPORT_ERROR is not None:
        logger.warning(
            "model_builder.core недоступен: %s", _CORE_IMPORT_ERROR
        )

    def _offline_unavailable(name: str):  # noqa: ANN001
        def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError(
                "model_builder.core недоступен (OFFLINE_MODE=1 или отсутствуют зависимости); "
                f"обращение к {name} невозможно"
            ) from _CORE_IMPORT_ERROR

        return _raise

    class _OfflineProxy:
        __offline_stub__ = True

        def __init__(self, name: str) -> None:
            self.__name__ = name

        def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError(
                "model_builder.core недоступен (OFFLINE_MODE=1 или отсутствуют зависимости); "
                f"обращение к {self.__name__} невозможно"
            ) from _CORE_IMPORT_ERROR

        def __getattr__(self, item: str):  # type: ignore[override]
            raise RuntimeError(
                "model_builder.core недоступен (OFFLINE_MODE=1 или отсутствуют зависимости); "
                f"обращение к {self.__name__}.{item} невозможно"
            ) from _CORE_IMPORT_ERROR

        def __repr__(self) -> str:  # pragma: no cover - диагностика
            return f"<OfflineProxy {self.__name__}>"

    IS_RAY_STUB = True
    DQN = _OfflineProxy("DQN")
    DummyVecEnv = _OfflineProxy("DummyVecEnv")
    PPO = _OfflineProxy("PPO")
    RLAgent = _OfflineProxy("RLAgent")
    SB3_AVAILABLE = False
    TradingEnv = _OfflineProxy("TradingEnv")
    KERAS_FRAMEWORKS: tuple[str, ...] = ()
    _freeze_keras_base_layers = _offline_unavailable("_freeze_keras_base_layers")
    _freeze_torch_base_layers = _offline_unavailable("_freeze_torch_base_layers")
    _get_torch_modules = _offline_unavailable("_get_torch_modules")
    _train_model_keras = _offline_unavailable("_train_model_keras")
    _train_model_lightning = _offline_unavailable("_train_model_lightning")
    _train_model_remote = _offline_unavailable("_train_model_remote")
    check_dataframe_empty = _offline_unavailable("check_dataframe_empty")
    ensure_writable_directory = _offline_unavailable("ensure_writable_directory")
    fit_scaler = _offline_unavailable("fit_scaler")
    generate_time_series_splits = _offline_unavailable("generate_time_series_splits")
    gym = _OfflineProxy("gym")
    is_cuda_available = _offline_unavailable("is_cuda_available")
    prepare_features = _offline_unavailable("prepare_features")
    ray = _OfflineProxy("ray")
    shap = _OfflineProxy("shap")
    spaces = _OfflineProxy("spaces")
    validate_host = _offline_unavailable("validate_host")
    JOBLIB_AVAILABLE = False
    MODEL_DIR = None
    MODEL_FILE = None
    joblib = types.SimpleNamespace(
        dump=_offline_unavailable("joblib.dump"),
        load=_offline_unavailable("joblib.load"),
    )
    save_artifacts = _offline_unavailable("save_artifacts")
    _is_within_directory = _offline_unavailable("_is_within_directory")
    _resolve_model_artifact = _offline_unavailable("_resolve_model_artifact")
    _safe_model_file_path = _offline_unavailable("_safe_model_file_path")

_API_AVAILABLE = True
_API_IMPORT_ERROR: ImportError | None = None
_API_IMPORT_TRACEBACK = ""

try:
    from . import api as _api
except ImportError as exc:
    _API_AVAILABLE = False
    _API_IMPORT_ERROR = exc
    _API_IMPORT_TRACEBACK = traceback.format_exc()

    def _api_stub(name: str):
        """Return a callable stub emitting a helpful error for missing Flask."""

        def _stub(*args, **kwargs):  # type: ignore[no-untyped-def]
            logger.warning(
                "model_builder.api недоступен (Flask не установлен); вызов %s пропущен",
                name,
            )
            raise RuntimeError(
                "model_builder.api недоступен (Flask не установлен). "
                "См. исходный ImportError и трассировку ниже:\n"
                f"{_API_IMPORT_TRACEBACK.strip()}"
            ) from _API_IMPORT_ERROR

        return _stub

    class _ApiAppStub:
        """Мнимое приложение Flask, предотвращающее использование REST API."""

        def __getattr__(self, item):  # type: ignore[override]
            return _api_stub(f"api_app.{item}")

        def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return _api_stub("api_app")(*args, **kwargs)

    def _load_model_stub() -> None:
        logger.warning(
            "model_builder.api недоступен (Flask не установлен); загрузка модели пропущена. "
            "См. исходный ImportError:\n%s",
            _API_IMPORT_TRACEBACK.strip(),
        )

    _api = types.SimpleNamespace()
    _api._model = None
    _api._load_model = _load_model_stub
    _api.api_app = _ApiAppStub()
    _api.configure_logging = _api_stub("configure_logging")
    _api.main = _api_stub("main")
    _api.ping = _api_stub("ping")
    _api.predict_route = _api_stub("predict_route")
    _api.train_route = _api_stub("train_route")

api_app = _api.api_app
configure_logging = _api.configure_logging
api_main = _api.main
ping = _api.ping
predict_route = _api.predict_route
train_route = _api.train_route


def _load_model() -> None:
    """Reload the cached scikit-learn model via :mod:`model_builder.api`."""

    if not _API_AVAILABLE:
        logger.warning(
            "model_builder.api недоступен, загрузка модели пропущена. "
            "См. исходный ImportError:\n%s",
            _API_IMPORT_TRACEBACK.strip(),
        )
        return

    _api._load_model()
    globals()["_model"] = _api._model


class _ModelBuilderModule(types.ModuleType):
    """Module wrapper syncing ``_model`` with :mod:`model_builder.api`."""

    def __getattr__(self, name: str):  # type: ignore[override]
        if name == "_model":
            return _api._model
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if _CORE_AVAILABLE and _core_module and hasattr(_core_module, name):
                return getattr(_core_module, name)
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value) -> None:  # type: ignore[override]
        if name == "_model":
            _api._model = value
            globals()["_model"] = value
        else:
            super().__setattr__(name, value)
            if _CORE_AVAILABLE and _core_module and hasattr(_core_module, name):
                _core_module.__dict__[name] = value

sys.modules[__name__].__class__ = _ModelBuilderModule

if _CORE_AVAILABLE and os.getenv("ALLOW_GYM_STUB", "1").strip().lower() in {"0", "false", "no"}:
    if getattr(getattr(_core_module, "gym", None), "_BOT_GYM_STUB", False):
        raise ImportError("gymnasium package is required")
    import importlib

    try:
        importlib.import_module("gymnasium")
    except ImportError as exc:
        raise ImportError("gymnasium package is required") from exc

__all__ = [
    "ModelBuilder",
    "OfflineModelBuilder",
    "api_app",
    "configure_logging",
    "api_main",
    "ping",
    "predict_route",
    "train_route",
    "_load_model",
    "_model",
]

if _CORE_AVAILABLE:
    __all__.extend(
        [
            "IS_RAY_STUB",
            "DQN",
            "DummyVecEnv",
            "PPO",
            "RLAgent",
            "SB3_AVAILABLE",
            "TradingEnv",
            "KERAS_FRAMEWORKS",
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
        ]
    )
