"""Configuration loader for the trading bot.

This module defines the :class:`BotConfig` dataclass along with helpers to
load configuration values from ``config.json`` and environment variables.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import threading
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)


class MissingEnvError(Exception):
    """Raised when required environment variables are missing."""

    def __init__(self, missing_keys: list[str]):
        self.missing_keys = tuple(missing_keys)
        message = "Missing required environment variables: " + ", ".join(missing_keys)
        super().__init__(message)


def _load_env_file() -> dict[str, str]:
    """Загрузить значения из ``.env`` при наличии python-dotenv.

    Библиотека ``python-dotenv`` не является обязательной зависимостью для
    тестов, поэтому модуль должен корректно работать и без неё. Вместо
    перехвата ошибки импорта используется ``importlib.util.find_spec`` чтобы
    избежать ``ModuleNotFoundError`` и соответствовать требованиям стиля.
    """

    spec = importlib.util.find_spec("dotenv")
    if spec is None:
        logger.warning(
            "python-dotenv не установлен: значения из .env будут проигнорированы"
        )
        return {}

    module = importlib.import_module("dotenv")
    dotenv_values = getattr(module, "dotenv_values", None)
    if callable(dotenv_values):
        loaded = dotenv_values()
        return dict(loaded) if loaded else {}

    logger.warning(
        "Модуль python-dotenv не содержит функцию dotenv_values; .env пропущен"
    )
    return {}


_env: dict[str, str] = _load_env_file()


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Прочитать булево значение из переменных окружения и ``.env``."""

    raw = os.getenv(name) or _env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def validate_env(required_keys: list[str]) -> None:
    """Ensure that required environment variables are present.

    Parameters
    ----------
    required_keys:
        List of environment variable names that must be defined. The check is
        skipped when ``TEST_MODE=1`` so tests can run without configuring the
        full environment.
    """

    if os.getenv("TEST_MODE") == "1":
        return

    missing_keys: list[str] = []
    for key in required_keys:
        if not (os.getenv(key) or _env.get(key)):
            missing_keys.append(key)

    if missing_keys:
        raise MissingEnvError(missing_keys)


OFFLINE_MODE = _get_bool_env("OFFLINE_MODE", False)

try:
    validate_env(
        [
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
            "TRADE_MANAGER_TOKEN",
            "TRADE_RISK_USD",
            "BYBIT_API_KEY",
            "BYBIT_API_SECRET",
        ]
    )
except MissingEnvError as exc:
    missing = ", ".join(exc.missing_keys)
    if OFFLINE_MODE:
        logger.warning(
            "OFFLINE_MODE=1: запуск офлайн-режима из-за отсутствующих переменных: %s",
            missing,
        )
    else:
        logger.critical(
            "Не заданы обязательные переменные окружения: %s",
            missing,
        )
        raise

# Load defaults from config.json lazily
# Resolve the default configuration file. Test runs should always use the
# repository's bundled ``config.json`` regardless of any ``CONFIG_PATH`` value
# defined in the environment (for example via ``.env``).
_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.json"


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def _resolve_config_path(raw: str | os.PathLike[str] | None) -> Path:
    """Return a config path confined to ``_CONFIG_DIR``."""

    if raw is None:
        return _DEFAULT_CONFIG_PATH

    if raw == "":
        return _DEFAULT_CONFIG_PATH

    try:
        candidate = Path(raw)
    except TypeError:
        logger.warning("Invalid CONFIG_PATH %r; falling back to default", raw)
        return _DEFAULT_CONFIG_PATH

    if not candidate.is_absolute():
        candidate = (_CONFIG_DIR / candidate).resolve(strict=False)
    else:
        try:
            candidate = candidate.resolve(strict=False)
        except OSError as exc:
            logger.warning(
                "Failed to resolve CONFIG_PATH %s: %s; using default",
                candidate,
                exc,
            )
            return _DEFAULT_CONFIG_PATH

    if candidate.is_symlink():
        logger.warning("CONFIG_PATH %s is a symlink; using default", candidate)
        return _DEFAULT_CONFIG_PATH

    if not _is_within_directory(candidate, _CONFIG_DIR):
        logger.warning(
            "CONFIG_PATH %s escapes %s; using default",
            candidate,
            _CONFIG_DIR,
        )
        return _DEFAULT_CONFIG_PATH

    return candidate


_ENV_CONFIG_PATH = None if os.getenv("TEST_MODE") == "1" else os.getenv("CONFIG_PATH")
CONFIG_PATH = str(_resolve_config_path(_ENV_CONFIG_PATH))
# Cached defaults; populated on first load
DEFAULTS: dict[str, Any] | None = None
DEFAULTS_LOCK = threading.Lock()


class ConfigLoadError(Exception):
    """Raised when the default configuration cannot be loaded."""


def load_defaults() -> dict[str, Any]:
    """Load default configuration from CONFIG_PATH on first access."""
    global DEFAULTS
    with DEFAULTS_LOCK:
        if DEFAULTS is None:
            try:
                path = _resolve_config_path(CONFIG_PATH)
                with open(path, "r", encoding="utf-8") as f:
                    DEFAULTS = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("Failed to load %s: %s", path, exc)
                raise ConfigLoadError from exc
    return DEFAULTS


def _get_default(key: str, fallback: Any) -> Any:
    defaults = load_defaults()
    return defaults.get(key, fallback)


@dataclass
class BotConfig:
    exchange: str = _get_default("exchange", "bybit")
    timeframe: str = _get_default("timeframe", "1m")
    secondary_timeframe: str = _get_default("secondary_timeframe", "2h")
    ws_url: str = _get_default("ws_url", "wss://stream.bybit.com/v5/public/linear")
    backup_ws_urls: list[str] = field(
        default_factory=lambda: _get_default(
            "backup_ws_urls", ["wss://stream.bybit.com/v5/public/linear"]
        )
    )
    max_concurrent_requests: int = _get_default("max_concurrent_requests", 10)
    max_volume_batch: int = _get_default("max_volume_batch", 50)
    history_batch_size: int = _get_default("history_batch_size", 10)
    max_symbols: int = _get_default("max_symbols", 50)
    max_subscriptions_per_connection: int = _get_default(
        "max_subscriptions_per_connection", 15
    )
    ws_subscription_batch_size: int | None = _get_default(
        "ws_subscription_batch_size", None
    )
    ws_rate_limit: int = _get_default("ws_rate_limit", 20)
    ws_reconnect_interval: int = _get_default("ws_reconnect_interval", 5)
    max_reconnect_attempts: int = _get_default("max_reconnect_attempts", 10)
    ws_inactivity_timeout: int = _get_default("ws_inactivity_timeout", 30)
    latency_log_interval: int = _get_default("latency_log_interval", 3600)
    load_threshold: float = _get_default("load_threshold", 0.8)
    leverage: int = _get_default("leverage", 10)
    max_position_pct: float = _get_default("max_position_pct", 0.1)
    min_risk_per_trade: float = _get_default("min_risk_per_trade", 0.01)
    max_risk_per_trade: float = _get_default("max_risk_per_trade", 0.05)
    risk_sharpe_loss_factor: float = _get_default("risk_sharpe_loss_factor", 0.5)
    risk_sharpe_win_factor: float = _get_default("risk_sharpe_win_factor", 1.5)
    risk_vol_min: float = _get_default("risk_vol_min", 0.5)
    risk_vol_max: float = _get_default("risk_vol_max", 2.0)
    max_positions: int = _get_default("max_positions", 5)
    top_signals: int = _get_default(
        "top_signals", _get_default("max_positions", 5)
    )
    check_interval: float = _get_default("check_interval", 60.0)
    data_cleanup_interval: int = _get_default("data_cleanup_interval", 3600)
    base_probability_threshold: float = _get_default("base_probability_threshold", 0.6)
    trailing_stop_percentage: float = _get_default("trailing_stop_percentage", 1.0)
    trailing_stop_coeff: float = _get_default("trailing_stop_coeff", 1.0)
    retrain_threshold: float = _get_default("retrain_threshold", 0.1)
    forget_window: int = _get_default("forget_window", 259200)
    trailing_stop_multiplier: float = _get_default("trailing_stop_multiplier", 1.0)
    tp_multiplier: float = _get_default("tp_multiplier", 2.0)
    sl_multiplier: float = _get_default("sl_multiplier", 1.0)
    min_sharpe_ratio: float = _get_default("min_sharpe_ratio", 0.5)
    performance_window: int = _get_default("performance_window", 86400)
    min_data_length: int = _get_default("min_data_length", 1000)
    history_retention: int = _get_default("history_retention", 200)
    lstm_timesteps: int = _get_default("lstm_timesteps", 60)
    lstm_batch_size: int = _get_default("lstm_batch_size", 32)
    model_type: str = _get_default("model_type", "transformer")
    nn_framework: str = _get_default("nn_framework", "pytorch")
    prediction_target: str = _get_default("prediction_target", "direction")
    trading_fee: float = _get_default("trading_fee", 0.0)
    ema30_period: int = _get_default("ema30_period", 30)
    ema100_period: int = _get_default("ema100_period", 100)
    ema200_period: int = _get_default("ema200_period", 200)
    atr_period_default: int = _get_default("atr_period_default", 14)
    rsi_window: int = _get_default("rsi_window", 14)
    macd_window_slow: int = _get_default("macd_window_slow", 26)
    macd_window_fast: int = _get_default("macd_window_fast", 12)
    macd_window_sign: int = _get_default("macd_window_sign", 9)
    adx_window: int = _get_default("adx_window", 14)
    bollinger_window: int = _get_default("bollinger_window", 20)
    ulcer_window: int = _get_default("ulcer_window", 14)
    volume_profile_update_interval: int = _get_default(
        "volume_profile_update_interval", 300
    )
    funding_update_interval: int = _get_default("funding_update_interval", 300)
    oi_update_interval: int = _get_default("oi_update_interval", 300)
    cache_dir: str = _get_default("cache_dir", "/app/cache")
    log_dir: str = _get_default("log_dir", "/app/logs")
    ray_num_cpus: int = _get_default("ray_num_cpus", 2)
    n_splits: int = _get_default("n_splits", 3)
    optimization_interval: int = _get_default("optimization_interval", 7200)
    shap_cache_duration: int = _get_default("shap_cache_duration", 86400)
    volatility_threshold: float = _get_default("volatility_threshold", 0.02)
    ema_crossover_lookback: int = _get_default("ema_crossover_lookback", 7200)
    pullback_period: int = _get_default("pullback_period", 3600)
    pullback_volatility_coeff: float = _get_default("pullback_volatility_coeff", 1.0)
    retrain_interval: int = _get_default("retrain_interval", 86400)
    min_liquidity: int = _get_default("min_liquidity", 1000000)
    ws_queue_size: int = _get_default("ws_queue_size", 10000)
    ws_min_process_rate: int = _get_default("ws_min_process_rate", 1)
    disk_buffer_size: int = _get_default("disk_buffer_size", 10000)
    prediction_history_size: int = _get_default("prediction_history_size", 100)
    telegram_queue_size: int = _get_default("telegram_queue_size", 100)
    optuna_trials: int = _get_default("optuna_trials", 20)
    optimizer_method: str = _get_default("optimizer_method", "tpe")
    holdout_warning_ratio: float = _get_default("holdout_warning_ratio", 0.3)
    enable_grid_search: bool = _get_default("enable_grid_search", False)
    loss_streak_threshold: int = _get_default("loss_streak_threshold", 3)
    win_streak_threshold: int = _get_default("win_streak_threshold", 3)
    threshold_adjustment: float = _get_default("threshold_adjustment", 0.05)
    threshold_decay_rate: float = _get_default("threshold_decay_rate", 0.1)
    target_change_threshold: float = _get_default("target_change_threshold", 0.001)
    backtest_interval: int = _get_default("backtest_interval", 604800)
    rl_model: str = _get_default("rl_model", "PPO")
    rl_framework: str = _get_default("rl_framework", "stable_baselines3")
    rl_timesteps: int = _get_default("rl_timesteps", 10000)
    rl_use_imitation: bool = _get_default("rl_use_imitation", False)
    drawdown_penalty: float = _get_default("drawdown_penalty", 0.0)
    mlflow_tracking_uri: str = _get_default("mlflow_tracking_uri", "mlruns")
    mlflow_enabled: bool = _get_default("mlflow_enabled", False)
    use_strategy_optimizer: bool = _get_default("use_strategy_optimizer", False)
    portfolio_metric: str = _get_default("portfolio_metric", "sharpe")
    use_polars: bool = _get_default("use_polars", False)
    fine_tune_epochs: int = _get_default("fine_tune_epochs", 5)
    use_transfer_learning: bool = _get_default("use_transfer_learning", False)
    order_retry_attempts: int = _get_default("order_retry_attempts", 3)
    order_retry_delay: float = _get_default("order_retry_delay", 1.0)
    reversal_margin: float = _get_default("reversal_margin", 0.05)
    transformer_weight: float = _get_default("transformer_weight", 0.5)
    ema_weight: float = _get_default("ema_weight", 0.2)
    gpt_weight: float = _get_default("gpt_weight", 0.3)
    early_stopping_patience: int = _get_default("early_stopping_patience", 3)
    balance_key: str | None = _get_default("balance_key", None)
    enable_notifications: bool = _get_default("enable_notifications", True)
    save_unsent_telegram: bool = _get_default("save_unsent_telegram", False)
    unsent_telegram_path: str = _get_default("unsent_telegram_path", "unsent_telegram.log")
    service_factories: dict[str, str] = field(
        default_factory=lambda: dict(_get_default("service_factories", {}))
    )

    def __post_init__(self) -> None:
        if self.ws_subscription_batch_size is None:
            self.ws_subscription_batch_size = self.max_subscriptions_per_connection
        self._validate_types()
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be between 0 and 1")

    @staticmethod
    def _isinstance(value: Any, typ: Any) -> bool:
        if isinstance(typ, types.UnionType):
            return any(BotConfig._isinstance(value, t) for t in get_args(typ))

        origin = get_origin(typ)
        if typ is float:
            return isinstance(value, (int, float))
        if origin is None:
            return isinstance(value, typ)
        if origin in {Union, types.UnionType}:
            return any(BotConfig._isinstance(value, t) for t in get_args(typ))
        if origin is list:
            if not isinstance(value, list):
                return False
            subtype = get_args(typ)[0] if get_args(typ) else Any
            return all(BotConfig._isinstance(v, subtype) for v in value)
        return isinstance(value, origin)

    def _validate_types(self) -> None:
        type_hints = get_type_hints(BotConfig)
        for fdef in fields(self):
            val = getattr(self, fdef.name)
            expected = type_hints.get(fdef.name, fdef.type)
            if not self._isinstance(val, expected):
                raise ValueError(f"Invalid type for {fdef.name}")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def update(self, other: dict[str, Any]) -> None:
        for k, v in other.items():
            setattr(self, k, v)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _convert(value: str, typ: type, fallback: Any | None = None) -> Any:
    if typ is bool:
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        logger.warning("Unknown boolean value %r", value)
        if fallback is not None:
            return fallback
        raise ValueError(f"Invalid boolean value: {value}")
    if typ is int:
        try:
            return int(value)
        except ValueError:
            logger.warning("Failed to convert %r to int", value)
            if fallback is not None:
                return fallback
            raise
    if typ is float:
        try:
            return float(value)
        except ValueError:
            logger.warning("Failed to convert %r to float", value)
            if fallback is not None:
                return fallback
            raise
    origin = get_origin(typ)
    if typ is list or origin is list:
        subtypes = get_args(typ)
        subtype = subtypes[0] if subtypes else str
        try:
            items = json.loads(value)
        except json.JSONDecodeError:
            items = [v.strip().strip("'\"") for v in value.split(",") if v.strip()]
        if not isinstance(items, list):
            items = [items]
        converted = []
        for item in items:
            try:
                converted.append(item if isinstance(item, subtype) else _convert(str(item), subtype))
            except TypeError:
                converted.append(item)
        return converted
    if fallback is not None:
        return fallback
    return value


def load_config(path: str = CONFIG_PATH) -> BotConfig:
    """Load configuration from JSON file and environment variables."""
    cfg: dict[str, Any] = {}
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (_CONFIG_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()
    allowed_dir = Path(CONFIG_PATH).resolve().parent
    if not candidate.is_relative_to(allowed_dir):
        raise ValueError(f"Path {candidate} is outside of {allowed_dir}")
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as f:
            try:
                cfg.update(json.load(f))
            except json.JSONDecodeError as exc:
                logger.warning("Failed to decode %s: %s", candidate, exc)
                f.seek(0)
                content = f.read()
                end = content.rfind("}")
                if end != -1:
                    try:
                        cfg.update(json.loads(content[: end + 1]))
                    except json.JSONDecodeError as fallback_exc:
                        logger.warning(
                            "Failed to recover configuration from %s: %s",
                            candidate,
                            fallback_exc,
                        )
    type_hints = get_type_hints(BotConfig)
    for fdef in fields(BotConfig):
        env_val = os.getenv(fdef.name.upper())
        if env_val is not None:
            expected_type = type_hints.get(fdef.name, fdef.type)
            origin = get_origin(expected_type)
            if origin is Union and type(None) in get_args(expected_type):
                expected_type = next(
                    (t for t in get_args(expected_type) if t is not type(None)),
                    Any,
                )
            if fdef.default is not MISSING:
                fallback = fdef.default
            elif fdef.default_factory is not MISSING:
                fallback = fdef.default_factory()
            else:
                fallback = None
            try:
                cfg[fdef.name] = _convert(env_val, expected_type, fallback)
            except ValueError:
                expected_name = getattr(expected_type, "__name__", str(expected_type))
                logger.warning(
                    "Ignoring %s: expected value of type %s",
                    fdef.name.upper(),
                    expected_name,
                )
    return BotConfig(**cfg)
