from __future__ import annotations

"""Configuration loader for the trading bot.

This module defines the :class:`BotConfig` dataclass along with helpers to
load configuration values from ``config.json`` and environment variables.
"""

import json
import logging
import os
from pathlib import Path
from dataclasses import MISSING, dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints


logger = logging.getLogger(__name__)

# Load defaults from config.json
CONFIG_PATH = os.getenv(
    "CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.json")
)
try:
    with open(CONFIG_PATH, "r") as f:
        DEFAULTS = json.load(f)
except (OSError, json.JSONDecodeError) as exc:
    logger.warning("Failed to load %s: %s", CONFIG_PATH, exc)
    DEFAULTS = {}


def _get_default(key: str, fallback: Any) -> Any:
    return DEFAULTS.get(key, fallback)


@dataclass
class BotConfig:
    exchange: str = _get_default("exchange", "bybit")
    timeframe: str = _get_default("timeframe", "1m")
    secondary_timeframe: str = _get_default("secondary_timeframe", "2h")
    ws_url: str = _get_default("ws_url", "wss://stream.bybit.com/v5/public/linear")
    backup_ws_urls: List[str] = field(
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
    ws_subscription_batch_size: Optional[int] = _get_default(
        "ws_subscription_batch_size", None
    )
    ws_rate_limit: int = _get_default("ws_rate_limit", 20)
    ws_reconnect_interval: int = _get_default("ws_reconnect_interval", 5)
    max_reconnect_attempts: int = _get_default("max_reconnect_attempts", 10)
    ws_inactivity_timeout: int = _get_default("ws_inactivity_timeout", 30)
    latency_log_interval: int = _get_default("latency_log_interval", 3600)
    load_threshold: float = _get_default("load_threshold", 0.8)
    leverage: int = _get_default("leverage", 10)
    min_risk_per_trade: float = _get_default("min_risk_per_trade", 0.01)
    max_risk_per_trade: float = _get_default("max_risk_per_trade", 0.05)
    risk_sharpe_loss_factor: float = _get_default("risk_sharpe_loss_factor", 0.5)
    risk_sharpe_win_factor: float = _get_default("risk_sharpe_win_factor", 1.5)
    risk_vol_min: float = _get_default("risk_vol_min", 0.5)
    risk_vol_max: float = _get_default("risk_vol_max", 2.0)
    max_positions: int = _get_default("max_positions", 5)
    top_signals: int = _get_default("top_signals", DEFAULTS.get("max_positions", 5))
    check_interval: int = _get_default("check_interval", 60)
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
    early_stopping_patience: int = _get_default("early_stopping_patience", 3)
    balance_key: Optional[str] = _get_default("balance_key", None)

    def __post_init__(self) -> None:
        if self.ws_subscription_batch_size is None:
            self.ws_subscription_batch_size = self.max_subscriptions_per_connection

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def update(self, other: Dict[str, Any]) -> None:
        for k, v in other.items():
            setattr(self, k, v)

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


def _convert(value: str, typ: type, fallback: Any | None = None) -> Any:
    if typ is bool:
        return value.lower() in {"1", "true", "yes", "on"}
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
    cfg: Dict[str, Any] = {}
    resolved_path = Path(path).resolve()
    allowed_dir = Path(CONFIG_PATH).resolve().parent
    if not resolved_path.is_relative_to(allowed_dir):
        raise ValueError(f"Path {resolved_path} is outside of {allowed_dir}")
    if resolved_path.exists():
        with open(resolved_path, "r") as f:
            try:
                cfg.update(json.load(f))
            except json.JSONDecodeError as exc:
                logger.warning("Failed to decode %s: %s", resolved_path, exc)
                f.seek(0)
                content = f.read()
                end = content.rfind("}")
                if end != -1:
                    try:
                        cfg.update(json.loads(content[: end + 1]))
                    except json.JSONDecodeError:
                        pass
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
