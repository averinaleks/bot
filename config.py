from __future__ import annotations

"""Configuration loader for the trading bot.

This module defines the :class:`BotConfig` dataclass along with helpers to
load configuration values from ``config.json`` and environment variables.
"""

import json
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List

# Load defaults from config.json
CONFIG_PATH = os.getenv(
    "CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.json")
)
try:
    with open(CONFIG_PATH, "r") as f:
        DEFAULTS = json.load(f)
except Exception:
    DEFAULTS = {}


def _get_default(key: str, fallback: Any) -> Any:
    return DEFAULTS.get(key, fallback)


@dataclass
class BotConfig:
    exchange: str = _get_default("exchange", "bybit")
    timeframe: str = _get_default("timeframe", "1m")
    secondary_timeframe: str = _get_default("secondary_timeframe", "2h")
    ws_url: str = _get_default("ws_url", "wss://stream.bybit.com/v5/public/linear")
    private_ws_url: str = _get_default(
        "private_ws_url", "wss://stream.bybit.com/v5/private"
    )
    backup_ws_urls: List[str] = field(
        default_factory=lambda: _get_default(
            "backup_ws_urls", ["wss://stream.bybit.com/v5/public/linear"]
        )
    )
    max_concurrent_requests: int = _get_default("max_concurrent_requests", 10)
    max_symbols: int = _get_default("max_symbols", 50)
    max_subscriptions_per_connection: int = _get_default(
        "max_subscriptions_per_connection", 15
    )
    ws_rate_limit: int = _get_default("ws_rate_limit", 20)
    ws_reconnect_interval: int = _get_default("ws_reconnect_interval", 5)
    max_reconnect_attempts: int = _get_default("max_reconnect_attempts", 10)
    latency_log_interval: int = _get_default("latency_log_interval", 3600)
    load_threshold: float = _get_default("load_threshold", 0.8)
    leverage: int = _get_default("leverage", 10)
    min_risk_per_trade: float = _get_default("min_risk_per_trade", 0.01)
    max_risk_per_trade: float = _get_default("max_risk_per_trade", 0.05)
    max_positions: int = _get_default("max_positions", 5)
    check_interval: int = _get_default("check_interval", 60)
    data_cleanup_interval: int = _get_default("data_cleanup_interval", 3600)
    base_probability_threshold: float = _get_default("base_probability_threshold", 0.6)
    trailing_stop_percentage: float = _get_default("trailing_stop_percentage", 1.0)
    trailing_stop_coeff: float = _get_default("trailing_stop_coeff", 1.0)
    retrain_threshold: float = _get_default("retrain_threshold", 0.1)
    retrain_volatility_threshold: float = _get_default(
        "retrain_volatility_threshold", 0.02
    )
    forget_window: int = _get_default("forget_window", 86400)
    trailing_stop_multiplier: float = _get_default("trailing_stop_multiplier", 1.0)
    tp_multiplier: float = _get_default("tp_multiplier", 2.0)
    sl_multiplier: float = _get_default("sl_multiplier", 1.0)
    kelly_win_prob: float = _get_default("kelly_win_prob", 0.6)
    min_sharpe_ratio: float = _get_default("min_sharpe_ratio", 0.5)
    performance_window: int = _get_default("performance_window", 86400)
    min_data_length: int = _get_default("min_data_length", 1000)
    lstm_timesteps: int = _get_default("lstm_timesteps", 60)
    lstm_batch_size: int = _get_default("lstm_batch_size", 32)
    model_type: str = _get_default("model_type", "cnn_lstm")
    nn_framework: str = _get_default("nn_framework", "pytorch")
    ema30_period: int = _get_default("ema30_period", 30)
    ema100_period: int = _get_default("ema100_period", 100)
    ema200_period: int = _get_default("ema200_period", 200)
    atr_period_default: int = _get_default("atr_period_default", 14)
    rsi_window: int = _get_default("rsi_window", 14)
    macd_window_slow: int = _get_default("macd_window_slow", 26)
    macd_window_fast: int = _get_default("macd_window_fast", 12)
    macd_window_sign: int = _get_default("macd_window_sign", 9)
    adx_window: int = _get_default("adx_window", 14)
    volume_profile_update_interval: int = _get_default(
        "volume_profile_update_interval", 300
    )
    model_save_path: str = _get_default("model_save_path", "/app/models")
    cache_dir: str = _get_default("cache_dir", "/app/cache")
    log_dir: str = _get_default("log_dir", "/app/logs")
    ray_num_cpus: int = _get_default("ray_num_cpus", 4)
    max_recovery_attempts: int = _get_default("max_recovery_attempts", 3)
    n_splits: int = _get_default("n_splits", 5)
    optimization_interval: int = _get_default("optimization_interval", 7200)
    shap_cache_duration: int = _get_default("shap_cache_duration", 86400)
    volatility_threshold: float = _get_default("volatility_threshold", 0.02)
    ema_crossover_lookback: int = _get_default("ema_crossover_lookback", 7200)
    pullback_period: int = _get_default("pullback_period", 3600)
    pullback_volatility_coeff: float = _get_default("pullback_volatility_coeff", 1.0)
    retrain_interval: int = _get_default("retrain_interval", 86400)
    min_liquidity: int = _get_default("min_liquidity", 1000000)
    ws_queue_size: int = _get_default("ws_queue_size", 10000)
    ws_min_process_rate: int = _get_default("ws_min_process_rate", 30)
    disk_buffer_size: int = _get_default("disk_buffer_size", 10000)
    prediction_history_size: int = _get_default("prediction_history_size", 100)
    telegram_queue_size: int = _get_default("telegram_queue_size", 100)
    optuna_trials: int = _get_default("optuna_trials", 20)
    enable_grid_search: bool = _get_default("enable_grid_search", False)
    loss_streak_threshold: int = _get_default("loss_streak_threshold", 3)
    win_streak_threshold: int = _get_default("win_streak_threshold", 3)
    threshold_adjustment: float = _get_default("threshold_adjustment", 0.05)
    target_change_threshold: float = _get_default("target_change_threshold", 0.001)
    backtest_interval: int = _get_default("backtest_interval", 604800)
    rl_model: str = _get_default("rl_model", "PPO")
    rl_framework: str = _get_default("rl_framework", "stable_baselines3")
    rl_timesteps: int = _get_default("rl_timesteps", 10000)
    mlflow_tracking_uri: str = _get_default("mlflow_tracking_uri", "mlruns")
    mlflow_enabled: bool = _get_default("mlflow_enabled", False)

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


def _convert(value: str, typ: type) -> Any:
    if typ is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    if typ is int:
        return int(value)
    if typ is float:
        return float(value)
    if typ is list or typ == List[str]:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [v.strip() for v in value.split(",") if v.strip()]
    return value


def load_config(path: str = CONFIG_PATH) -> BotConfig:
    """Load configuration from JSON file and environment variables."""
    cfg: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg.update(json.load(f))
    for fdef in fields(BotConfig):
        env_val = os.getenv(fdef.name.upper())
        if env_val is not None:
            cfg[fdef.name] = _convert(env_val, fdef.type)
    return BotConfig(**cfg)
