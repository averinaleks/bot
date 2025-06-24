from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List


@dataclass
class BotConfig:
    exchange: str = "bybit"
    timeframe: str = "1m"
    secondary_timeframe: str = "2h"
    ws_url: str = "wss://stream.bybit.com/v5/public/linear"
    private_ws_url: str = "wss://stream.bybit.com/v5/private"
    backup_ws_urls: List[str] = field(default_factory=lambda: [
        "wss://stream.bybit.com/v5/public/linear"
    ])
    max_concurrent_requests: int = 10
    max_symbols: int = 50
    max_subscriptions_per_connection: int = 15
    ws_rate_limit: int = 20
    ws_reconnect_interval: int = 5
    max_reconnect_attempts: int = 10
    latency_log_interval: int = 3600
    load_threshold: float = 0.8
    leverage: int = 10
    min_risk_per_trade: float = 0.01
    max_risk_per_trade: float = 0.05
    max_positions: int = 5
    check_interval: int = 60
    data_cleanup_interval: int = 3600
    base_probability_threshold: float = 0.6
    trailing_stop_percentage: float = 1.0
    trailing_stop_coeff: float = 1.0
    retrain_threshold: float = 0.1
    retrain_volatility_threshold: float = 0.02
    forget_window: int = 86400
    trailing_stop_multiplier: float = 1.0
    tp_multiplier: float = 2.0
    sl_multiplier: float = 1.0
    kelly_win_prob: float = 0.6
    min_sharpe_ratio: float = 0.5
    performance_window: int = 86400
    min_data_length: int = 1000
    lstm_timesteps: int = 60
    lstm_batch_size: int = 32
    model_type: str = "cnn_lstm"
    nn_framework: str = "pytorch"
    ema30_period: int = 30
    ema100_period: int = 100
    ema200_period: int = 200
    atr_period_default: int = 14
    rsi_window: int = 14
    macd_window_slow: int = 26
    macd_window_fast: int = 12
    macd_window_sign: int = 9
    adx_window: int = 14
    volume_profile_update_interval: int = 300
    model_save_path: str = "/app/models"
    cache_dir: str = "/app/cache"
    log_dir: str = "/app/logs"
    ray_num_cpus: int = 4
    max_recovery_attempts: int = 3
    n_splits: int = 5
    optimization_interval: int = 7200
    shap_cache_duration: int = 86400
    volatility_threshold: float = 0.02
    ema_crossover_lookback: int = 7200
    pullback_period: int = 3600
    pullback_volatility_coeff: float = 1.0
    retrain_interval: int = 86400
    min_liquidity: int = 1000000
    ws_queue_size: int = 10000
    ws_min_process_rate: int = 30
    disk_buffer_size: int = 10000
    prediction_history_size: int = 100
    telegram_queue_size: int = 100
    optuna_trials: int = 20
    enable_grid_search: bool = False
    loss_streak_threshold: int = 3
    win_streak_threshold: int = 3
    threshold_adjustment: float = 0.05
    target_change_threshold: float = 0.001
    backtest_interval: int = 604800
    rl_model: str = "PPO"
    rl_framework: str = "stable_baselines3"
    rl_timesteps: int = 10000
    mlflow_tracking_uri: str = "mlruns"
    mlflow_enabled: bool = False

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
            return [v.strip() for v in value.split(',') if v.strip()]
    return value


def load_config(path: str = "config.json") -> BotConfig:
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
