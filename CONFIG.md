# Configuration Reference

This document lists all configuration parameters available for the trading bot.
Each option corresponds to a field in `config.json` and the `BotConfig`
dataclass.  Default values are shown for reference.

List parameters supplied via environment variables must be provided either as
JSON arrays (e.g. `["ws://a", "ws://b"]`) or as comma-separated values such as
`ws://a,ws://b`.

The GPT analysis service should return JSON with the following fields:
`signal` ("buy"/"sell"/"hold"), `tp_mult` and `sl_mult` — multipliers applied
to take‑profit and stop‑loss distances.

| Parameter | Default | Description |
|-----------|---------|-------------|
| exchange | bybit | Trading exchange to connect to. |
| timeframe | 2h | Primary candle interval. |
| secondary_timeframe | 2h | Additional interval used for confirmation. |
| ws_url | wss://stream.bybit.com/v5/public/linear | Public WebSocket endpoint. |
| backup_ws_urls | ['wss://stream.bybit.com/v5/public/linear'] | Fallback WebSocket endpoints. |
| max_concurrent_requests | 10 | Limit for simultaneous HTTP requests. |
| max_volume_batch | 50 | Markets processed per volume fetch batch. |
| history_batch_size | 10 | Number of candles fetched per history request. |
| max_symbols | 50 | Maximum symbols to track. |
| max_subscriptions_per_connection | 15 | WebSocket subscriptions allowed per connection. |
| ws_subscription_batch_size | – | Symbols subscribed per batch when connecting; defaults to `max_subscriptions_per_connection`. |
| ws_rate_limit | 20 | WebSocket message rate limit. |
| ws_reconnect_interval | 5 | Seconds to wait before reconnecting. |
| max_reconnect_attempts | 10 | Maximum WebSocket reconnect attempts. |
| ws_inactivity_timeout | 30 | Seconds of inactivity before reconnecting. |
| latency_log_interval | 3600 | Interval for logging network latency. |
| load_threshold | 0.8 | CPU load threshold before throttling. |
| leverage | 10 | Default leverage for positions. |
| min_risk_per_trade | 0.01 | Minimum fraction of equity to risk per trade. |
| max_risk_per_trade | 0.05 | Maximum fraction of equity to risk per trade. |
| risk_sharpe_loss_factor | 0.5 | Risk reduction factor for poor Sharpe ratios. |
| risk_sharpe_win_factor | 1.5 | Risk increase factor for strong Sharpe ratios. |
| risk_vol_min | 0.5 | Lower bound for volatility based risk scaling. |
| risk_vol_max | 2.0 | Upper bound for volatility based risk scaling. |
| max_positions | 5 | Maximum simultaneous open positions. |
| top_signals | 5 | Number of top ranked signals to trade. |
| check_interval | 60 | Seconds between trade evaluation cycles. |
| data_cleanup_interval | 3600 | Interval for removing old cached data. |
| base_probability_threshold | 0.6 | Starting probability threshold for trades. |
| trailing_stop_percentage | 1.0 | Percent drop from peak price to trigger trailing stop. |
| trailing_stop_coeff | 1.0 | Multiplier for dynamic trailing stop distance. |
| retrain_threshold | 0.0 | Accuracy drop required to trigger retraining. |
| forget_window | 259200 | Seconds after which old data is discarded. |
| trailing_stop_multiplier | 1.0 | Multiplier applied to trailing stop distance. |
| tp_multiplier | 2.0 | Take‑profit distance relative to stop size. |
| sl_multiplier | 1.0 | Stop‑loss distance relative to stop size. |
| min_sharpe_ratio | 0.5 | Minimum acceptable Sharpe ratio. |
| performance_window | 86400 | Seconds of trade history used for performance metrics. |
| min_data_length | 200 | Minimum number of candles required to trade. |
| history_retention | 200 | Candles kept in memory per symbol. |
| lstm_timesteps | 60 | Sequence length fed into LSTM models. |
| lstm_batch_size | 32 | Training batch size for LSTM models. |
| model_type | transformer | Prediction model type. |
| nn_framework | pytorch | Neural network framework to use. |
| prediction_target | direction | Target variable for model outputs. |
| trading_fee | 0.0 | Trading fee applied to each trade. |
| ema30_period | 30 | Period for 30‑EMA indicator. |
| ema100_period | 100 | Period for 100‑EMA indicator. |
| ema200_period | 200 | Period for 200‑EMA indicator. |
| atr_period_default | 14 | Default Average True Range period. |
| rsi_window | 14 | Window for Relative Strength Index. |
| macd_window_slow | 26 | Slow EMA period for MACD. |
| macd_window_fast | 12 | Fast EMA period for MACD. |
| macd_window_sign | 9 | Signal line period for MACD. |
| adx_window | 14 | Window for Average Directional Index. |
| bollinger_window | 20 | Window for Bollinger Bands. |
| ulcer_window | 14 | Window for Ulcer Index. |
| volume_profile_update_interval | 300 | Seconds between volume profile updates. |
| funding_update_interval | 300 | Seconds between funding rate updates. |
| oi_update_interval | 300 | Seconds between open interest updates. |
| cache_dir | /app/cache | Directory for cached data. |
| log_dir | /app/logs | Directory for log files. |
| ray_num_cpus | 2 | CPU cores allocated to Ray tasks. |
| n_splits | 3 | Cross‑validation splits for model tuning. |
| optimization_interval | 7200 | Base interval between parameter optimizations. |
| shap_cache_duration | 86400 | Seconds to cache SHAP values. |
| volatility_threshold | 0.02 | Volatility level that triggers earlier optimization. |
| ema_crossover_lookback | 7200 | Lookback period for EMA crossover filter. |
| pullback_period | 3600 | Lookback window for pullback detection. |
| pullback_volatility_coeff | 1.0 | Volatility scaling for pullback logic. |
| retrain_interval | 86400 | Minimum interval between model retraining runs. |
| min_liquidity | 1000000 | Minimum quote volume for tradable symbols. |
| ws_queue_size | 10000 | Maximum size of WebSocket message queue. |
| ws_min_process_rate | 1 | Minimum messages per second considered healthy. |
| disk_buffer_size | 10000 | Items retained in on‑disk buffer. |
| prediction_history_size | 100 | Number of recent predictions stored. |
| telegram_queue_size | 100 | Maximum queued Telegram messages. |
| optuna_trials | 20 | Number of Optuna optimization trials. |
| optimizer_method | tpe | Optuna sampler method. |
| holdout_warning_ratio | 0.3 | Warn when holdout data exceeds this fraction. |
| enable_grid_search | False | Run additional GridSearch after Optuna. |
| loss_streak_threshold | 3 | Losses in a row before raising threshold. |
| win_streak_threshold | 3 | Wins in a row before lowering threshold. |
| threshold_adjustment | 0.05 | Probability adjustment applied per streak. |
| threshold_decay_rate | 0.1 | Rate at which threshold reverts to baseline. |
| target_change_threshold | 0.001 | Minimal price change to label as positive. |
| backtest_interval | 604800 | Seconds between automatic backtests. |
| rl_model | PPO | Reinforcement learning algorithm. |
| rl_framework | stable_baselines3 | Framework providing the RL agents. |
| rl_timesteps | 10000 | Timesteps used when training RL agent. |
| rl_use_imitation | False | Whether to use imitation learning in RL. |
| drawdown_penalty | 0.0 | Penalty coefficient for drawdowns in RL. |
| mlflow_tracking_uri | mlruns | MLflow tracking URI. |
| mlflow_enabled | False | Enable MLflow experiment logging. |
| use_strategy_optimizer | False | Optimize parameters at portfolio level. |
| portfolio_metric | sharpe | Metric used for portfolio optimization. |
| use_polars | False | Use Polars instead of pandas for data frames. |
| fine_tune_epochs | 5 | Epochs for fine‑tuning the model. |
| use_transfer_learning | False | Enable transfer learning for models. |
| order_retry_attempts | 3 | Number of times to retry failed orders. |
| order_retry_delay | 1.0 | Seconds to wait before retrying an order. |
| reversal_margin | 0.05 | Margin beyond opposite threshold to reverse trade. |
| transformer_weight | 0.5 | Ensemble weight for transformer model. |
| ema_weight | 0.2 | Ensemble weight for EMA strategy. |
| early_stopping_patience | 3 | Epochs with no improvement before early stop. |
| balance_key | – | Balance key for equity; defaults to symbol quote currency. |

