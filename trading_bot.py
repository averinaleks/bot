import asyncio
import json
import os
import signal
import sys
import torch
import ray
import psutil
import ccxt.async_support as ccxt_async
from telegram.ext import Application
from jsonschema import validate, ValidationError
from data_handler import DataHandler
from model_builder import ModelBuilder
from trade_manager import TradeManager
from optimizer import ParameterOptimizer
from model_builder import RLAgent
from utils import logger, TelegramLogger, check_dataframe_empty
import pandas as pd
import numpy as np
import importlib.metadata
import shutil

CONFIG_SCHEMA = {
    "type": "object",
    "required": [
        'exchange', 'timeframe', 'secondary_timeframe', 'ws_url', 'private_ws_url', 'backup_ws_urls',
        'max_concurrent_requests', 'max_symbols', 'ws_rate_limit',
        'ws_reconnect_interval', 'max_reconnect_attempts', 'latency_log_interval', 'load_threshold',
        'leverage', 'min_risk_per_trade', 'max_risk_per_trade',
        'max_positions', 'check_interval', 'data_cleanup_interval', 'base_probability_threshold',
        'trailing_stop_percentage', 'trailing_stop_coeff', 'trailing_stop_multiplier', 'tp_multiplier', 'sl_multiplier',
        'kelly_win_prob', 'min_sharpe_ratio', 'retrain_threshold', 'retrain_volatility_threshold',
        'performance_window', 'forget_window', 'min_data_length', 'lstm_timesteps',
        'lstm_batch_size', 'ema30_period', 'ema100_period', 'ema200_period',
        'atr_period_default', 'rsi_window', 'macd_window_slow', 'macd_window_fast', 'macd_window_sign', 'adx_window',
        'volume_profile_update_interval', 'model_save_path', 'cache_dir', 'log_dir',
        'ray_num_cpus', 'max_recovery_attempts', 'n_splits', 'optimization_interval', 'shap_cache_duration',
        'retrain_interval', 'volatility_threshold', 'ema_crossover_lookback', 'pullback_period',
        'pullback_volatility_coeff', 'min_liquidity', 'ws_queue_size', 'ws_min_process_rate',
        'max_subscriptions_per_connection',
        'disk_buffer_size', 'prediction_history_size', 'optuna_trials',
        'loss_streak_threshold', 'win_streak_threshold', 'threshold_adjustment',
        'target_change_threshold', 'backtest_interval'
    ],
    "properties": {
        "exchange": {"type": "string"},
        "timeframe": {"type": "string"},
        "secondary_timeframe": {"type": "string"},
        "ws_url": {"type": "string"},
        "private_ws_url": {"type": "string"},
        "backup_ws_urls": {"type": "array", "items": {"type": "string"}},
        "max_concurrent_requests": {"type": "integer", "minimum": 1},
        "max_symbols": {"type": "integer", "minimum": 1},
        "ws_rate_limit": {"type": "integer", "minimum": 1},
        "ws_reconnect_interval": {"type": "integer", "minimum": 1},
        "max_reconnect_attempts": {"type": "integer", "minimum": 1},
        "latency_log_interval": {"type": "integer", "minimum": 1},
        "load_threshold": {"type": "number", "minimum": 0},
        "leverage": {"type": "integer", "minimum": 1},
        "min_risk_per_trade": {"type": "number", "minimum": 0},
        "max_risk_per_trade": {"type": "number", "minimum": 0},
        "max_positions": {"type": "integer", "minimum": 1},
        "check_interval": {"type": "integer", "minimum": 1},
        "data_cleanup_interval": {"type": "integer", "minimum": 1},
        "base_probability_threshold": {"type": "number", "minimum": 0},
        "trailing_stop_multiplier": {"type": "number", "minimum": 0},
        "tp_multiplier": {"type": "number", "minimum": 0},
        "sl_multiplier": {"type": "number", "minimum": 0},
        "kelly_win_prob": {"type": "number", "minimum": 0},
        "min_sharpe_ratio": {"type": "number", "minimum": 0},
        "trailing_stop_percentage": {"type": "number", "minimum": 0},
        "trailing_stop_coeff": {"type": "number", "minimum": 0},
        "retrain_threshold": {"type": "number", "minimum": 0},
        "retrain_volatility_threshold": {"type": "number", "minimum": 0},
        "performance_window": {"type": "integer", "minimum": 1},
        "forget_window": {"type": "integer", "minimum": 1},
        "min_data_length": {"type": "integer", "minimum": 1},
        "lstm_timesteps": {"type": "integer", "minimum": 1},
        "lstm_batch_size": {"type": "integer", "minimum": 1},
        "ema30_period": {"type": "integer", "minimum": 1},
        "ema100_period": {"type": "integer", "minimum": 1},
        "ema200_period": {"type": "integer", "minimum": 1},
        "atr_period_default": {"type": "integer", "minimum": 1},
        "rsi_window": {"type": "integer", "minimum": 1},
        "macd_window_slow": {"type": "integer", "minimum": 1},
        "macd_window_fast": {"type": "integer", "minimum": 1},
        "macd_window_sign": {"type": "integer", "minimum": 1},
        "adx_window": {"type": "integer", "minimum": 1},
        "volume_profile_update_interval": {"type": "integer", "minimum": 1},
        "model_save_path": {"type": "string"},
        "cache_dir": {"type": "string"},
        "log_dir": {"type": "string"},
        "ray_num_cpus": {"type": "integer", "minimum": 1},
        "max_recovery_attempts": {"type": "integer", "minimum": 1},
        "n_splits": {"type": "integer", "minimum": 1},
        "optimization_interval": {"type": "integer", "minimum": 1},
        "shap_cache_duration": {"type": "integer", "minimum": 1},
        "retrain_interval": {"type": "integer", "minimum": 1},
        "volatility_threshold": {"type": "number", "minimum": 0},
        "ema_crossover_lookback": {"type": "integer", "minimum": 1},
        "pullback_period": {"type": "integer", "minimum": 1},
        "pullback_volatility_coeff": {"type": "number", "minimum": 0},
        "min_liquidity": {"type": "integer", "minimum": 1},
        "ws_queue_size": {"type": "integer", "minimum": 1},
        "ws_min_process_rate": {"type": "integer", "minimum": 1},
        "max_subscriptions_per_connection": {"type": "integer", "minimum": 1},
        "disk_buffer_size": {"type": "integer", "minimum": 1},
        "prediction_history_size": {"type": "integer", "minimum": 1},
        "optuna_trials": {"type": "integer", "minimum": 1},
        "loss_streak_threshold": {"type": "integer", "minimum": 1},
        "win_streak_threshold": {"type": "integer", "minimum": 1},
        "threshold_adjustment": {"type": "number", "minimum": 0},
        "target_change_threshold": {"type": "number", "minimum": 0},
        "backtest_interval": {"type": "integer", "minimum": 1}
    }
}

async def monitor_resources(telegram_bot, chat_id, interval=300):
    # Мониторинг системных ресурсов
    try:
        while True:
            cpu_percent = await asyncio.get_event_loop().run_in_executor(None, psutil.cpu_percent, 1)
            memory = await asyncio.get_event_loop().run_in_executor(None, psutil.virtual_memory)
            memory_percent = memory.percent
            gpu_usage = 0
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    gpu_usage = (allocated / max_allocated * 100) if max_allocated > 0 else 0
                except Exception as e:
                    logger.warning(f"Ошибка получения статистики GPU: {e}")
            message = f"⚠️ Нагрузка: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, GPU {gpu_usage:.1f}%"
            if cpu_percent > 90 or memory_percent > 90 or gpu_usage > 90:
                await TelegramLogger(telegram_bot, chat_id).send_telegram_message(message)
                logger.warning(message)
            await asyncio.sleep(interval)
    except Exception as e:
        logger.error(f"Ошибка мониторинга ресурсов: {e}")
        await TelegramLogger(telegram_bot, chat_id).send_telegram_message(f"Ошибка мониторинга ресурсов: {e}")

async def check_library_versions(telegram_bot, chat_id):
    # Проверка версий используемых библиотек
    try:
        versions = {
            "ccxt": importlib.metadata.version("ccxt"),
            "torch": torch.__version__,
            "ray": importlib.metadata.version("ray"),
            "optuna": importlib.metadata.version("optuna"),
            "shap": importlib.metadata.version("shap"),
            "pandas": importlib.metadata.version("pandas"),
            "numpy": importlib.metadata.version("numpy"),
        }
        logger.info(f"Версии библиотек: {versions}")
        await TelegramLogger(telegram_bot, chat_id).send_telegram_message(f"Версии библиотек: {versions}")
    except Exception as e:
        logger.error(f"Ошибка проверки версий библиотек: {e}")
        await TelegramLogger(telegram_bot, chat_id).send_telegram_message(f"Ошибка проверки версий библиотек: {e}")

async def main():
    telegram_bot = None
    exchange = None
    application = None
    telegram_logger = None
    try:
        config_path = os.getenv('CONFIG_PATH', '/app/config.json')
        if not os.path.exists(config_path):
            logger.error(f"Файл конфигурации не найден: {config_path}")
            return
        with open(config_path, 'r') as f:
            config = json.load(f)
        try:
            validate(instance=config, schema=CONFIG_SCHEMA)
        except ValidationError as e:
            logger.error(f"Некорректный формат config.json: {e.message}")
            return
        if config['exchange'] != 'bybit':
            logger.error(f"Ожидалась биржа 'bybit', найдено: {config['exchange']}")
            return

        cache_dir = config['cache_dir']
        disk_usage = shutil.disk_usage(cache_dir)
        if disk_usage.free / disk_usage.total < 0.1:
            logger.error(f"Недостаточно свободного места на диске в {cache_dir}: {disk_usage.free / (1024 ** 3):.2f} ГБ")
            return

        for dir_key in ['model_save_path', 'cache_dir', 'log_dir']:
            dir_path = config[dir_key]
            os.makedirs(dir_path, exist_ok=True)
            if not os.path.exists(dir_path):
                logger.error(f"Директория {dir_key} не существует: {dir_path}")
                return
            if not os.access(dir_path, os.W_OK):
                logger.error(f"Нет прав на запись в директорию {dir_key}: {dir_path}")
                return

        gpu_available = torch.cuda.is_available()
        if not gpu_available:
            logger.warning("GPU недоступен, используется только CPU")
        ray_num_cpus = config.get('ray_num_cpus', 4)
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        ray_memory = max(2, min(available_memory * 0.5, 8))
        ray.init(
            num_cpus=ray_num_cpus,
            num_gpus=1 if gpu_available else 0,
            object_store_memory=int(ray_memory * 1024 ** 3 * 0.5),
            ignore_reinit_error=True
        )
        logger.info(f"Ray инициализирован с {ray_num_cpus} CPU, {ray_memory:.2f} ГБ памяти, GPU: {gpu_available}")

        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if not telegram_token or not chat_id:
            logger.error("TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не установлены")
            return
        application = Application.builder().token(telegram_token).build()
        telegram_bot = application.bot
        telegram_logger = TelegramLogger(telegram_bot, chat_id)
        await check_library_versions(telegram_bot, chat_id)

        exchange_config = {
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_API_SECRET'),
            'enableRateLimit': True,
            'asyncio_loop': asyncio.get_event_loop()
        }
        exchange = ccxt_async.bybit(exchange_config)

        data_handler = DataHandler(config, exchange, telegram_bot, chat_id)
        rl_agent = RLAgent(config, data_handler, None)
        trade_manager = TradeManager(config, data_handler, None, telegram_bot, chat_id, rl_agent)
        model_builder = ModelBuilder(config, data_handler, trade_manager)
        rl_agent.model_builder = model_builder
        trade_manager.model_builder = model_builder
        parameter_optimizer = ParameterOptimizer(config, data_handler)
        data_handler.parameter_optimizer = parameter_optimizer

        await data_handler.load_initial()

        shutdown_event = asyncio.Event()

        async def handle_shutdown(loop):
            # Обработка graceful shutdown
            logger.info("Получен сигнал завершения, инициируется graceful shutdown")
            shutdown_event.set()
            tasks = [task for task in asyncio.all_tasks(loop) if task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            for symbol in trade_manager.positions['symbol'].unique():
                df = data_handler.ohlcv.xs(symbol, level='symbol', drop_level=False)
                if not check_dataframe_empty(df, f"shutdown {symbol}"):
                    await trade_manager.close_position(symbol, df['close'].iloc[-1], "Shutdown")
            await asyncio.gather(
                loop.shutdown_asyncgens(),
                exchange.close(),
                return_exceptions=True
            )
            ray.shutdown()
            if gpu_available:
                torch.cuda.empty_cache()
            logger.info("Программа завершена")
            sys.exit(0)

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(handle_shutdown(loop)))

        tasks = [
            asyncio.create_task(data_handler.subscribe_to_klines(data_handler.usdt_pairs), name="subscribe_to_klines"),
            asyncio.create_task(model_builder.train(), name="train_models"),
            asyncio.create_task(rl_agent.train(), name="train_rl"),
            asyncio.create_task(trade_manager.run(), name="trade_manager_run"),
            asyncio.create_task(
                optimize_parameters_periodically(
                    parameter_optimizer,
                    telegram_bot,
                    chat_id,
                    shutdown_event,
                    interval=config['optimization_interval'] // 2,
                ),
                name="optimize_parameters",
            ),
            asyncio.create_task(monitor_resources(telegram_bot, chat_id), name="monitor_resources"),
            asyncio.create_task(
                monitor_model_performance(
                    model_builder, telegram_bot, chat_id, shutdown_event, interval=3600
                ),
                name="monitor_model_performance",
            ),
            asyncio.create_task(
                run_backtests_periodically(
                    model_builder,
                    telegram_bot,
                    chat_id,
                    shutdown_event,
                    interval=config.get('backtest_interval', 604800),
                ),
                name="run_backtests",
            ),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for task, result in zip(tasks, results):
            if isinstance(result, asyncio.CancelledError):
                logger.info(f"Задача {task.get_name()} отменена")
                continue
            if isinstance(result, Exception):
                logger.error(f"Ошибка в задаче {task.get_name()}: {result}")
                await telegram_logger.send_telegram_message(
                    f"❌ Ошибка в задаче {task.get_name()}: {result}")
                raise result

    except Exception as e:
        logger.error(f"Критическая ошибка в main: {e}")
        if telegram_bot and chat_id:
            await telegram_logger.send_telegram_message(f"Критическая ошибка: {e}")
        raise
    finally:
        if application:
            await application.shutdown()
        if exchange:
            await exchange.close()
        ray.shutdown()
        if gpu_available:
            torch.cuda.empty_cache()
        logger.info("Программа завершена")

async def optimize_parameters_periodically(parameter_optimizer, telegram_bot, chat_id, shutdown_event: asyncio.Event, interval: int = 7200):
    # Периодическая оптимизация параметров с уменьшенным интервалом
    try:
        while not shutdown_event.is_set():
            min_interval = interval
            for symbol in parameter_optimizer.data_handler.usdt_pairs:
                logger.info(f"Оптимизация параметров для {symbol}")
                best_params = await parameter_optimizer.optimize(symbol)
                if best_params:
                    logger.info(f"Обновлены параметры для {symbol}: {best_params}")
                    await TelegramLogger(telegram_bot, chat_id).send_telegram_message(
                        f"Параметры оптимизированы для {symbol}: {best_params}"
                    )
                indicators = parameter_optimizer.data_handler.indicators.get(symbol)
                volatility = indicators.volatility if indicators else parameter_optimizer.volatility_threshold
                symbol_interval = parameter_optimizer.get_opt_interval(symbol, volatility)
                min_interval = min(min_interval, symbol_interval)
            await asyncio.sleep(min_interval)
    except Exception as e:
        logger.error(f"Ошибка в периодической оптимизации: {e}")
        await TelegramLogger(telegram_bot, chat_id).send_telegram_message(f"Ошибка оптимизации: {e}")

async def monitor_model_performance(model_builder, telegram_bot, chat_id, shutdown_event: asyncio.Event, interval: int = 3600):
    # Мониторинг производительности моделей и инициирование переобучения
    try:
        while not shutdown_event.is_set():
            for symbol in model_builder.data_handler.usdt_pairs:
                indicators = model_builder.data_handler.indicators.get(symbol)
                if not indicators:
                    continue
                volatility = indicators.volatility
                volatility_change = abs(volatility - model_builder.trade_manager.last_volatility.get(symbol, 0.0)) / max(model_builder.trade_manager.last_volatility.get(symbol, 0.01), 0.01)
                returns = model_builder.trade_manager.returns_by_symbol.get(symbol, [])
                current_time = pd.Timestamp.now(tz='UTC').timestamp()
                recent_returns = [r for t, r in returns if current_time - t <= model_builder.config['performance_window']]
                sharpe_ratio = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6) * np.sqrt(365 * 24 * 60 / model_builder.config['performance_window']) if recent_returns else 0.0
                if sharpe_ratio < model_builder.config.get('min_sharpe_ratio', 0.5) or volatility_change > 0.5:
                    logger.info(f"Инициировано переобучение для {symbol}: Sharpe={sharpe_ratio:.2f}, Изменение волатильности={volatility_change:.2f}")
                    await model_builder.retrain_symbol(symbol)
                    await TelegramLogger(telegram_bot, chat_id).send_telegram_message(
                        f"🔄 Переобучение для {symbol}: Sharpe={sharpe_ratio:.2f}, Волатильность={volatility_change:.2f}"
                    )
            await asyncio.sleep(interval)
    except Exception as e:
        logger.error(f"Ошибка мониторинга производительности моделей: {e}")
        await TelegramLogger(telegram_bot, chat_id).send_telegram_message(f"Ошибка мониторинга моделей: {e}")

async def run_backtests_periodically(model_builder, telegram_bot, chat_id, shutdown_event: asyncio.Event, interval: int = 604800):
    try:
        while not shutdown_event.is_set():
            results = await model_builder.backtest_all()
            if results:
                msg = ", ".join(f"{s}:{v:.2f}" for s, v in results.items())
                await TelegramLogger(telegram_bot, chat_id).send_telegram_message(
                    f"📊 Backtest Sharpe: {msg}")
            await asyncio.sleep(interval)
    except Exception as e:
        logger.error(f"Ошибка бектеста: {e}")
        await TelegramLogger(telegram_bot, chat_id).send_telegram_message(f"Ошибка бектеста: {e}")

async def run_bot():
    restart_attempts = 0
    max_restart_attempts = 3
    while True:
        try:
            await main()
            break
        except Exception as e:
            restart_attempts += 1
            if restart_attempts >= max_restart_attempts:
                logger.critical("Превышено максимальное количество попыток перезапуска")
                raise
            logger.info(f"Попытка перезапуска {restart_attempts}/{max_restart_attempts}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(run_bot())
