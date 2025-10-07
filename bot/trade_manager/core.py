"""Trading engine for executing and managing positions.

This module coordinates order placement, risk management and Telegram
notifications while interacting with the :class:`ModelBuilder` and exchange.
"""


import asyncio
import atexit
import signal
import os
import types
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Tuple,
    TypeAlias,
    cast,
)
import shutil
import numpy as np
from bot import test_stubs
from bot.dotenv_utils import load_dotenv
from bot.ray_compat import ray  # noqa: E402
import httpx  # noqa: E402
import inspect  # noqa: E402
from bot.utils_loader import require_utils  # noqa: E402
from bot.trade_manager import order_utils  # noqa: E402
from .errors import TradeManagerTaskError

_utils = require_utils(
    "logger",
    "is_cuda_available",
    "check_dataframe_empty_async",
    "safe_api_call",
    "TelegramLogger",
)

logger = _utils.logger
is_cuda_available = _utils.is_cuda_available
_check_df_async = _utils.check_dataframe_empty_async
safe_api_call = _utils.safe_api_call
TelegramLogger = _utils.TelegramLogger

test_stubs.apply()


def _is_test_mode_enabled() -> bool:
    """Return ``True`` when the lightweight test stubs should be active."""

    try:
        if getattr(test_stubs, "IS_TEST_MODE", False):
            return True
    except Exception:  # pragma: no cover - defensive guard for exotic stubs
        pass
    return os.getenv("TEST_MODE") == "1"


class _TestModeFlag:
    """Boolean-like proxy that reflects the current test mode."""

    def __bool__(self) -> bool:
        return _is_test_mode_enabled()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return str(_is_test_mode_enabled())

    __str__ = __repr__

    def __int__(self) -> int:  # pragma: no cover - compatibility helper
        return int(_is_test_mode_enabled())


IS_TEST_MODE = _TestModeFlag()


def is_test_mode_enabled() -> bool:
    """Public helper returning whether the trade manager runs in test mode."""

    return bool(IS_TEST_MODE)

aiohttp: Any
try:  # pragma: no cover - optional dependency
    import aiohttp as _aiohttp
except Exception:  # pragma: no cover - minimal stub
    aiohttp = types.SimpleNamespace(ClientError=Exception)
else:
    aiohttp = _aiohttp

pd: Any
try:  # pragma: no cover - optional dependency
    import pandas as _pd
except ImportError as exc:  # allow missing pandas
    logging.getLogger(__name__).warning("pandas import failed: %s", exc)
    pd = types.SimpleNamespace(
        DataFrame=dict,
        Series=list,
        MultiIndex=types.SimpleNamespace(from_arrays=lambda *a, **k: []),
    )
else:
    pd = _pd

if TYPE_CHECKING:
    from telegram_logger import TelegramLogger as TelegramLoggerType
else:  # pragma: no cover - type-checking aid
    TelegramLoggerType = Any

# ``configure_logging`` может отсутствовать в тестовых заглушках
try:  # pragma: no cover - fallback для тестов
    from bot.utils import configure_logging  # noqa: E402
except ImportError:  # pragma: no cover - заглушка
    def configure_logging() -> None:
        """Stubbed logging configurator."""
        pass
from bot import config as bot_config  # noqa: E402

BotConfig: TypeAlias = bot_config.BotConfig
from services import stubs as service_stubs  # noqa: E402
from services.logging_utils import sanitize_log_value  # noqa: E402
from telegram_logger import resolve_unsent_path  # noqa: E402
import contextlib  # noqa: E402
from bot.http_client import close_async_http_client as close_http_client  # noqa: E402
from services.offline import (  # noqa: E402
    ensure_offline_env,
    generate_placeholder_credential,
)

_httpx: Any
_offline_intent = service_stubs.is_offline_env()
try:  # pragma: no cover - bot_config may lack OFFLINE_MODE in unusual setups
    _offline_intent = _offline_intent or bool(getattr(bot_config, "OFFLINE_MODE", False))
except Exception:  # pragma: no cover - defensive guard
    pass

if _offline_intent:
    _httpx = service_stubs.create_httpx_stub()
else:
    try:  # pragma: no cover - optional dependency
        import httpx as _imported_httpx  # type: ignore  # noqa: E402
    except Exception:  # pragma: no cover - fallback to stub on import issues
        _httpx = service_stubs.create_httpx_stub()
    else:
        _httpx = cast(Any, _imported_httpx)

httpx = cast(Any, _httpx)
HTTPError = httpx.HTTPError

torch: Any
try:  # pragma: no cover - optional dependency
    import torch as _torch  # noqa: E402
except ImportError as exc:  # optional dependency may not be installed
    logging.getLogger(__name__).warning("torch import failed: %s", exc)

    def _tensor(*args: Any, **kwargs: Any) -> Any:
        return args[0]

    torch = types.SimpleNamespace(
        tensor=_tensor,
        float32=float,
        no_grad=contextlib.nullcontext,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        amp=types.SimpleNamespace(
            autocast=lambda *a, **k: contextlib.nullcontext()
        ),
    )
else:
    torch = cast(Any, _torch)
import multiprocessing as mp  # noqa: E402


@dataclass(frozen=True)
class _TaskSpec:
    """Описание фоновой задачи TradeManager."""

    name: str
    factory: Callable[[], Awaitable[Any]]
    critical: bool
    restart: bool = False


def setup_multiprocessing() -> None:
    """Ensure multiprocessing uses the 'spawn' start method."""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

# Determine computation device once

device_type = "cuda" if is_cuda_available() else "cpu"

_HOSTNAME_RE = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z0-9-]{1,63})*$")


class InvalidHostError(ValueError):
    pass


def _predict_model(model, tensor) -> np.ndarray:
    """Run model forward pass."""
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type):
        return model(tensor).squeeze().float().cpu().numpy()


@ray.remote(num_cpus=1, num_gpus=1 if is_cuda_available() else 0)
def _predict_model_proc(model, tensor) -> np.ndarray:
    """Execute ``_predict_model`` in a separate process."""
    return _predict_model(model, tensor)


async def _predict_async(model, tensor) -> np.ndarray:
    """Asynchronously run the prediction process and return the result."""
    obj_ref = _predict_model_proc.remote(model, tensor)
    return await asyncio.to_thread(ray.get, obj_ref)


def _calibrate_output(calibrator, value: float) -> float:
    """Run calibrator prediction in a worker thread."""
    return calibrator.predict_proba([[value]])[0, 1]


def _register_cleanup_handlers(tm: "TradeManager") -> None:
    """Register atexit and signal handlers for graceful shutdown."""

    if getattr(tm, "_cleanup_registered", False):
        return

    setattr(tm, "_cleanup_registered", True)

    def _handler(*_args):
        logger.info("Остановка TradeManager")
        tm.shutdown()
        try:

            asyncio.run(TelegramLogger.shutdown())
        except RuntimeError:
            # event loop may already be closed
            pass
        listener = getattr(tm, "_listener", None)
        if listener is not None:
            listener.stop()

    atexit.register(_handler)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda s, f: _handler())
        except ValueError:
            # signal may fail if not in main thread
            pass


class TradeManager:
    """Handles trading logic and sends Telegram notifications.

    Parameters
    ----------
    config : dict
        Bot configuration.
    data_handler : DataHandler
        Instance providing market data.
    model_builder
        Associated ModelBuilder instance.
    telegram_bot : telegram.Bot or compatible
        Bot used to send messages.
    chat_id : str | int
        Telegram chat identifier.
    rl_agent : optional
        Reinforcement learning agent used for decisions.
    """

    telegram_logger: TelegramLoggerType | types.SimpleNamespace

    def __init__(
        self,
        config: BotConfig,
        data_handler,
        model_builder,
        telegram_bot,
        chat_id,
        rl_agent=None,
        *,
        telegram_logger_factory: Callable[..., TelegramLoggerType] | None = None,
        gpt_client_factory=None,
    ):
        # Ensure environment variables from optional .env file are available
        # before we read Telegram-related settings. ``load_dotenv`` is a no-op
        # when python-dotenv isn't installed, so calling it is safe in tests.
        load_dotenv()

        if bot_config.OFFLINE_MODE:
            ensure_offline_env(
                {
                    "TELEGRAM_BOT_TOKEN": lambda: generate_placeholder_credential(
                        "telegram-token"
                    ),
                    "TELEGRAM_CHAT_ID": lambda: generate_placeholder_credential(
                        "telegram-chat"
                    ),
                }
            )

        self.config = config
        self.data_handler = data_handler
        self.model_builder = model_builder
        self.rl_agent = rl_agent
        self.telegram_logger_factory = telegram_logger_factory
        self.gpt_client_factory = gpt_client_factory
        if (
            not config.enable_notifications
            or not os.environ.get("TELEGRAM_BOT_TOKEN")
            or not os.environ.get("TELEGRAM_CHAT_ID")
        ):
            logger.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; Telegram alerts will not be sent"
            )
            async def _noop(*_, **__):
                pass

            self.telegram_logger = types.SimpleNamespace(
                info=lambda *a, **k: None,
                warning=lambda *a, **k: None,
                send_telegram_message=_noop,
            )
        else:
            unsent_path = None
            if config.save_unsent_telegram:
                try:
                    unsent_path = str(
                        resolve_unsent_path(
                            config.log_dir, config.unsent_telegram_path
                        )
                    )
                except ValueError as exc:
                    logger.warning(
                        "Ignoring unsafe unsent_telegram_path %s: %s",
                        sanitize_log_value(config.unsent_telegram_path),
                        exc,
                    )
                    unsent_path = None
            logger_cls = telegram_logger_factory or TelegramLogger
            try:
                self.telegram_logger = cast(
                    TelegramLoggerType,
                    logger_cls(
                        telegram_bot,
                        chat_id,
                        max_queue_size=config.get("telegram_queue_size"),
                        unsent_path=unsent_path,
                    ),
                )
            except TypeError:  # pragma: no cover - stub without args
                stub_factory = cast(Callable[[], TelegramLoggerType], logger_cls)
                self.telegram_logger = stub_factory()
            if not hasattr(self.telegram_logger, "send_telegram_message"):
                async def _noop(*a, **k):
                    pass

                setattr(self.telegram_logger, "send_telegram_message", _noop)
        self.positions = self._init_positions_frame()
        self.returns_by_symbol: dict[str, list[tuple[float, float]]] = (
            self._init_returns_state()
        )
        self.position_lock = asyncio.Lock()
        self.returns_lock = asyncio.Lock()
        self.tasks: list[asyncio.Task] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self._failure_notified = False
        self._critical_error = False
        self.exchange = data_handler.exchange
        self.max_positions = config.get("max_positions", 5)
        self.top_signals = config.get("top_signals", self.max_positions)
        self.leverage = config.get("leverage", 10)
        self.max_position_pct = config.get("max_position_pct", 0.1)
        self.min_risk_per_trade = config.get("min_risk_per_trade", 0.01)
        self.max_risk_per_trade = config.get("max_risk_per_trade", 0.05)
        self.check_interval = config.get("check_interval", 60.0)
        self.performance_window = config.get("performance_window", 86400)
        self.state_file = os.path.join(config["cache_dir"], "trade_manager_state.parquet")
        self.returns_file = os.path.join(
            config["cache_dir"], "trade_manager_returns.json"
        )
        self.last_save_time = time.time()
        self.save_interval = 900
        self.positions_changed = False
        self.last_volatility = {symbol: 0.0 for symbol in data_handler.usdt_pairs}
        self.last_stats_day = int(time.time() // 86400)
        self._min_retrain_size: dict[str, int] = {}
        self.load_state()
        self.http_client = None

    def _init_positions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "position",
                "size",
                "entry_price",
                "tp_multiplier",
                "sl_multiplier",
                "stop_loss_price",
                "highest_price",
                "lowest_price",
                "breakeven_triggered",
                "last_checked_ts",
                "last_trailing_ts",
            ],
            index=pd.MultiIndex.from_arrays(
                [pd.Index([], dtype=object), pd.DatetimeIndex([], tz="UTC")],
                names=["symbol", "timestamp"],
            ),
        )

    def _init_returns_state(self) -> dict[str, list[tuple[float, float]]]:
        pairs = getattr(self.data_handler, "usdt_pairs", [])
        return {symbol: [] for symbol in pairs}

    def _has_position(self, symbol: str) -> bool:
        """Check if a position for ``symbol`` exists using the MultiIndex."""
        return (
            "symbol" in self.positions.index.names
            and symbol in self.positions.index.get_level_values("symbol")
        )

    async def compute_risk_per_trade(self, symbol: str, volatility: float) -> float:
        base_risk = self.config.get("risk_per_trade", self.min_risk_per_trade)
        async with self.returns_lock:
            returns = [
                r
                for t, r in self.returns_by_symbol.get(symbol, [])
                if time.time() - t <= self.performance_window
            ]
        sharpe = (
            np.mean(returns)
            / (np.std(returns) + 1e-6)
            * np.sqrt(365 * 24 * 60 * 60 / self.performance_window)
            if returns
            else 0.0
        )
        if sharpe < 0:
            base_risk *= self.config.get("risk_sharpe_loss_factor", 0.5)
        elif sharpe > 1:
            base_risk *= self.config.get("risk_sharpe_win_factor", 1.5)
        threshold = max(self.config.get("volatility_threshold", 0.02), 1e-6)
        vol_coeff = volatility / threshold
        vol_coeff = max(
            self.config.get("risk_vol_min", 0.5),
            min(self.config.get("risk_vol_max", 2.0), vol_coeff),
        )
        base_risk *= vol_coeff
        return min(self.max_risk_per_trade, max(self.min_risk_per_trade, base_risk))

    async def get_sharpe_ratio(self, symbol: str) -> float:
        async with self.returns_lock:
            returns = [
                r
                for t, r in self.returns_by_symbol.get(symbol, [])
                if time.time() - t <= self.performance_window
            ]
        if not returns:
            return 0.0
        return (
            np.mean(returns)
            / (np.std(returns) + 1e-6)
            * np.sqrt(365 * 24 * 60 * 60 / self.performance_window)
        )

    async def get_loss_streak(self, symbol: str) -> int:
        async with self.returns_lock:
            returns = [r for _, r in self.returns_by_symbol.get(symbol, [])]
        count = 0
        for r in reversed(returns):
            if r < 0:
                count += 1
            else:
                break
        return count

    async def get_win_streak(self, symbol: str) -> int:
        async with self.returns_lock:
            returns = [r for _, r in self.returns_by_symbol.get(symbol, [])]
        count = 0
        for r in reversed(returns):
            if r > 0:
                count += 1
            else:
                break
        return count

    async def compute_stats(self) -> Dict[str, float]:
        """Return overall win rate, average profit/loss and max drawdown."""
        async with self.returns_lock:
            all_returns = [r for vals in self.returns_by_symbol.values() for _, r in vals]
        total = len(all_returns)
        win_rate = sum(1 for r in all_returns if r > 0) / total if total else 0.0
        avg_pnl = float(np.mean(all_returns)) if all_returns else 0.0
        if all_returns:
            cum = np.cumsum(all_returns)
            running_max = np.maximum.accumulate(cum)
            drawdowns = running_max - cum
            max_dd = float(np.max(drawdowns))
        else:
            max_dd = 0.0
        return {
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "max_drawdown": max_dd,
        }

    def get_stats(self) -> Dict[str, float]:
        """Synchronous wrapper for :py:meth:`compute_stats`."""
        if self.loop and self.loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.compute_stats(), self.loop)
            return fut.result()
        return asyncio.run(self.compute_stats())

    def save_state(self):
        if not self.positions_changed or (
            time.time() - self.last_save_time < self.save_interval
        ):
            return
        try:
            os.makedirs(self.config["cache_dir"], exist_ok=True)
            disk_usage = shutil.disk_usage(self.config["cache_dir"])
            if disk_usage.free / (1024**3) < 0.5:
                logger.warning(
                    "Not enough space to persist state: %.2f GB left",
                    disk_usage.free / (1024 ** 3),
                )
                return
            tmp_state = f"{self.state_file}.tmp"
            tmp_returns = f"{self.returns_file}.tmp"
            try:
                self.positions.to_parquet(tmp_state)
            except Exception as exc:  # pragma: no cover - optional deps missing
                logger.warning(
                    "Parquet support unavailable, falling back to JSON: %s",
                    exc,
                )
                # ``orient='split'`` cannot be round-tripped with MultiIndex in
                # recent pandas versions. Serialize as records and rebuild the
                # index on load instead to avoid ``NotImplementedError``.
                self.positions.drop(columns="symbol", errors="ignore").reset_index().to_json(
                    tmp_state, orient="records", date_format="iso"
                )
            with open(tmp_returns, "w", encoding="utf-8") as f:
                json.dump(self.returns_by_symbol, f)
            os.replace(tmp_state, self.state_file)
            os.replace(tmp_returns, self.returns_file)
            self.last_save_time = time.time()
            self.positions_changed = False
            logger.info("Состояние TradeManager сохранено")
        except (OSError, ValueError) as e:
            logger.exception("Failed to save state (%s): %s", type(e).__name__, e)
            for path in (locals().get("tmp_state"), locals().get("tmp_returns")):
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except OSError as cleanup_err:
                    logger.exception("Не удалось удалить временный файл %s: %s", path, cleanup_err)
            raise

    def load_state(self):
        corrupted: list[str] = []
        state_loaded = False
        returns_loaded = False
        try:
            if os.path.exists(self.state_file):
                try:
                    self.positions = pd.read_parquet(self.state_file)
                except Exception:
                    df = pd.read_json(self.state_file, orient="records")
                    if not df.empty:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                        self.positions = df.set_index(["symbol", "timestamp"])
                    else:
                        self.positions = df
                if "timestamp" in self.positions.index.names:
                    ts_level = self.positions.index.get_level_values("timestamp")
                    if ts_level.tz is None:
                        self.positions = (
                            self.positions
                            .tz_localize("UTC", level="timestamp")
                            .tz_convert("UTC", level="timestamp")
                        )
                    else:
                        self.positions = self.positions.tz_convert(
                            "UTC", level="timestamp"
                        )
                self._sort_positions()
                if "last_trailing_ts" not in self.positions.columns:
                    self.positions["last_trailing_ts"] = pd.NaT
                state_loaded = True
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            corrupted.append(
                self._handle_corrupted_state_file(self.state_file, exc)
            )
            self.positions = self._init_positions_frame()

        try:
            if os.path.exists(self.returns_file):
                with open(self.returns_file, "r", encoding="utf-8") as f:
                    self.returns_by_symbol = json.load(f)
                returns_loaded = True
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            corrupted.append(
                self._handle_corrupted_state_file(self.returns_file, exc)
            )
            self.returns_by_symbol = self._init_returns_state()

        self.returns_by_symbol = {
            **self._init_returns_state(),
            **self.returns_by_symbol,
        }

        if returns_loaded or state_loaded:
            logger.info("Состояние TradeManager загружено")

        if corrupted:
            self._dispatch_state_reset_notice([c for c in corrupted if c])

    def _sort_positions(self) -> None:
        """Ensure positions are sorted by symbol then timestamp."""
        if not self.positions.empty:
            self.positions.sort_index(level=["symbol", "timestamp"], inplace=True)

    async def get_positions_snapshot(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable snapshot of open positions."""

        async with self.position_lock:
            df = self.positions.copy()

        if df.empty:
            return []

        if "timestamp" in df.index.names:
            df = df.reset_index()

        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient="records")

        for record in records:
            for key, value in list(record.items()):
                if isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
        return records

    def _handle_corrupted_state_file(self, path: str, exc: Exception) -> str:
        sanitized_path = sanitize_log_value(path)
        logger.warning(
            "Поврежден файл состояния %s (%s). Состояние будет очищено.",
            sanitized_path,
            exc,
        )
        if os.path.exists(path):
            quarantine_name = f"{path}.corrupt.{int(time.time())}"
            try:
                shutil.move(path, quarantine_name)
            except OSError as move_err:
                logger.debug(
                    "Не удалось переместить поврежденный файл %s: %s",
                    sanitized_path,
                    move_err,
                )
                try:
                    os.remove(path)
                except OSError as remove_err:
                    logger.warning(
                        "Не удалось удалить поврежденный файл %s: %s",
                        sanitized_path,
                        remove_err,
                    )
                else:
                    logger.warning("Поврежденный файл %s удален", sanitized_path)
            else:
                logger.warning(
                    "Поврежденный файл %s перемещен в %s",
                    sanitized_path,
                    sanitize_log_value(quarantine_name),
                )
        return sanitized_path

    def _dispatch_state_reset_notice(self, files: list[str]) -> None:
        if not files:
            return
        unique_files = sorted(set(files))
        message = (
            "Очистка состояния TradeManager: поврежденные файлы "
            + ", ".join(unique_files)
        )
        sender = getattr(self.telegram_logger, "send_telegram_message", None)
        if not callable(sender):
            return
        try:
            result = sender(message, urgent=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Не удалось инициировать уведомление Telegram об очистке состояния: %s",
                exc,
            )
            return
        if inspect.isawaitable(result):
            awaitable = cast(Awaitable[Any], result)

            async def _consume(value: Awaitable[Any]) -> Any:
                return await value

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    asyncio.run(_consume(awaitable))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "Не удалось отправить уведомление Telegram об очистке состояния: %s",
                        exc,
                    )
            else:
                try:
                    coroutine = (
                        awaitable if inspect.iscoroutine(awaitable) else _consume(awaitable)
                    )
                    loop.create_task(coroutine)
                except RuntimeError as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "Не удалось запланировать уведомление Telegram об очистке состояния: %s",
                        exc,
                    )


    async def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        params: Dict | None = None,
        *,
        use_lock: bool = True,
    ) -> Optional[Dict]:
        params = params or {}

        async def _execute_once() -> Optional[Dict]:
            try:
                order_params = {"category": "linear", **params}
                order_type = order_params.get("type", "market")
                tp_price = order_params.get("takeProfitPrice")
                sl_price = order_params.get("stopLossPrice")
                if (tp_price is not None or sl_price is not None) and hasattr(
                    self.exchange, "create_order_with_take_profit_and_stop_loss"
                ):
                    order_params = {
                        k: v
                        for k, v in order_params.items()
                        if k not in {"takeProfitPrice", "stopLossPrice"}
                    }
                    order = await safe_api_call(
                        self.exchange,
                        "create_order_with_take_profit_and_stop_loss",
                        symbol,
                        order_type,
                        side,
                        size,
                        price if order_type != "market" else None,
                        tp_price,
                        sl_price,
                        order_params,
                    )
                else:
                    if tp_price is not None:
                        order_params["takeProfitPrice"] = tp_price
                    if sl_price is not None:
                        order_params["stopLossPrice"] = sl_price
                    order = await safe_api_call(
                        self.exchange,
                        "create_order",
                        symbol,
                        order_type,
                        side,
                        size,
                        price,
                        order_params,
                    )
                logger.info(
                    "Order placed: %s, %s, size=%s, price=%s, type=%s",
                    symbol,
                    side,
                    size,
                    price,
                    order_type,
                )
                await self.telegram_logger.send_telegram_message(
                    f"✅ Order: {symbol} {side.upper()} size={size:.4f} @ {price:.2f} ({order_type})"
                )

                if isinstance(order, dict):
                    ret_code = order.get("retCode") or order.get("ret_code")
                    if ret_code is not None and ret_code != 0:
                        logger.error("Ордер не подтверждён: %s", order)
                        await self.telegram_logger.send_telegram_message(
                            f"❌ Order not confirmed {symbol}: retCode {ret_code}"
                        )
                        return None

                return order
            except (httpx.HTTPError, RuntimeError) as e:
                logger.exception(
                    "Не удалось разместить ордер для %s (%s): %s",
                    symbol,
                    type(e).__name__,
                    e,
                )
                await self.telegram_logger.send_telegram_message(
                    f"❌ Order error {symbol}: {e}"
                )
                raise

        async def _submit() -> Optional[Dict]:
            attempts = max(1, int(self.config.get("order_retry_attempts", 3)))
            delay = float(self.config.get("order_retry_delay", 1))
            description = f"order {symbol}"

            def _on_exception(attempt: int, exc: BaseException) -> None:
                logger.error(
                    "Order attempt %s for %s failed (%s): %s",
                    attempt + 1,
                    symbol,
                    type(exc).__name__,
                    exc,
                )

            def _on_failed(attempt: int, result: Any) -> None:
                logger.warning(
                    "Order attempt %s for %s returned error: %s",
                    attempt + 1,
                    symbol,
                    result,
                )

            return await order_utils.execute_with_retries(
                _execute_once,
                attempts=attempts,
                delay=delay,
                sleep=asyncio.sleep,
                logger=logger,
                description=description,
                exceptions=(httpx.HTTPError, RuntimeError),
                should_retry=order_utils.order_needs_retry,
                on_exception=_on_exception,
                on_failed_result=_on_failed,
            )

        if use_lock:
            async with self.position_lock:
                return await _submit()
        else:
            return await _submit()

    async def calculate_position_size(
        self, symbol: str, price: float, atr: float, sl_multiplier: float
    ) -> float:
        try:
            if price <= 0 or atr <= 0:
                logger.warning(
                    "Invalid inputs for %s: price=%s, atr=%s",
                    symbol,
                    price,
                    atr,
                )
                return 0.0
            account = await safe_api_call(self.exchange, "fetch_balance")
            balance_key = self.config.get("balance_key")
            if not balance_key:
                sym = symbol.split(":", 1)[0]
                balance_key = sym.split("/")[1] if "/" in sym else "USDT"
            equity = float(account.get("total", {}).get(balance_key, 0))
            if equity <= 0:
                logger.warning("Недостаточный баланс для %s", symbol)
                await self.telegram_logger.send_telegram_message(
                    f"⚠️ Insufficient balance for {symbol}: equity={equity}"
                )
                return 0.0
            ohlcv = self.data_handler.ohlcv
            if (
                "symbol" in ohlcv.index.names
                and symbol in ohlcv.index.get_level_values("symbol")
            ):
                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
            else:
                df = None
            volatility = (
                df["close"].pct_change().std()
                if df is not None and not df.empty
                else self.config.get("volatility_threshold", 0.02)
            )
            risk_per_trade = await self.compute_risk_per_trade(symbol, volatility)
            position_size = order_utils.calculate_position_size(
                equity=equity,
                risk_per_trade=risk_per_trade,
                atr=atr,
                sl_multiplier=sl_multiplier,
                leverage=self.leverage,
                price=price,
                max_position_pct=self.max_position_pct,
            )
            if position_size <= 0:
                logger.warning("Некорректный размер позиции для %s", symbol)
                return 0.0
            logger.info(
                "Position size for %s: %.4f (risk %.2f USDT, ATR %.2f)",
                symbol,
                position_size,
                equity * risk_per_trade,
                atr,
            )
            return position_size
        except (httpx.HTTPError, KeyError, ValueError, RuntimeError) as e:
            logger.exception(
                "Не удалось вычислить размер позиции для %s (%s): %s",
                symbol,
                type(e).__name__,
                e,
            )
            raise

    def calculate_stop_loss_take_profit(
        self,
        side: str,
        price: float,
        atr: float,
        sl_multiplier: float,
        tp_multiplier: float,
    ) -> Tuple[float, float]:
        """Return stop-loss and take-profit prices."""
        return order_utils.calculate_stop_loss_take_profit(
            side, price, atr, sl_multiplier, tp_multiplier
        )

    async def open_position(self, symbol: str, side: str, price: float, params: Dict):
        try:
            async with self.position_lock:
                self._sort_positions()
                if len(self.positions) >= self.max_positions:
                    logger.warning(
                        "Достигнуто максимальное число позиций: %s",
                        self.max_positions,
                    )
                    return
                if side not in {"buy", "sell"}:
                    logger.warning("Некорректная сторона %s для %s", side, symbol)
                    return
                if self._has_position(symbol):
                    logger.warning("Позиция по %s уже открыта", symbol)
                    return

            if not await self.data_handler.is_data_fresh(symbol):
                logger.warning("Устаревшие данные для %s, пропуск сделки", symbol)
                return
            atr = await self.data_handler.get_atr(symbol)
            if atr <= 0:
                logger.warning(
                    "Данные ATR отсутствуют для %s, повтор позже",
                    symbol,
                )
                return
            sl_mult = params.get("sl_multiplier", self.config["sl_multiplier"])
            tp_mult = params.get("tp_multiplier", self.config["tp_multiplier"])
            size = await self.calculate_position_size(symbol, price, atr, sl_mult)
            if size <= 0:
                logger.warning("Размер позиции слишком мал для %s", symbol)
                return
            stop_loss_price, take_profit_price = self.calculate_stop_loss_take_profit(
                side, price, atr, sl_mult, tp_mult
            )
            protective_plan = order_utils.build_protective_order_plan(
                side,
                entry_price=price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )
            order_params = protective_plan.as_order_params(self.leverage)
            order = await self.place_order(
                symbol, side, size, price, order_params, use_lock=False
            )
            if order_utils.order_needs_retry(order):
                logger.error(
                    "Order failed for %s after %s attempts",
                    symbol,
                    self.config.get("order_retry_attempts", 3),
                )
                await self.telegram_logger.send_telegram_message(
                    f"❌ Order failed {symbol}: retries exhausted"
                )
                return
            if isinstance(order, dict) and not (
                order.get("id") or order.get("orderId") or order.get("result")
            ):
                logger.error(
                    "Order confirmation missing id for %s: %s",
                    symbol,
                    order,
                )
                await self.telegram_logger.send_telegram_message(
                    f"❌ Order confirmation missing id {symbol}"
                )
                return
            # Use an explicit timezone-aware timestamp for the position index
            timestamp = pd.Timestamp.now(tz="UTC")
            pos_sign = 1 if side == "buy" else -1
            new_position = {
                "symbol": symbol,
                "side": side,
                "position": pos_sign,
                "size": size,
                "entry_price": price,
                "tp_multiplier": tp_mult,
                "sl_multiplier": sl_mult,
                "stop_loss_price": stop_loss_price,
                "highest_price": price if pos_sign == 1 else float("inf"),
                "lowest_price": price if pos_sign == -1 else 0.0,
                "breakeven_triggered": False,
            }
            idx = (symbol, timestamp)
            async with self.position_lock:
                if self._has_position(symbol):
                    logger.warning(
                        "Позиция по %s уже открыта после размещения ордера",
                        symbol,
                    )
                    return
                if len(self.positions) >= self.max_positions:
                    logger.warning(
                        "Достигнуто максимальное число позиций после размещения ордера: %s",
                        self.max_positions,
                    )
                    return
                self.positions.loc[idx, :] = new_position
                self._sort_positions()
                self.positions_changed = True
            self.save_state()
            logger.info(
                "Position opened: %s, %s, size=%s, entry=%s",
                symbol,
                side,
                size,
                price,
            )
            await self.telegram_logger.send_telegram_message(
                (
                    f"📈 {symbol} {side.upper()} size={size:.4f} @ {price:.2f} "
                    f"SL={stop_loss_price:.2f} TP={take_profit_price:.2f}"
                ),
                urgent=True,
            )
        except (httpx.HTTPError, RuntimeError, ValueError, OSError) as e:
            logger.exception(
                "Не удалось открыть позицию для %s (%s): %s",
                symbol,
                type(e).__name__,
                e,
            )
            await self.telegram_logger.send_telegram_message(
                f"❌ Failed to open position {symbol}: {e}"
            )
            raise

    async def close_position(
        self, symbol: str, exit_price: float, reason: str = "Manual"
    ):
        # Fetch current position details under locks
        async with self.position_lock:
            async with self.returns_lock:
                self._sort_positions()
                if "symbol" in self.positions.index.names:
                    try:
                        position_df = self.positions.xs(
                            symbol, level="symbol", drop_level=False
                        )
                    except KeyError:
                        position_df = pd.DataFrame()
                else:
                    position_df = pd.DataFrame()
                if position_df.empty:
                    logger.warning("Позиция по %s не найдена", symbol)
                    return
                position = position_df.iloc[0]
                pos_idx = position_df.index[0]
                pos_sign = position["position"]
                side = "sell" if pos_sign == 1 else "buy"
                size = position["size"]
                entry_price = position["entry_price"]

        # Submit the order outside the locks
        try:
            order = await self.place_order(
                symbol,
                side,
                size,
                exit_price,
                use_lock=False,
            )
        except (httpx.HTTPError, RuntimeError) as e:  # pragma: no cover - network issues
            logger.exception(
                "Не удалось закрыть позицию для %s (%s): %s",
                symbol,
                type(e).__name__,
                e,
            )
            await self.telegram_logger.send_telegram_message(
                f"❌ Failed to close position {symbol}: {e}"
            )
            raise

        if not order:
            return

        profit = (exit_price - entry_price) * size * pos_sign
        profit *= self.leverage

        # Re-acquire locks to update state, verifying position still exists
        async with self.position_lock:
            async with self.returns_lock:
                if (
                    "symbol" in self.positions.index.names
                    and pos_idx in self.positions.index
                ):
                    self.positions = self.positions.drop(pos_idx)
                    self._sort_positions()
                    self.positions_changed = True
                    self.returns_by_symbol[symbol].append(
                        (pd.Timestamp.now(tz="UTC").timestamp(), profit)
                    )
                    self.save_state()
                else:
                    logger.warning(
                        "Position for %s modified before close confirmation", symbol
                    )

        logger.info(
            "Position closed: %s, profit=%.2f, reason=%s",
            symbol,
            profit,
            reason,
        )
        await self.telegram_logger.send_telegram_message(
            f"📉 {symbol} {position['side'].upper()} exit={exit_price:.2f} PnL={profit:.2f} USDT ({reason})",
            urgent=True,
        )

    async def check_trailing_stop(self, symbol: str, current_price: float):
        should_close = False
        exit_price = current_price
        async with self.position_lock:
            try:
                self._sort_positions()
                if "symbol" in self.positions.index.names:
                    try:
                        position_df = self.positions.xs(
                            symbol, level="symbol", drop_level=False
                        )
                    except KeyError:
                        position_df = pd.DataFrame()
                else:
                    position_df = pd.DataFrame()
                if position_df.empty:
                    logger.debug("Позиция по %s не найдена", symbol)
                    return
                position = position_df.iloc[0]
                atr = await self.data_handler.get_atr(symbol)
                if atr <= 0:
                    logger.debug("Данные ATR отсутствуют для %s, повтор позже", symbol)
                    return
                trailing_stop_distance = atr * self.config.get(
                    "trailing_stop_multiplier", 1.0
                )
                tick_size = 0.0
                if hasattr(self.data_handler, "get_tick_size"):
                    ts = self.data_handler.get_tick_size(symbol)
                    tick_size = await ts if inspect.isawaitable(ts) else ts

                profit_pct = (
                    (current_price - position["entry_price"])
                    / position["entry_price"]
                    * 100
                    if position["side"] == "buy"
                    else (position["entry_price"] - current_price)
                    / position["entry_price"]
                    * 100
                )
                profit_atr = (
                    current_price - position["entry_price"]
                    if position["side"] == "buy"
                    else position["entry_price"] - current_price
                )

                trigger_pct = self.config.get("trailing_stop_percentage", 1.0)
                trigger_atr = self.config.get("trailing_stop_coeff", 1.0) * atr

                if not position["breakeven_triggered"] and (
                    profit_pct >= trigger_pct or profit_atr >= trigger_atr
                ):
                    close_size = position["size"] * 0.5
                    side = "sell" if position["side"] == "buy" else "buy"
                    await self.place_order(
                        symbol,
                        side,
                        close_size,
                        current_price,
                        use_lock=False,
                    )
                    remaining_size = position["size"] - close_size
                    self.positions.loc[
                        pd.IndexSlice[symbol, :], "size"
                    ] = remaining_size
                    breakeven_sl = position["entry_price"] + (
                        tick_size if position["side"] == "buy" else -tick_size
                    )
                    self.positions.loc[
                        pd.IndexSlice[symbol, :], "stop_loss_price"
                    ] = breakeven_sl
                    self.positions.loc[
                        pd.IndexSlice[symbol, :], "breakeven_triggered"
                    ] = True
                    self.positions_changed = True
                    self.save_state()
                    await self.telegram_logger.send_telegram_message(
                        f"🏁 {symbol} moved to breakeven, partial profits taken"
                    )

                idx = pd.IndexSlice[symbol, :]
                ohlcv = self.data_handler.ohlcv
                if (
                    "symbol" in ohlcv.index.names
                    and symbol in ohlcv.index.get_level_values("symbol")
                ):
                    df = ohlcv.xs(symbol, level="symbol", drop_level=False)
                    current_ts = df.index.get_level_values("timestamp")[-1]
                    last_trailing = position.get("last_trailing_ts")
                    if pd.isna(last_trailing) or current_ts > last_trailing:
                        close_price = df["close"].iloc[-1]
                        if position["side"] == "buy":
                            new_highest = max(position["highest_price"], close_price)
                            self.positions.loc[idx, "highest_price"] = new_highest
                        else:
                            new_lowest = min(position["lowest_price"], close_price)
                            self.positions.loc[idx, "lowest_price"] = new_lowest
                        self.positions.loc[idx, "last_trailing_ts"] = current_ts
                        self.positions_changed = True
                        self.save_state()
                        position = self.positions.xs(symbol, level="symbol").iloc[0]

                if position["side"] == "buy":
                    trailing_stop_price = (
                        position["highest_price"] - trailing_stop_distance
                    )
                    if current_price <= trailing_stop_price:
                        should_close = True
                        if trailing_stop_price - current_price > tick_size:
                            logger.debug(
                                "Price skipped trailing stop for %s: stop=%.2f, price=%.2f",
                                symbol,
                                trailing_stop_price,
                                current_price,
                            )
                        exit_price = current_price
                else:
                    trailing_stop_price = (
                        position["lowest_price"] + trailing_stop_distance
                    )
                    if current_price >= trailing_stop_price:
                        should_close = True
                        if current_price - trailing_stop_price > tick_size:
                            logger.debug(
                                "Price skipped trailing stop for %s: stop=%.2f, price=%.2f",
                                symbol,
                                trailing_stop_price,
                                current_price,
                            )
                        exit_price = current_price

                self.positions.loc[idx, "last_checked_ts"] = pd.Timestamp.now(tz="UTC")
            except (KeyError, ValueError) as e:
                logger.exception(
                    "Failed trailing stop check for %s (%s): %s",
                    symbol,
                    type(e).__name__,
                    e,
                )
                raise
        if should_close:
            await self.close_position(symbol, exit_price, "Trailing Stop")

    async def check_stop_loss_take_profit(self, symbol: str, current_price: float):
        close_reason = None
        async with self.position_lock:
            try:
                self._sort_positions()
                if "symbol" in self.positions.index.names:
                    try:
                        position = self.positions.xs(symbol, level="symbol")
                    except KeyError:
                        position = pd.DataFrame()
                else:
                    position = pd.DataFrame()
                if position.empty:
                    return
                position = position.iloc[0]
                ohlcv = self.data_handler.ohlcv
                if (
                    "symbol" not in ohlcv.index.names
                    or symbol not in ohlcv.index.get_level_values("symbol")
                ):
                    return
                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
                current_ts = df.index.get_level_values("timestamp")[-1]
                last_checked = position.get("last_checked_ts")
                if pd.notna(last_checked) and last_checked >= current_ts:
                    return
                self.positions.loc[pd.IndexSlice[symbol, :], "last_checked_ts"] = current_ts
                self.positions_changed = True
                self.save_state()
                indicators = self.data_handler.indicators.get(symbol)
                if not indicators or not indicators.atr.iloc[-1]:
                    return
                atr = indicators.atr.iloc[-1]
                if position["breakeven_triggered"]:
                    stop_loss = position["stop_loss_price"]
                    take_profit = (
                        position["entry_price"] + position["tp_multiplier"] * atr
                        if position["side"] == "buy"
                        else position["entry_price"] - position["tp_multiplier"] * atr
                    )
                else:
                    if position["side"] == "buy":
                        stop_loss = (
                            position["entry_price"] - position["sl_multiplier"] * atr
                        )
                        take_profit = (
                            position["entry_price"] + position["tp_multiplier"] * atr
                        )
                    else:
                        stop_loss = (
                            position["entry_price"] + position["sl_multiplier"] * atr
                        )
                        take_profit = (
                            position["entry_price"] - position["tp_multiplier"] * atr
                        )
                    self.positions.loc[
                        pd.IndexSlice[symbol, :], "stop_loss_price"
                    ] = stop_loss
                if position["side"] == "buy" and current_price <= stop_loss:
                    close_reason = "Stop Loss"
                elif position["side"] == "sell" and current_price >= stop_loss:
                    close_reason = "Stop Loss"
                elif position["side"] == "buy" and current_price >= take_profit:
                    close_reason = "Take Profit"
                elif position["side"] == "sell" and current_price <= take_profit:
                    close_reason = "Take Profit"
            except (KeyError, ValueError) as e:
                logger.exception(
                    "Failed SL/TP check for %s (%s): %s",
                    symbol,
                    type(e).__name__,
                    e,
                )
                raise
        if close_reason:
            await self.close_position(symbol, current_price, close_reason)

    async def check_exit_signal(self, symbol: str, current_price: float):
        try:
            if self.model_builder is None:
                return
            model = self.model_builder.predictive_models.get(symbol)
            if not model:
                logger.debug("Модель для %s не найдена", symbol)
                return
            async with self.position_lock:
                self._sort_positions()
                if "symbol" in self.positions.index.names:
                    try:
                        position = self.positions.xs(symbol, level="symbol")
                    except KeyError:
                        position = pd.DataFrame()
                else:
                    position = pd.DataFrame()
                num_positions = len(self.positions)
            if position.empty:
                return
            position = position.iloc[0]
            indicators = self.data_handler.indicators.get(symbol)
            empty = await _check_df_async(
                indicators.df, f"check_exit_signal {symbol}"
            )
            if not indicators or empty:
                return
            features = self.model_builder.get_cached_features(symbol)
            if features is None or len(features) < self.config["lstm_timesteps"]:
                try:
                    features = await self.model_builder.prepare_lstm_features(
                        symbol, indicators
                    )
                except (RuntimeError, ValueError) as exc:
                    logger.debug(
                        "Не удалось подготовить признаки для %s (%s): %s",
                        symbol,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                    return
                self.model_builder.feature_cache[symbol] = features
            if len(features) < self.config["lstm_timesteps"]:
                logger.debug(
                    "Not enough features for %s: %s", symbol, len(features)
                )
                return
            X = np.array([features[-self.config["lstm_timesteps"] :]])
            X_tensor = torch.tensor(
                X, dtype=torch.float32, device=self.model_builder.device
            )
            prediction = float(await _predict_async(model, X_tensor))
            calibrator = self.model_builder.calibrators.get(symbol)
            if calibrator is not None:
                prediction = await asyncio.to_thread(
                    _calibrate_output,
                    calibrator,
                    float(prediction),
                )
            rl_signal = None
            if self.rl_agent and symbol in self.rl_agent.models:
                rl_feat = np.append(
                    features[-1],
                    [float(prediction), num_positions / max(1, self.max_positions)],
                ).astype(np.float32)
                rl_signal = self.rl_agent.predict(symbol, rl_feat)
                if rl_signal == "open_long":
                    return "buy"
                if rl_signal == "open_short":
                    return "sell"
                if rl_signal == "close":
                    await self.close_position(symbol, current_price, "RL Signal")
                    return
                if rl_signal == "open_long" and position["side"] == "sell":
                    await self.close_position(symbol, current_price, "RL Reverse")
                    params = await self.data_handler.parameter_optimizer.optimize(symbol)
                    await self.open_position(symbol, "buy", current_price, params)
                    return
                if rl_signal == "open_short" and position["side"] == "buy":
                    await self.close_position(symbol, current_price, "RL Reverse")
                    params = await self.data_handler.parameter_optimizer.optimize(symbol)
                    await self.open_position(symbol, "sell", current_price, params)
                    return
            long_threshold, short_threshold = (
                await self.model_builder.adjust_thresholds(symbol, prediction)
            )
            logger.info(
                "Пороги модели для %s: long=%.2f, short=%.2f",
                symbol,
                long_threshold,
                short_threshold,
            )
            if position["side"] == "buy" and prediction < short_threshold:
                logger.info(
                    "Сигнал выхода из лонга по модели для %s: пред=%.4f, порог=%.2f",
                    symbol,
                    prediction,
                    short_threshold,
                )
                await self.close_position(symbol, current_price, "Model Exit Signal")
                if prediction <= short_threshold - self.config.get("reversal_margin", 0.05):
                    opposite = "sell"
                    ema_ok = await self.evaluate_ema_condition(symbol, opposite)
                    if ema_ok:
                        async with self.position_lock:
                            already_open = self._has_position(symbol)
                        if not already_open:
                            params = await self.data_handler.parameter_optimizer.optimize(symbol)
                            await self.open_position(symbol, opposite, current_price, params)
            elif position["side"] == "sell" and prediction > long_threshold:
                logger.info(
                    "Сигнал выхода из шорта по модели для %s: пред=%.4f, порог=%.2f",
                    symbol,
                    prediction,
                    long_threshold,
                )
                await self.close_position(symbol, current_price, "Model Exit Signal")
                if prediction >= long_threshold + self.config.get("reversal_margin", 0.05):
                    opposite = "buy"
                    ema_ok = await self.evaluate_ema_condition(symbol, opposite)
                    if ema_ok:
                        async with self.position_lock:
                            already_open = self._has_position(symbol)
                        if not already_open:
                            params = await self.data_handler.parameter_optimizer.optimize(symbol)
                            await self.open_position(symbol, opposite, current_price, params)
        except (httpx.HTTPError, RuntimeError, ValueError) as e:
            logger.exception(
                "Не удалось проверить сигнал модели для %s (%s): %s",
                symbol,
                type(e).__name__,
                e,
            )
            raise

    async def monitor_performance(self):
        while True:
            try:
                async with self.returns_lock:
                    current_time = pd.Timestamp.now(tz="UTC").timestamp()
                    for symbol in self.returns_by_symbol:
                        returns = [
                            r
                            for t, r in self.returns_by_symbol[symbol]
                            if current_time - t <= self.performance_window
                        ]
                        self.returns_by_symbol[symbol] = [
                            (t, r)
                            for t, r in self.returns_by_symbol[symbol]
                            if current_time - t <= self.performance_window
                        ]
                        if returns:
                            sharpe_ratio = (
                                np.mean(returns)
                                / (np.std(returns) + 1e-6)
                                * np.sqrt(365 * 24 * 60 * 60 / self.performance_window)
                            )
                            logger.info(
                                "Sharpe Ratio for %s: %.2f",
                                symbol,
                                sharpe_ratio,
                            )
                            ohlcv = self.data_handler.ohlcv
                            if (
                                "symbol" in ohlcv.index.names
                                and symbol in ohlcv.index.get_level_values("symbol")
                            ):
                                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
                            else:
                                df = None
                            if df is not None and not df.empty:
                                volatility = df["close"].pct_change().std()
                                volatility_change = abs(
                                    volatility - self.last_volatility.get(symbol, 0.0)
                                ) / max(self.last_volatility.get(symbol, 0.01), 0.01)
                                self.last_volatility[symbol] = volatility
                                if (
                                    sharpe_ratio
                                    < self.config.get("min_sharpe_ratio", 0.5)
                                    or volatility_change > 0.5
                                ):
                                    logger.info(
                                        "Retraining triggered for %s: Sharpe=%.2f, Volatility change=%.2f",
                                        symbol,
                                        sharpe_ratio,
                                        volatility_change,
                                    )
                                    retrained = await self._maybe_retrain_symbol(symbol)
                                    if retrained:
                                        await self.telegram_logger.send_telegram_message(
                                            (
                                                f"🔄 Retraining {symbol}: Sharpe={sharpe_ratio:.2f}, "
                                                f"Volatility={volatility_change:.2f}"
                                            )
                                        )
                            if sharpe_ratio < self.config.get("min_sharpe_ratio", 0.5):
                                logger.warning(
                                    "Low Sharpe Ratio for %s: %.2f",
                                    symbol,
                                    sharpe_ratio,
                                )
                                await self.telegram_logger.send_telegram_message(
                                    f"⚠️ Low Sharpe Ratio for {symbol}: {sharpe_ratio:.2f}"
                                )
                current_day = int(current_time // 86400)
                if current_day != self.last_stats_day:
                    stats = await self.compute_stats()
                    logger.info(
                        "Daily stats: win_rate=%.2f%% avg_pnl=%.2f max_drawdown=%.2f",
                        stats["win_rate"] * 100,
                        stats["avg_pnl"],
                        stats["max_drawdown"],
                    )
                    self.last_stats_day = current_day
                await asyncio.sleep(self.performance_window / 10)
            except asyncio.CancelledError:
                raise
            except (httpx.HTTPError, ValueError, RuntimeError) as e:
                logger.exception(
                    "Performance monitoring error (%s): %s",
                    type(e).__name__,
                    e,
                )
                await asyncio.sleep(1)
                continue

    async def manage_positions(self):
        while True:
            try:
                async with self.position_lock:
                    symbols = []
                    if "symbol" in self.positions.index.names:
                        symbols = self.positions.index.get_level_values("symbol").unique()
                for symbol in symbols:
                    ohlcv = self.data_handler.ohlcv
                    if (
                        "symbol" in ohlcv.index.names
                        and symbol in ohlcv.index.get_level_values("symbol")
                    ):
                        df = ohlcv.xs(symbol, level="symbol", drop_level=False)
                    else:
                        df = None
                    empty = await _check_df_async(df, f"manage_positions {symbol}")
                    if empty:
                        continue
                    current_price = df["close"].iloc[-1]
                    if self._has_position(symbol):
                        res = self.check_trailing_stop(symbol, current_price)
                        if inspect.isawaitable(res):
                            await res
                    if self._has_position(symbol):
                        res = self.check_stop_loss_take_profit(symbol, current_price)
                        if inspect.isawaitable(res):
                            await res
                    if self._has_position(symbol):
                        res = self.check_exit_signal(symbol, current_price)
                        if inspect.isawaitable(res):
                            await res
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                raise
            except (
                ValueError,
                RuntimeError,
                KeyError,
                httpx.HTTPError,
                aiohttp.ClientError,
            ) as e:
                logger.exception(
                    "Error managing positions (%s): %s",
                    type(e).__name__,
                    e,
                )
                await asyncio.sleep(1)
                continue

    async def evaluate_ema_condition(self, symbol: str, signal: str) -> bool:
        try:
            ohlcv_2h = self.data_handler.ohlcv_2h
            if (
                "symbol" in ohlcv_2h.index.names
                and symbol in ohlcv_2h.index.get_level_values("symbol")
            ):
                df_2h = ohlcv_2h.xs(symbol, level="symbol", drop_level=False)
            else:
                df_2h = None
            indicators_2h = self.data_handler.indicators_2h.get(symbol)
            empty = await _check_df_async(df_2h, f"evaluate_ema_condition {symbol}")
            if empty or not indicators_2h:
                logger.warning(
                    "No data or indicators for %s on 2h timeframe",
                    symbol,
                )
                return False
            ema30 = indicators_2h.ema30
            ema100 = indicators_2h.ema100
            close = df_2h["close"]
            timestamps = df_2h.index.get_level_values("timestamp")
            lookback_period = pd.Timedelta(
                seconds=self.config["ema_crossover_lookback"]
            )
            recent_data = df_2h[timestamps >= timestamps[-1] - lookback_period]
            if len(recent_data) < 2:
                logger.debug(
                    "Not enough data to check EMA crossover for %s",
                    symbol,
                )
                return False
            ema30_recent = ema30[-len(recent_data) :]
            ema100_recent = ema100[-len(recent_data) :]
            crossover_long = (ema30_recent.iloc[-2] <= ema100_recent.iloc[-2]) and (
                ema30_recent.iloc[-1] > ema100_recent.iloc[-1]
            )
            crossover_short = (ema30_recent.iloc[-2] >= ema100_recent.iloc[-2]) and (
                ema30_recent.iloc[-1] < ema100_recent.iloc[-1]
            )
            if (signal == "buy" and not crossover_long) or (
                signal == "sell" and not crossover_short
            ):
                logger.debug(
                    "EMA crossover not confirmed for %s, signal=%s",
                    symbol,
                    signal,
                )
                return False
            pullback_period = pd.Timedelta(seconds=self.config["pullback_period"])
            pullback_data = df_2h[timestamps >= timestamps[-1] - pullback_period]
            volatility = close.pct_change().std() if not close.empty else 0.02
            pullback_threshold = (
                ema30.iloc[-1] * self.config["pullback_volatility_coeff"] * volatility
            )
            pullback_zone_high = ema30.iloc[-1] + pullback_threshold
            pullback_zone_low = ema30.iloc[-1] - pullback_threshold
            pullback_occurred = False
            for i in range(len(pullback_data)):
                price = pullback_data["close"].iloc[i]
                if pullback_zone_low <= price <= pullback_zone_high:
                    pullback_occurred = True
                    break
            if not pullback_occurred:
                logger.debug(
                    "No pullback to EMA30 for %s, signal=%s",
                    symbol,
                    signal,
                )
                return False
            current_price = close.iloc[-1]
            if (signal == "buy" and current_price <= ema30.iloc[-1]) or (
                signal == "sell" and current_price >= ema30.iloc[-1]
            ):
                logger.debug("Цена не консолидирована для %s, сигнал=%s", symbol, signal)
                return False
            logger.info("Условия EMA выполнены для %s, сигнал=%s", symbol, signal)
            return True
        except (KeyError, ValueError) as e:
            logger.exception(
                "Не удалось проверить условия EMA для %s (%s): %s",
                symbol,
                type(e).__name__,
                e,
            )
            raise

    async def _dataset_has_multiple_classes(
        self, symbol: str, features: np.ndarray | None = None
    ) -> tuple[bool, int]:
        if features is None:
            indicators = self.data_handler.indicators.get(symbol)
            empty = (
                await _check_df_async(indicators.df, f"dataset check {symbol}")
                if indicators
                else True
            )
            if not indicators or empty:
                return False, 0
            try:
                features = await self.model_builder.prepare_lstm_features(
                    symbol, indicators
                )
            except (RuntimeError, ValueError) as exc:
                logger.debug(
                    "Не удалось подготовить признаки для %s (%s): %s",
                    symbol,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                return False, 0
        required_len = self.config["lstm_timesteps"] * 2
        if len(features) < required_len:
            return False, len(features)
        _, y = self.model_builder.prepare_dataset(features)
        return len(np.unique(y)) >= 2, len(y)

    async def _maybe_retrain_symbol(
        self, symbol: str, features: np.ndarray | None = None
    ) -> bool:
        has_classes, size = await self._dataset_has_multiple_classes(symbol, features)
        if not has_classes or size <= self._min_retrain_size.get(symbol, 0):
            self._min_retrain_size[symbol] = max(
                size, self._min_retrain_size.get(symbol, 0)
            )
            logger.debug(
                "Insufficient class labels for %s; postponing retrain", symbol
            )
            return False
        try:
            await self.model_builder.retrain_symbol(symbol)
            self._min_retrain_size.pop(symbol, None)
            return True
        except (httpx.HTTPError, aiohttp.ClientError, ConnectionError, RuntimeError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 400:
                self._min_retrain_size[symbol] = size
                logger.info(
                    "Retraining deferred for %s until more data accumulates", symbol
                )
                return False
            logger.error(
                "Retraining failed for %s (%s): %s",
                symbol,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            raise

    async def evaluate_signal(self, symbol: str, return_prob: bool = False):
        try:
            model = self.model_builder.predictive_models.get(symbol)
            indicators = self.data_handler.indicators.get(symbol)
            empty = (
                await _check_df_async(indicators.df, f"evaluate_signal {symbol}")
                if indicators
                else True
            )
            if not indicators or empty:
                return None
            ohlcv = self.data_handler.ohlcv
            if (
                "symbol" in ohlcv.index.names
                and symbol in ohlcv.index.get_level_values("symbol")
            ):
                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
            else:
                df = None
            if not await self.data_handler.is_data_fresh(symbol):
                logger.debug("Устаревшие данные для %s, пропуск сигнала", symbol)
                return None
            if df is not None and not df.empty:
                # Расчёт волатильности ранее сохранялся в переменную, которая не
                # использовалась.  Убираем присваивание, чтобы избежать F841 и
                # лишних предупреждений от линтера.
                df["close"].pct_change().std()
            else:
                self.config.get("volatility_threshold", 0.02)
            features = self.model_builder.get_cached_features(symbol)
            if features is None or len(features) < self.config["lstm_timesteps"]:
                try:
                    features = await self.model_builder.prepare_lstm_features(
                        symbol, indicators
                    )
                except (RuntimeError, ValueError) as exc:
                    logger.debug(
                        "Не удалось подготовить признаки для %s (%s): %s",
                        symbol,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                    return None
                self.model_builder.feature_cache[symbol] = features
            if len(features) < self.config["lstm_timesteps"]:
                logger.debug(
                    "Not enough features for %s: %s", symbol, len(features)
                )
                return None
            if not model:
                logger.debug("Модель для %s ещё не обучена", symbol)
                if not await self._maybe_retrain_symbol(symbol, features):
                    return None
                model = self.model_builder.predictive_models.get(symbol)
                if not model:
                    return None
            X = np.array([features[-self.config["lstm_timesteps"] :]])
            X_tensor = torch.tensor(
                X, dtype=torch.float32, device=self.model_builder.device
            )
            prediction = float(await _predict_async(model, X_tensor))
            calibrator = self.model_builder.calibrators.get(symbol)
            if calibrator is not None:
                prediction = await asyncio.to_thread(
                    _calibrate_output,
                    calibrator,
                    float(prediction),
                )

            if self.config.get("prediction_target", "direction") == "pnl":
                cost = 2 * self.config.get("trading_fee", 0.0)
                if prediction > cost:
                    signal = "buy"
                elif prediction < -cost:
                    signal = "sell"
                else:
                    signal = None
                return (signal, float(prediction)) if return_prob else signal

            long_threshold, short_threshold = (
                await self.model_builder.adjust_thresholds(symbol, prediction)
            )
            signal = None
            if prediction > long_threshold:
                signal = "buy"
            elif prediction < short_threshold:
                signal = "sell"

            rl_signal = None
            if self.rl_agent and symbol in self.rl_agent.models:
                async with self.position_lock:
                    num_positions = len(self.positions)
                rl_feat = np.append(
                    features[-1],
                    [float(prediction), num_positions / max(1, self.max_positions)],
                ).astype(np.float32)
                rl_signal = self.rl_agent.predict(symbol, rl_feat)
                if rl_signal == "open_long":
                    return "buy"
                if rl_signal == "open_short":
                    return "sell"

            ema_signal = None
            check = self.evaluate_ema_condition(symbol, "buy")
            if inspect.isawaitable(check):
                ema_buy = await check
            else:
                ema_buy = check
            if ema_buy:
                ema_signal = "buy"
            else:
                check = self.evaluate_ema_condition(symbol, "sell")
                if inspect.isawaitable(check):
                    ema_sell = await check
                else:
                    ema_sell = check
                if ema_sell:
                    ema_signal = "sell"

            final = None
            weights = {
                "transformer": self.config.get("transformer_weight", 0.5),
                "ema": self.config.get("ema_weight", 0.2),
            }
            scores = {"buy": 0.0, "sell": 0.0}
            scores["buy"] += weights["transformer"] * float(prediction)
            scores["sell"] += weights["transformer"] * (1.0 - float(prediction))
            if ema_signal == "buy":
                scores["buy"] += weights["ema"]
            elif ema_signal == "sell":
                scores["sell"] += weights["ema"]
            
            gpt_signal = None
            try:
                from bot import trading_bot as tb  # type: ignore[attr-defined]
                gpt_signal = tb.GPT_ADVICE.signal
            except Exception:
                gpt_signal = None
            if gpt_signal in ("buy", "sell"):
                weights["gpt"] = self.config.get("gpt_weight", 0.3)
                if gpt_signal == "buy":
                    scores["buy"] += weights["gpt"]
                else:
                    scores["sell"] += weights["gpt"]

            total_weight = sum(weights.values())
            if scores["buy"] > scores["sell"] and scores["buy"] >= total_weight / 2:
                final = "buy"
            elif scores["sell"] > scores["buy"] and scores["sell"] >= total_weight / 2:
                final = "sell"
            else:
                final = None
            if rl_signal == "open_long":
                final = "buy"
            elif rl_signal == "open_short":
                final = "sell"
            if final:
                logger.info(
                      "Voting result for %s -> %s (scores %.2f/%.2f)",
                      symbol,
                      final,
                    scores["buy"],
                    scores["sell"],
                )
            if return_prob:
                return final, float(prediction)
            return final
        except (httpx.HTTPError, RuntimeError, ValueError) as e:
            logger.exception(
                "Failed to evaluate signal for %s (%s): %s",
                symbol,
                type(e).__name__,
                e,
            )
            raise

    async def gather_pending_signals(self):
        """Collect and rank signals for all symbols."""
        signals = []
        async with self.position_lock:
            if "symbol" in self.positions.index.names:
                open_symbols = set(
                    self.positions.index.get_level_values("symbol").unique()
                )
            else:
                open_symbols = set()
        for symbol in self.data_handler.usdt_pairs:
            if symbol in open_symbols:
                continue
            result = await self.evaluate_signal(symbol, return_prob=True)
            if not result:
                continue
            signal, prob = result
            if not signal:
                continue
            ohlcv = self.data_handler.ohlcv
            if (
                "symbol" in ohlcv.index.names
                and symbol in ohlcv.index.get_level_values("symbol")
            ):
                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
            else:
                df = None
            empty = await _check_df_async(df, f"gather_pending_signals {symbol}")
            if empty:
                continue
            price = df["close"].iloc[-1]
            atr = await self.data_handler.get_atr(symbol)
            score = float(prob) * float(atr)
            signals.append({"symbol": symbol, "signal": signal, "score": score, "price": price})
        signals.sort(key=lambda s: s["score"], reverse=True)
        return signals

    async def execute_top_signals_once(self):
        """Open positions for the best-ranked signals."""
        signals = await self.gather_pending_signals()
        for info in signals[: self.top_signals]:
            params = await self.data_handler.parameter_optimizer.optimize(info["symbol"])
            await self.open_position(info["symbol"], info["signal"], info["price"], params)

    async def ranked_signal_loop(self):
        while True:
            try:
                await self.execute_top_signals_once()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                raise
            except (ValueError, RuntimeError, httpx.HTTPError) as e:
                logger.exception(
                    "Error processing ranked signals (%s): %s",
                    type(e).__name__,
                    e,
                )
                await asyncio.sleep(1)

    async def _notify_auxiliary_failure(self, spec: _TaskSpec, exc: Exception) -> None:
        message = f"⚠️ Task {spec.name} failed: {exc}"
        try:
            await self.telegram_logger.send_telegram_message(message)
        except Exception as send_exc:  # pragma: no cover - defensive logging
            logger.debug(
                "Не удалось отправить уведомление Telegram о сбое задачи %s: %s",
                spec.name,
                send_exc,
            )

    async def _run_background_task(self, spec: _TaskSpec) -> None:
        base_delay = 0.1
        max_delay = 30.0
        delay = base_delay
        while True:
            try:
                await spec.factory()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if spec.critical or not spec.restart:
                    raise
                logger.exception(
                    "Некритичная задача %s завершилась с ошибкой: %s",
                    spec.name,
                    exc,
                )
                await self._notify_auxiliary_failure(spec, exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            else:
                if not spec.restart:
                    break
                delay = base_delay
                await asyncio.sleep(delay)

    async def run(self):
        try:
            self.loop = asyncio.get_running_loop()
            self._failure_notified = False
            self._critical_error = False
            specs = [
                _TaskSpec(
                    name="monitor_performance",
                    factory=self.monitor_performance,
                    critical=False,
                    restart=True,
                ),
                _TaskSpec(
                    name="manage_positions",
                    factory=self.manage_positions,
                    critical=True,
                ),
                _TaskSpec(
                    name="ranked_signal_loop",
                    factory=self.ranked_signal_loop,
                    critical=True,
                ),
            ]
            task_map: dict[asyncio.Task[Any], _TaskSpec] = {}
            self.tasks = []
            for spec in specs:
                task = asyncio.create_task(
                    self._run_background_task(spec), name=spec.name
                )
                self.tasks.append(task)
                task_map[task] = spec

            pending: set[asyncio.Task[Any]] = set(self.tasks)
            try:
                while pending:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for finished in done:
                        spec = task_map.get(finished)
                        try:
                            finished.result()
                        except asyncio.CancelledError:
                            continue
                        except Exception as exc:
                            self._failure_notified = True
                            logger.error(
                                "Задача %s завершилась с ошибкой: %s",
                                finished.get_name(),
                                exc,
                            )
                            await self.telegram_logger.send_telegram_message(
                                f"❌ Task {finished.get_name()} failed: {exc}"
                            )
                            for task in pending:
                                task.cancel()
                            if pending:
                                await asyncio.gather(
                                    *pending, return_exceptions=True
                                )
                            raise TradeManagerTaskError(str(exc)) from exc
                        else:
                            if spec and spec.critical:
                                self._failure_notified = True
                                logger.error(
                                    "Критическая задача %s завершилась без ошибок, но преждевременно",
                                    spec.name,
                                )
                                for task in pending:
                                    task.cancel()
                                if pending:
                                    await asyncio.gather(
                                        *pending, return_exceptions=True
                                    )
                                raise TradeManagerTaskError(
                                    f"Task {spec.name} terminated unexpectedly"
                                )
            finally:
                await asyncio.gather(*self.tasks, return_exceptions=True)
        except (httpx.HTTPError, RuntimeError, ValueError) as e:
            logger.exception(
                "Critical error in TradeManager (%s): %s",
                type(e).__name__,
                e,
            )
            if not self._failure_notified:
                await self.telegram_logger.send_telegram_message(
                    f"❌ Critical TradeManager error: {e}"
                )
            self._critical_error = True
            if self.loop and self.loop.is_running() and not _is_test_mode_enabled():
                self.loop.stop()
            raise
        finally:
            self.tasks.clear()

    async def stop(self) -> None:
        """Cancel running tasks and shut down Telegram logging."""
        for task in list(self.tasks):
            task.cancel()
        for task in list(self.tasks):
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("Cancelled task %s during shutdown", getattr(task, "get_name", lambda: repr(task))())
        self.tasks.clear()

        await TelegramLogger.shutdown()
        await close_http_client()

    def shutdown(self) -> None:
        """Synchronous wrapper for graceful shutdown."""
        if self.loop and self.loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.stop(), self.loop)
            try:
                fut.result()
            except (RuntimeError, ValueError) as e:
                logger.exception("Ошибка при ожидании остановки (%s): %s", type(e).__name__, e)
        else:
            try:
                asyncio.run(self.stop())
            except RuntimeError:
                # event loop already closed
                pass
        try:
            should_shutdown = False
            if _is_test_mode_enabled():
                should_shutdown = True
            else:
                is_initialized = getattr(ray, "is_initialized", None)
                if callable(is_initialized):
                    should_shutdown = bool(is_initialized())

            if should_shutdown and callable(getattr(ray, "shutdown", None)):
                ray.shutdown()
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - cleanup errors
            logger.exception("Не удалось завершить Ray (%s): %s", type(exc).__name__, exc)


