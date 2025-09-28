#!/usr/bin/env python3
"""Command-line entry point for running the trading bot or a simulation."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Protocol

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from bot.config import BotConfig


logger = logging.getLogger("TradingBot")


class ExchangeFactory(Protocol):
    """Callable returning an exchange-like object."""

    def __call__(self) -> Any:
        ...


class TelegramLoggerFactory(Protocol):
    """Factory returning a Telegram logger handler."""

    def __call__(self, *args: Any, **kwargs: Any) -> logging.Handler:
        ...


class GPTClientProtocol(Protocol):
    """Subset of GPT client behaviour used by the bot."""

    def query(self, prompt: str) -> str:  # pragma: no cover - simple delegation
        ...

    async def query_async(self, prompt: str) -> str:
        ...

    async def query_json_async(self, prompt: str) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class ServiceBundle:
    """Resolved service implementations used across the application."""

    exchange: Any | None
    telegram_logger_factory: TelegramLoggerFactory | None
    gpt_client: GPTClientProtocol | None


def _import_object(spec: str) -> Any:
    """Return the object referenced by *spec* (``pkg.mod:attr`` syntax)."""

    module_name: str
    attr_name: str
    if ":" in spec:
        module_name, attr_name = spec.split(":", 1)
    else:
        module_name, attr_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:  # pragma: no cover - defensive branch
        raise ImportError(f"{spec!r} is not importable") from exc


def _resolve_service_object(raw: Any) -> Any:
    """Resolve dotted-path strings to Python objects."""

    if isinstance(raw, str):
        return _import_object(raw)
    return raw


def _instantiate_service(candidate: Any) -> Any:
    """Instantiate *candidate* when appropriate, preserving callables."""

    resolved = _resolve_service_object(candidate)
    if inspect.isclass(resolved):
        try:
            return resolved()
        except TypeError:
            return resolved
    if callable(resolved):
        try:
            return resolved()
        except TypeError:
            return resolved
    return resolved


class ModuleGPTClientAdapter:
    """Adapter exposing module-level GPT helpers as an object."""

    def query(self, prompt: str) -> str:
        from bot.gpt_client import query_gpt

        return query_gpt(prompt)

    async def query_async(self, prompt: str) -> str:
        from bot.gpt_client import query_gpt_async

        return await query_gpt_async(prompt)

    async def query_json_async(self, prompt: str) -> dict[str, Any]:
        from bot.gpt_client import query_gpt_json_async

        return await query_gpt_json_async(prompt)


def _build_service_bundle(cfg: "BotConfig", offline: bool) -> ServiceBundle:
    """Resolve configured service implementations for the current run."""

    overrides: dict[str, Any] = dict(getattr(cfg, "service_factories", {}) or {})
    if offline:
        try:
            from services.offline import get_offline_service_factories

            for name, spec in get_offline_service_factories().items():
                overrides.setdefault(name, spec)
        except Exception:  # pragma: no cover - defensive import
            logger.exception("Failed to import offline service factories")

    exchange_obj: Any | None = None
    telegram_factory: TelegramLoggerFactory | None = None
    gpt_client: GPTClientProtocol | None = None

    exchange_spec = overrides.get("exchange")
    if exchange_spec is not None:
        exchange_obj = _instantiate_service(exchange_spec)

    telegram_spec = overrides.get("telegram_logger")
    if telegram_spec is not None:
        resolved = _resolve_service_object(telegram_spec)
        if not callable(resolved):
            logger.warning(
                "Telegram logger override %r is not callable; ignoring", telegram_spec
            )
        else:
            telegram_factory = resolved  # type: ignore[assignment]
    if telegram_factory is None:
        from bot.telegram_logger import TelegramLogger

        telegram_factory = TelegramLogger

    gpt_spec = overrides.get("gpt_client")
    if gpt_spec is not None:
        candidate = _instantiate_service(gpt_spec)
        if isinstance(candidate, ModuleGPTClientAdapter) or hasattr(
            candidate, "query_json_async"
        ):
            gpt_client = candidate  # type: ignore[assignment]
        else:
            logger.warning(
                "Configured GPT client %r does not expose expected interface; using default",
                gpt_spec,
            )
    if gpt_client is None:
        gpt_client = ModuleGPTClientAdapter()

    return ServiceBundle(
        exchange=exchange_obj,
        telegram_logger_factory=telegram_factory,
        gpt_client=gpt_client,
    )


def parse_symbols(raw: str | None) -> list[str] | None:
    """Convert a comma-separated string of *raw* symbols to a list."""

    if not raw:
        return None
    symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return symbols or None


def parse_args() -> argparse.Namespace:
    """Return parsed command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the trading bot or a historical simulation")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the configuration JSON file (defaults to config.json)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline stubs to avoid network calls",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated list of symbols to monitor (defaults to BTCUSDT)",
    )

    subparsers = parser.add_subparsers(dest="command")

    trade_parser = subparsers.add_parser("trade", help="Run the live trading loop")
    trade_parser.add_argument(
        "--runtime",
        type=float,
        default=60.0,
        help="Optional runtime limit in seconds (set to 0 for no limit)",
    )

    simulate_parser = subparsers.add_parser("simulate", help="Replay cached market data")
    simulate_parser.add_argument("--start", required=True, help="Simulation start timestamp (YYYY-MM-DD)")
    simulate_parser.add_argument("--end", required=True, help="Simulation end timestamp (YYYY-MM-DD)")
    simulate_parser.add_argument(
        "--speed",
        type=float,
        default=60.0,
        help="Time acceleration factor (1.0 runs in real time)",
    )

    args = parser.parse_args()
    if not args.command:
        args.command = "trade"
    if getattr(args, "runtime", None) is not None and args.runtime <= 0:
        args.runtime = None
    args.symbols = parse_symbols(getattr(args, "symbols", None))
    return args


def configure_environment(args: argparse.Namespace) -> bool:
    """Apply environment configuration and return the resulting offline flag."""

    if args.offline:
        os.environ["OFFLINE_MODE"] = "1"
    offline_env = os.getenv("OFFLINE_MODE", "0").strip().lower()
    offline_mode = offline_env in {"1", "true", "yes", "on"}
    if offline_mode:
        os.environ.setdefault("TEST_MODE", "1")
    return offline_mode


def ensure_directories(cfg: "BotConfig") -> None:
    """Ensure cache and log directories exist and are writable."""

    for attr, fallback in (("cache_dir", "cache"), ("log_dir", "logs")):
        value = Path(getattr(cfg, attr))
        try:
            value.mkdir(parents=True, exist_ok=True)
            if not os.access(value, os.W_OK):  # pragma: no cover - filesystem guard
                raise PermissionError
        except (OSError, PermissionError):
            fallback_path = Path(fallback)
            fallback_path.mkdir(parents=True, exist_ok=True)
            setattr(cfg, attr, str(fallback_path))
            value = fallback_path
        if attr == "log_dir":
            os.environ["LOG_DIR"] = str(value)
        if attr == "cache_dir":
            os.environ["CACHE_DIR"] = str(value)


def prepare_data_handler(handler, cfg: "BotConfig", symbols: list[str] | None) -> None:
    """Populate missing attributes required by downstream services."""

    pairs = symbols or getattr(handler, "usdt_pairs", None)
    if not pairs:
        pairs = ["BTCUSDT"]
    handler.usdt_pairs = list(dict.fromkeys(pairs))

    handler.indicators = getattr(handler, "indicators", {}) or {}
    handler.indicators_2h = getattr(handler, "indicators_2h", {}) or {}

    empty_df = pd.DataFrame()
    for sym in handler.usdt_pairs:
        handler.indicators.setdefault(sym, SimpleNamespace(df=empty_df))
        handler.indicators_2h.setdefault(sym, SimpleNamespace(df=empty_df))

    handler.ohlcv = getattr(handler, "ohlcv", getattr(handler, "_ohlcv", empty_df))
    handler.ohlcv_2h = getattr(handler, "ohlcv_2h", getattr(handler, "_ohlcv_2h", empty_df))

    existing_funding = getattr(handler, "funding_rates", {}) or {}
    handler.funding_rates = {sym: float(existing_funding.get(sym, 0.0)) for sym in handler.usdt_pairs}

    existing_interest = getattr(handler, "open_interest", {}) or {}
    handler.open_interest = {sym: float(existing_interest.get(sym, 0.0)) for sym in handler.usdt_pairs}

    optimizer = getattr(handler, "parameter_optimizer", None)
    if optimizer is None or not hasattr(optimizer, "optimize"):
        async def _return_config(_symbol: str) -> dict[str, object]:
            return cfg.asdict()

        handler.parameter_optimizer = SimpleNamespace(optimize=_return_config)
    else:
        optimize = getattr(optimizer, "optimize", None)
        if optimize is not None and not asyncio.iscoroutinefunction(optimize):
            async def _wrapped(symbol: str, _func=optimize):
                result = _func(symbol)
                if asyncio.iscoroutine(result):
                    return await result
                return result

            optimizer.optimize = _wrapped

    if not hasattr(handler, "is_data_fresh"):
        async def _always_fresh(_symbol: str, timeframe: str = "primary", max_delay: float = 60.0) -> bool:  # noqa: ARG001
            return True

        handler.is_data_fresh = _always_fresh

    if not hasattr(handler, "get_atr"):
        async def _default_atr(_symbol: str) -> float:  # noqa: ARG001
            return float(cfg.get("atr_period_default", 14))

        handler.get_atr = _default_atr


def _log_mode(command: str, offline: bool) -> None:
    mode = "offline" if offline else "online"
    logger.info("Starting %s mode in %s configuration", command, mode)


def _build_components(cfg: "BotConfig", offline: bool, symbols: list[str] | None):
    services = _build_service_bundle(cfg, offline)

    from bot.data_handler import DataHandler
    from bot.model_builder import ModelBuilder
    from bot.trade_manager import TradeManager

    data_handler = DataHandler(cfg, None, None, exchange=services.exchange)
    prepare_data_handler(data_handler, cfg, symbols)

    model_builder = ModelBuilder(cfg, data_handler, None, gpt_client=services.gpt_client)
    trade_manager = TradeManager(
        cfg,
        data_handler,
        model_builder,
        None,
        None,
        telegram_logger_factory=services.telegram_logger_factory,
        gpt_client=services.gpt_client,
    )
    model_builder.trade_manager = trade_manager
    return data_handler, model_builder, trade_manager


async def run_trading_cycle(trade_manager, runtime: float | None) -> None:
    """Execute the trading loop with an optional runtime limit."""

    run_coro = getattr(trade_manager, "run", None)
    if not callable(run_coro):
        logger.warning("TradeManager has no run() coroutine; nothing to execute")
        return

    try:
        if runtime is not None:
            await asyncio.wait_for(run_coro(), timeout=runtime)
        else:
            await run_coro()
    except asyncio.TimeoutError:
        logger.info("Runtime limit reached; stopping trading loop")
    except Exception:
        logger.exception("Trading loop terminated due to an unexpected error")
        raise
    finally:
        stop = getattr(trade_manager, "stop", None)
        if callable(stop):
            with contextlib.suppress(Exception):
                await stop()


async def run_simulation_cycle(args: argparse.Namespace, data_handler, trade_manager) -> None:
    """Run the historical simulator with the provided arguments."""

    from bot.simulation import HistoricalSimulator

    start_ts = pd.to_datetime(args.start, utc=True)
    end_ts = pd.to_datetime(args.end, utc=True)
    simulator = HistoricalSimulator(data_handler, trade_manager)
    await simulator.run(start_ts, end_ts, args.speed)


async def main() -> None:
    args = parse_args()
    offline_mode = configure_environment(args)

    from bot.dotenv_utils import load_dotenv

    load_dotenv()

    import bot.config as config_module

    config_module.OFFLINE_MODE = offline_mode
    from bot.config import load_config
    from bot.utils import configure_logging

    cfg = load_config(args.config)
    ensure_directories(cfg)

    configure_logging()
    _log_mode(args.command, offline_mode)

    data_handler, _model_builder, trade_manager = _build_components(cfg, offline_mode, args.symbols)

    load_initial = getattr(data_handler, "load_initial", None)
    if callable(load_initial):
        try:
            result = load_initial()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Initial data load failed: %s", exc)

    if args.command == "simulate":
        await run_simulation_cycle(args, data_handler, trade_manager)
        stop = getattr(trade_manager, "stop", None)
        if callable(stop):
            with contextlib.suppress(Exception):
                await stop()
    else:
        await run_trading_cycle(trade_manager, getattr(args, "runtime", None))

    logger.info("Execution finished")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
