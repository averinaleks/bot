#!/usr/bin/env python3
"""Command-line entry point for running the trading bot or a simulation."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from bot.config import BotConfig


logger = logging.getLogger("TradingBot")


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


_SAFE_FACTORY_MODULE_PREFIXES = ("bot.", "services.")
_SAFE_FACTORY_ROOTS = frozenset({"bot", "services"})
_REPO_ROOT = Path(__file__).resolve().parent


def _validate_identifier_path(path: str, *, what: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split(".") if part)
    if not parts:
        raise ValueError(f"{what} must contain at least one identifier")
    for part in parts:
        if not part.isidentifier():
            raise ValueError(f"{what} contains invalid identifier segment {part!r}")
    return parts


def _ensure_safe_factory_module(module_name: str) -> None:
    if module_name in _SAFE_FACTORY_ROOTS:
        return
    if any(module_name.startswith(prefix) for prefix in _SAFE_FACTORY_MODULE_PREFIXES):
        return
    raise ValueError(
        "Factory modules must reside within the bot or services packages"
    )


def _load_factory_module(module_name: str) -> ModuleType:
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.loader is None or getattr(spec.loader, "exec_module", None) is None:
        raise ValueError(f"Unable to resolve factory module {module_name!r}")

    origin = getattr(spec, "origin", None)
    if not origin:
        raise ValueError(f"Factory module {module_name!r} has no import origin")

    module_path = Path(origin).resolve()
    try:
        module_path.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Factory module {module_name!r} must reside within the project directory"
        ) from exc

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _import_from_path(path: str) -> Any:
    module_name, _, attr_path = path.partition(":")
    if not module_name or not attr_path:
        raise ValueError(f"Invalid factory path: {path!r}")
    _validate_identifier_path(module_name, what="Module name")
    _ensure_safe_factory_module(module_name)
    attr_parts = _validate_identifier_path(attr_path, what="Attribute path")
    module = _load_factory_module(module_name)
    target: Any = module
    for part in attr_parts:
        target = getattr(target, part)
    return target


def _resolve_factory(cfg: "BotConfig", name: str) -> Callable[..., Any] | type | None:
    mapping = getattr(cfg, "service_factories", {}) or {}
    raw = mapping.get(name)
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return _import_from_path(raw)
        except (ImportError, AttributeError, ValueError) as exc:
            logger.error("Failed to import factory for %s from %s: %s", name, raw, exc)
            raise
    return raw


def _instantiate_factory(factory: Callable[..., Any] | type | None, cfg: "BotConfig") -> Any:
    if factory is None:
        return None
    for creator in (
        lambda: factory(cfg),
        lambda: factory(config=cfg),
        lambda: factory(),
    ):
        try:
            return creator()
        except TypeError:
            continue
    logger.warning("Failed to instantiate %r with supported signatures", factory)
    return None


def _build_components(cfg: "BotConfig", offline: bool, symbols: list[str] | None):
    service_factories = dict(getattr(cfg, "service_factories", {}) or {})
    if offline:
        from services.offline import OFFLINE_SERVICE_FACTORIES

        for key, value in OFFLINE_SERVICE_FACTORIES.items():
            service_factories.setdefault(key, value)
    cfg.service_factories = service_factories

    def _load_factory(name: str, *, optional: bool = False):
        try:
            factory = _resolve_factory(cfg, name)
        except (ImportError, AttributeError, ValueError) as exc:
            configured = service_factories.get(name)
            hint = "; no offline fallback is available" if not offline else ""
            raise ValueError(
                "Failed to load service factory %r from %r: %s%s"
                % (name, configured, exc, hint)
            ) from exc
        if factory is None:
            if optional:
                return None
            hint = "; no offline fallback is available" if not offline else ""
            raise ValueError(
                "No service factory configured for %r%s" % (name, hint)
            )
        return factory

    exchange_factory = _load_factory("exchange")
    exchange = _instantiate_factory(exchange_factory, cfg)

    from bot.data_handler import DataHandler
    from bot.model_builder import ModelBuilder
    from bot.trade_manager import TradeManager

    telegram_factory = _load_factory("telegram_logger", optional=True)
    gpt_factory = _load_factory("gpt_client", optional=True)

    data_handler = DataHandler(cfg, None, None, exchange=exchange)
    prepare_data_handler(data_handler, cfg, symbols)

    model_builder = ModelBuilder(cfg, data_handler, None, gpt_client_factory=gpt_factory)
    trade_manager = TradeManager(
        cfg,
        data_handler,
        model_builder,
        None,
        None,
        telegram_logger_factory=telegram_factory,
        gpt_client_factory=gpt_factory,
    )
    model_builder.trade_manager = trade_manager
    return data_handler, model_builder, trade_manager


async def run_trading_cycle(trade_manager, runtime: float | None) -> None:
    """Execute the trading loop with an optional runtime limit."""

    run_coro = getattr(trade_manager, "run", None)
    if not callable(run_coro):
        logger.warning("TradeManager has no run() coroutine; nothing to execute")
        return

    domain_error_map: dict[type[BaseException], str] = {}
    module_name = getattr(type(trade_manager), "__module__", "")
    module_candidates = [module_name]
    if module_name and "." in module_name:
        module_candidates.append(module_name.rsplit(".", 1)[0])

    seen: set[type[BaseException]] = set()
    collected: list[tuple[type[BaseException], str]] = []
    for candidate in module_candidates:
        module = sys.modules.get(candidate)
        if module is None:
            continue
        for attr in ("TradeManagerTaskError", "InvalidHostError"):
            error_cls = getattr(module, attr, None)
            if (
                isinstance(error_cls, type)
                and issubclass(error_cls, BaseException)
                and error_cls not in seen
            ):
                seen.add(error_cls)
                collected.append((error_cls, attr))

    if collected:
        domain_error_map = {cls: attr for cls, attr in collected}

    try:
        if runtime is not None:
            await asyncio.wait_for(run_coro(), timeout=runtime)
        else:
            await run_coro()
    except asyncio.TimeoutError:
        logger.info("Runtime limit reached; stopping trading loop")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        matched_attr: str | None = None
        matched_cls: type[BaseException] | None = None
        for cls, attr in domain_error_map.items():
            if isinstance(exc, cls):
                matched_attr = attr
                matched_cls = cls
                break

        if matched_attr is not None:
            message = "Trading loop aborted after TradeManager error"
            logger.error("%s: %s", message, exc, exc_info=True)

            original_args = getattr(exc, "args", ())
            enriched_args = (f"{message}: {exc}",)
            if original_args:
                enriched_args += tuple(original_args[1:])

            new_exc: BaseException | None = None
            if matched_cls is not None:
                try:
                    new_exc = matched_cls(*original_args)
                except Exception:  # pragma: no cover - defensive
                    new_exc = None

            if new_exc is None:
                new_exc = matched_cls(enriched_args[0]) if matched_cls else RuntimeError(enriched_args[0])
            assert new_exc is not None
            try:
                new_exc.args = enriched_args
            except Exception as assignment_error:  # pragma: no cover - defensive
                logger.debug(
                    "Не удалось заменить аргументы для исключения %s: %s",
                    type(new_exc).__name__,
                    assignment_error,
                )
            if hasattr(exc, "__dict__") and hasattr(new_exc, "__dict__"):
                new_exc.__dict__.update({k: v for k, v in exc.__dict__.items() if k not in new_exc.__dict__})

            raise new_exc.with_traceback(exc.__traceback__) from exc
        logger.exception("Unexpected error during trading loop")
        raise
    finally:
        stop = getattr(trade_manager, "stop", None)
        if callable(stop):
            with contextlib.suppress(Exception):
                await stop()


async def run_simulation_cycle(args: argparse.Namespace, data_handler, trade_manager) -> None:
    """Run the historical simulator with the provided arguments."""

    from bot.simulation import HistoricalSimulator, SimulationDataError

    start_ts = pd.to_datetime(args.start, utc=True)
    end_ts = pd.to_datetime(args.end, utc=True)
    simulator = HistoricalSimulator(data_handler, trade_manager)
    try:
        result = await simulator.run(start_ts, end_ts, args.speed)
    except SimulationDataError as exc:
        logger.error("Simulation failed: %s", exc)
        return

    if result.missing_symbols:
        logger.warning(
            "Symbols without data: %s",
            ", ".join(result.missing_symbols),
        )
    logger.info(
        "Simulation completed: %d iterations, %d updates across %d symbols",
        result.total_iterations,
        result.total_updates,
        len(result.processed_symbols),
    )


async def _maybe_load_initial(data_handler: Any) -> None:
    """Attempt to invoke ``load_initial`` on the data handler with safe error handling."""

    load_initial = getattr(data_handler, "load_initial", None)
    if not callable(load_initial):
        return

    try:
        result = load_initial()
        if asyncio.iscoroutine(result):
            await result
    except (OSError, IOError, RuntimeError) as exc:  # noqa: PERF203 - narrow defensive guard
        logger.warning("Initial data load failed: %s", exc, exc_info=True)


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

    await _maybe_load_initial(data_handler)

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
