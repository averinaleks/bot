#!/usr/bin/env python3
"""Command-line entry point for running the trading bot or a simulation."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import logging
import os
import sys
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from bot.config import BotConfig


logger = logging.getLogger("TradingBot")


def _assert_project_layout() -> None:
    """Проверяет, что запущен полный проект, а не урезанная копия."""

    required_modules = (
        "services",
        "data_handler",
        "model_builder",
        "bot",
    )

    repo_root = Path(__file__).resolve().parent

    def _is_in_repo(module_name: str) -> bool:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False

        candidates = []
        if spec.submodule_search_locations:
            candidates.extend(spec.submodule_search_locations)
        if spec.origin:
            candidates.append(spec.origin)

        for candidate in candidates:
            try:
                Path(candidate).resolve().relative_to(repo_root)
            except (OSError, ValueError):
                continue
            return True
        return False

    missing = []
    foreign = []
    for name in required_modules:
        spec = importlib.util.find_spec(name)
        if spec is None:
            missing.append(name)
            continue
        if not _is_in_repo(name):
            foreign.append((name, spec.origin or "<unknown>"))

    if missing or foreign:
        problems = []
        if missing:
            problems.append(
                "отсутствуют: %s" % ", ".join(missing)
            )
        if foreign:
            details = "; ".join(f"{name} -> {origin}" for name, origin in foreign)
            problems.append(
                "найдены сторонние модули (ожидались из %s): %s" % (repo_root, details)
            )

        raise SystemExit(
            "Отсутствуют необходимые модули проекта или найдено постороннее содержимое (%s).\n"
            "Убедитесь, что репозиторий склонирован полностью и каталог %s содержит "
            "подкаталоги services, data_handler, model_builder и bot." % ("; ".join(problems), repo_root)
        )


def _ensure_data_handler_package() -> None:
    """Prevent lingering test stubs from breaking ``data_handler`` imports."""

    module = sys.modules.get("data_handler")
    if module is None:
        return
    if isinstance(module, ModuleType) and getattr(module, "__path__", None):
        return
    sys.modules.pop("data_handler", None)


_ensure_data_handler_package()


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
        "--auto-offline",
        action="store_true",
        help="Automatically enable offline mode when required secrets are missing",
    )
    parser.add_argument(
        "--runtime",
        type=float,
        default=60.0,
        help="Optional runtime limit in seconds for trade mode (set to 0 for no limit)",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated list of symbols to monitor (defaults to BTCUSDT)",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("trade", help="Run the live trading loop")

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
    if args.command != "trade":
        args.runtime = None
    elif getattr(args, "runtime", None) is not None and args.runtime <= 0:
        args.runtime = None
    args.symbols = parse_symbols(getattr(args, "symbols", None))
    return args


def configure_environment(args: argparse.Namespace) -> bool:
    """Apply environment configuration and return the resulting offline flag."""

    # Load .env early without importing project packages to avoid triggering
    # config validation before we can decide on a safe mode.
    env_file_values: dict[str, str] = {}
    try:
        from dotenv import dotenv_values as _dotenv_values  # type: ignore

        env_file_values = {k: v for k, v in _dotenv_values().items() if v is not None}
    except Exception:
        env_file_values = {}

    if args.offline:
        os.environ["OFFLINE_MODE"] = "1"

    offline_env = os.getenv("OFFLINE_MODE", "0").strip().lower()
    offline_mode = offline_env in {"1", "true", "yes", "on"}

    required_keys = (
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TRADE_MANAGER_TOKEN",
        "TRADE_RISK_USD",
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
    )
    missing = [key for key in required_keys if not (os.getenv(key) or env_file_values.get(key))]

    if missing and not offline_mode:
        if args.auto_offline:
            offline_mode = True
            os.environ["OFFLINE_MODE"] = "1"
            logger.warning(
                "OFFLINE_MODE=1 включён автоматически (--auto-offline): отсутствуют обязательные переменные: %s",
                ", ".join(missing),
            )
        else:
            missing_msg = ", ".join(missing)
            raise SystemExit(
                "Отсутствуют обязательные переменные окружения: %s. "
                "Укажите их в .env/окружении или запустите бота с флагом --offline "
                "(или --auto-offline для автоматического перехода)." % missing_msg
            )

    if offline_mode:
        os.environ.setdefault("TEST_MODE", "1")
        try:
            from services import offline as offline_env_module
        except ImportError as exc:
            raise SystemExit(
                "OFFLINE_MODE=1: не удалось импортировать services.offline. "
                "Убедитесь, что каталог services/offline.py присутствует в проекте."
            ) from exc

        offline_env_module.ensure_offline_env()
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



def _resolve_dataset(handler: Any, primary: str, legacy: str) -> Any:
    value = getattr(handler, primary, None)
    if value is not None:
        return value
    legacy_value = getattr(handler, legacy, None)
    if legacy_value is not None:
        return legacy_value
    try:  # Local import keeps pandas optional
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return ()
    return pd.DataFrame()


def prepare_data_handler(handler, cfg: "BotConfig", symbols: list[str] | None) -> None:
    """Populate missing attributes required by downstream services."""

    pairs = symbols or getattr(handler, "usdt_pairs", None)
    if not pairs:
        pairs = ["BTCUSDT"]
    handler.usdt_pairs = list(dict.fromkeys(pairs))

    indicators = getattr(handler, "indicators", {}) or {}
    indicators_2h = getattr(handler, "indicators_2h", {}) or {}
    handler.indicators = dict(indicators)
    handler.indicators_2h = dict(indicators_2h)

    primary_dataset = _resolve_dataset(handler, "ohlcv", "_ohlcv")
    secondary_dataset = _resolve_dataset(handler, "ohlcv_2h", "_ohlcv_2h")
    handler.ohlcv = primary_dataset
    handler.ohlcv_2h = secondary_dataset

    for sym in handler.usdt_pairs:
        entry = handler.indicators.setdefault(sym, SimpleNamespace())
        if not hasattr(entry, "df"):
            entry.df = primary_dataset
        entry_2h = handler.indicators_2h.setdefault(sym, SimpleNamespace())
        if not hasattr(entry_2h, "df"):
            entry_2h.df = secondary_dataset

    existing_funding = getattr(handler, "funding_rates", {}) or {}
    handler.funding_rates = {sym: float(existing_funding.get(sym, 0.0)) for sym in handler.usdt_pairs}

    existing_interest = getattr(handler, "open_interest", {}) or {}
    handler.open_interest = {sym: float(existing_interest.get(sym, 0.0)) for sym in handler.usdt_pairs}

    optimizer = getattr(handler, "parameter_optimizer", None)
    if optimizer is None or not hasattr(optimizer, "optimize"):
        async def _return_config(_symbol: str) -> dict[str, object]:  # noqa: ARG001
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


_SAFE_FACTORY_MODULE_PREFIXES = ("bot.", "services.", "model_builder.", "data_handler.")
_SAFE_FACTORY_ROOTS = frozenset({"bot", "services", "model_builder", "data_handler"})
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


def _call_factory(factory: Callable[..., Any] | type, *args: Any, **kwargs: Any) -> Any:
    return factory(*args, **kwargs)


def _type_error_from_adapter(exc: TypeError) -> bool:
    tb = exc.__traceback__
    saw_adapter = False
    while tb is not None:
        frame = tb.tb_frame
        if frame.f_globals.get("__name__") == __name__ and frame.f_code.co_name == "_call_factory":
            saw_adapter = True
        elif saw_adapter:
            return False
        tb = tb.tb_next
    return saw_adapter


def _instantiate_factory(factory: Callable[..., Any] | type | None, cfg: "BotConfig") -> Any:
    if factory is None:
        return None

    signature = inspect.signature(factory)
    attempts: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
        ((cfg,), {}),
        ((), {"config": cfg}),
        ((), {}),
    )

    for args, kwargs in attempts:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        try:
            return _call_factory(factory, *args, **kwargs)
        except TypeError as exc:
            if not _type_error_from_adapter(exc):
                raise
            continue

    logger.warning(
        "Не удалось создать экземпляр %r: поддерживаются сигнатуры factory(cfg), factory(config=cfg) или factory()",
        factory,
    )
    return None


def _is_offline_override(candidate: Any) -> bool:
    """Return ``True`` if *candidate* already points to an offline stub."""

    if candidate is None:
        return False

    if isinstance(candidate, str):
        return "offline" in candidate.lower()

    target = candidate
    if not callable(candidate):
        target = type(candidate)

    if getattr(target, "__offline_stub__", False):
        return True

    module = getattr(target, "__module__", "")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", ""))
    fingerprint = f"{module}.{qualname}".lower()
    return "offline" in fingerprint


def _build_components(cfg: "BotConfig", offline: bool, symbols: list[str] | None):
    service_factories = dict(getattr(cfg, "service_factories", {}) or {})
    if offline:
        from services.offline import OFFLINE_SERVICE_FACTORIES

        forced_offline_keys = {
            "exchange",
            "telegram_logger",
            "gpt_client",
            "model_builder",
            "trade_manager",
        }
        for key, value in OFFLINE_SERVICE_FACTORIES.items():
            if key in forced_offline_keys:
                current = service_factories.get(key)
                if not _is_offline_override(current):
                    service_factories[key] = value
            else:
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
            if offline:
                hint = "; add an entry to service_factories in config.json"
            else:
                hint = (
                    "; add an entry to service_factories in config.json or run this "
                    "script with --offline"
                )
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
    model_builder_factory = _resolve_factory(cfg, "model_builder") or ModelBuilder
    trade_manager_factory = _resolve_factory(cfg, "trade_manager") or TradeManager

    data_handler = DataHandler(cfg, None, None, exchange=exchange)
    prepare_data_handler(data_handler, cfg, symbols)

    model_builder = _call_factory(
        model_builder_factory, cfg, data_handler, None, gpt_client_factory=gpt_factory
    )
    if offline:
        BotOfflineModelBuilder = None
        try:
            from bot.model_builder.offline import (
                OfflineModelBuilder as BotOfflineModelBuilder,
            )
        except Exception:
            try:
                from model_builder.offline import OfflineModelBuilder as BotOfflineModelBuilder
            except Exception:  # pragma: no cover - fallback when offline stub missing
                BotOfflineModelBuilder = None  # type: ignore[assignment]

        if BotOfflineModelBuilder is not None and not isinstance(
            model_builder, BotOfflineModelBuilder
        ):
            model_builder = BotOfflineModelBuilder(
                cfg, data_handler, None, gpt_client_factory=gpt_factory
            )
    trade_manager = _call_factory(
        trade_manager_factory,
        cfg,
        data_handler,
        model_builder,
        None,
        None,
        telegram_logger_factory=telegram_factory,
        gpt_client_factory=gpt_factory,
    )
    if hasattr(model_builder, "trade_manager"):
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

            if new_exc is None:  # pragma: no cover - defensive fallback
                raise RuntimeError(enriched_args[0])
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

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - simulation requires pandas
        raise RuntimeError(
            "Режим симуляции требует установленный pandas для разбора дат"
        ) from exc

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
    _assert_project_layout()
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
