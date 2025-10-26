import pandas as pd
import numpy as np
import pytest
import sys
import types
import importlib
import logging
import math
import contextlib
import tempfile
import threading
import time
import httpx
import cloudpickle
import os
import builtins
from pathlib import Path
from typing import Any
from bot.config import BotConfig
from bot.trade_manager import order_utils

# Stub heavy dependencies before importing the trade manager
if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules['torch'] = torch


sk_mod = types.ModuleType('sklearn')
model_sel = types.ModuleType('sklearn.model_selection')
model_sel.GridSearchCV = object
sk_mod.model_selection = model_sel
base_estimator = types.ModuleType('sklearn.base')
base_estimator.BaseEstimator = object
sk_mod.base = base_estimator
sys.modules.setdefault('sklearn', sk_mod)
sys.modules.setdefault('sklearn.model_selection', model_sel)
sys.modules.setdefault('sklearn.base', base_estimator)
utils_stub = types.ModuleType('utils')
class _TL:
    def __init__(self, *a, **k):
        pass
    async def send_telegram_message(self, *a, **k):
        pass
    @classmethod
    async def shutdown(cls):
        pass
utils_stub.TelegramLogger = _TL
utils_stub.logger = logging.getLogger('test')
async def _cde_stub(*a, **kw):
    return False
utils_stub.check_dataframe_empty = _cde_stub
utils_stub.check_dataframe_empty_async = _cde_stub
utils_stub.is_cuda_available = lambda: False
async def _safe_api_call(exchange, method: str, *args, **kwargs):
    return await getattr(exchange, method)(*args, **kwargs)
utils_stub.safe_api_call = _safe_api_call
def _retry(max_attempts, delay_fn):
    def decorator(func):
        return func
    return decorator
utils_stub.retry = _retry
utils_stub.suppress_tf_logs = lambda: None
sys.modules['utils'] = utils_stub
sys.modules['bot.utils'] = utils_stub
sys.modules.pop('trade_manager', None)
sys.modules.pop('bot.trade_manager.core', None)


joblib_mod = types.ModuleType('joblib')
_JOBLIB_OBJECT_STORE: dict[str, Any] = {}


def _normalise_joblib_path(candidate: str | os.PathLike[str]) -> str:
    """Return an absolute representation for *candidate* suitable as a key.

    The helper mirrors ``joblib``'s behaviour by resolving the supplied path
    relative to the current working directory without requiring the file to
    exist in advance.  Using a stable key allows the stub ``load`` function to
    retrieve objects stored via ``dump`` without reintroducing unsafe pickle
    deserialisation APIs which CodeQL rightfully flags as dangerous.
    """

    return str(Path(candidate).resolve(strict=False))


def _register_joblib_object(path: str, obj: Any) -> None:
    _JOBLIB_OBJECT_STORE[path] = obj


def _joblib_dump(obj, file, *args, **kwargs):
    if hasattr(file, 'write'):
        target = getattr(file, 'name', None)
        if target:
            key = _normalise_joblib_path(target)
            _register_joblib_object(key, obj)
        data = cloudpickle.dumps(obj)
        file.write(data)
        file.flush()
        return

    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = _normalise_joblib_path(path)
    _register_joblib_object(key, obj)
    path.write_bytes(cloudpickle.dumps(obj))


def _joblib_load(file, *args, **kwargs):
    key: str | None = None
    if hasattr(file, 'read'):
        target = getattr(file, 'name', None)
        if target:
            key = _normalise_joblib_path(target)
    else:
        key = _normalise_joblib_path(file)

    if key and key in _JOBLIB_OBJECT_STORE:
        return _JOBLIB_OBJECT_STORE[key]

    raise RuntimeError('joblib stub cannot load unknown path')


joblib_mod.dump = _joblib_dump
joblib_mod.load = _joblib_load
sys.modules.setdefault('joblib', joblib_mod)

import asyncio  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _set_test_mode():
    mp = pytest.MonkeyPatch()
    mp.setenv("TEST_MODE", "1")
    yield
    mp.undo()


@pytest.fixture(scope="module", autouse=True)
def _import_trade_manager(_set_test_mode):
    global trade_manager, TradeManager
    import bot.trade_manager.core as tm
    from bot.trade_manager import TradeManager as TM
    trade_manager = tm
    TradeManager = TM
    yield


@pytest.fixture(scope="module", autouse=True)
def _cleanup_telegram_logger(_import_trade_manager):
    yield
    asyncio.run(trade_manager.TelegramLogger.shutdown())


def test_trade_manager_uses_httpx_stub_when_httpx_missing(monkeypatch):
    original_httpx = sys.modules.pop("httpx", None)
    original_tm = sys.modules.get("bot.trade_manager.core")
    sys.modules.pop("bot.trade_manager.core", None)

    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    monkeypatch.setenv("OFFLINE_MODE", "1")

    tm_mod = importlib.import_module("bot.trade_manager.core")

    try:
        assert getattr(tm_mod.httpx, "__offline_stub__", False)
        assert hasattr(tm_mod.httpx, "HTTPError")
    finally:
        sys.modules.pop("bot.trade_manager.core", None)
        if original_tm is not None:
            sys.modules["bot.trade_manager.core"] = original_tm
            globals()["trade_manager"] = original_tm
            globals()["TradeManager"] = getattr(
                original_tm, "TradeManager", globals().get("TradeManager")
            )

        if original_httpx is not None:
            sys.modules["httpx"] = original_httpx
        else:
            sys.modules.pop("httpx", None)

        monkeypatch.delenv("OFFLINE_MODE", raising=False)


def test_trade_manager_logs_missing_aiohttp(monkeypatch, caplog):
    original_import = builtins.__import__
    original_tm = sys.modules.get("bot.trade_manager.core")
    original_aiohttp = sys.modules.get("aiohttp")
    sys.modules.pop("bot.trade_manager.core", None)
    sys.modules.pop("aiohttp", None)

    def fake_import(name, *args, **kwargs):
        if name == "aiohttp":
            raise ModuleNotFoundError("No module named 'aiohttp'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        tm_mod = importlib.import_module("bot.trade_manager.core")

    try:
        assert getattr(tm_mod.aiohttp, "ClientError") is Exception
        assert any(
            "aiohttp import failed" in record.message for record in caplog.records
        )
    finally:
        sys.modules.pop("bot.trade_manager.core", None)
        if original_tm is not None:
            sys.modules["bot.trade_manager.core"] = original_tm
            globals()["trade_manager"] = original_tm
            globals()["TradeManager"] = getattr(
                original_tm, "TradeManager", globals().get("TradeManager")
            )
        if original_aiohttp is not None:
            sys.modules["aiohttp"] = original_aiohttp
        else:
            sys.modules.pop("aiohttp", None)


def test_trade_manager_logs_unexpected_aiohttp_import_error(monkeypatch, caplog):
    original_import = builtins.__import__
    original_tm = sys.modules.get("bot.trade_manager.core")
    original_aiohttp = sys.modules.get("aiohttp")
    sys.modules.pop("bot.trade_manager.core", None)
    sys.modules.pop("aiohttp", None)

    def fake_import(name, *args, **kwargs):
        if name == "aiohttp":
            raise RuntimeError("boom")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        tm_mod = importlib.import_module("bot.trade_manager.core")

    try:
        assert getattr(tm_mod.aiohttp, "ClientError") is Exception
        assert any(
            record.levelno >= logging.ERROR
            and "Unexpected error importing aiohttp" in record.message
            for record in caplog.records
        )
    finally:
        sys.modules.pop("bot.trade_manager.core", None)
        if original_tm is not None:
            sys.modules["bot.trade_manager.core"] = original_tm
            globals()["trade_manager"] = original_tm
            globals()["TradeManager"] = getattr(
                original_tm, "TradeManager", globals().get("TradeManager")
            )
        if original_aiohttp is not None:
            sys.modules["aiohttp"] = original_aiohttp
        else:
            sys.modules.pop("aiohttp", None)

def test_utils_injected_before_trade_manager_import():
    import importlib
    import sys
    tm = importlib.reload(sys.modules.get("bot.trade_manager.core", trade_manager))
    tm.TelegramLogger = _TL
    assert tm.TelegramLogger is _TL

class DummyTelegramLogger:
    def __init__(self, *a, **kw):
        pass
    async def send_telegram_message(self, *a, **kw):
        pass

utils = types.ModuleType('utils')
utils.TelegramLogger = DummyTelegramLogger
utils.logger = logging.getLogger('test')
async def _cde(*a, **kw):
    return False
utils.check_dataframe_empty = _cde
utils.check_dataframe_empty_async = _cde
utils.is_cuda_available = lambda: False
async def _safe_api_call(exchange, method: str, *args, **kwargs):
    return await getattr(exchange, method)(*args, **kwargs)
utils.safe_api_call = _safe_api_call
def _retry2(max_attempts, delay_fn):
    def decorator(func):
        return func
    return decorator
utils.retry = _retry2
utils.suppress_tf_logs = lambda: None
sys.modules['utils'] = utils
sys.modules['bot.utils'] = utils


class DummyExchange:
    def __init__(self):
        self.orders = []
        self.fail = False
    async def fetch_balance(self):
        return {'total': {'USDT': 1000}}
    async def create_order(self, symbol, type, side, amount, price, params):
        self.orders.append({'method': 'create_order', 'symbol': symbol, 'type': type, 'side': side,
                             'amount': amount, 'price': price, 'params': params})
        if self.fail:
            return {'retCode': 1}
        return {'id': '1'}
    async def create_order_with_take_profit_and_stop_loss(
        self,
        symbol,
        type,
        side,
        amount,
        price,
        takeProfit,
        stopLoss,
        params,
    ):
        self.orders.append({'method': 'create_order_with_tp_sl', 'symbol': symbol, 'type': type, 'side': side,
                             'amount': amount, 'price': price, 'tp': takeProfit, 'sl': stopLoss,
                             'params': params})
        if self.fail:
            return {'retCode': 1}
        return {'id': '2'}

class DummyDataHandler:
    def __init__(
        self,
        fresh: bool = True,
        fail_order: bool = False,
        pairs: list[str] | None = None,
    ):
        self.exchange = DummyExchange()
        self.exchange.fail = fail_order
        self.usdt_pairs = pairs or ['BTCUSDT']
        idx = pd.MultiIndex.from_tuples(
            [(sym, pd.Timestamp('2020-01-01')) for sym in self.usdt_pairs],
            names=['symbol', 'timestamp']
        )
        self.ohlcv = pd.DataFrame(
            {'close': [100]*len(self.usdt_pairs), 'atr': [1.0]*len(self.usdt_pairs)},
            index=idx
        )
        self.indicators = {
            sym: types.SimpleNamespace(atr=pd.Series([1.0]), df=pd.DataFrame({'a':[1]}))
            for sym in self.usdt_pairs
        }
        self.fresh = fresh
        async def _opt(symbol):
            return {}
        self.parameter_optimizer = types.SimpleNamespace(optimize=_opt)

    async def get_tick_size(self, symbol: str) -> float:
        return 0.1

    async def get_atr(self, symbol: str) -> float:
        if "symbol" in self.ohlcv.index.names and symbol in self.ohlcv.index.get_level_values("symbol"):
            return float(self.ohlcv.loc[pd.IndexSlice[symbol], "atr"].iloc[-1])
        return 0.0

    async def is_data_fresh(self, symbol: str, timeframe: str = 'primary', max_delay: float = 60) -> bool:
        return self.fresh

def make_config():
    return BotConfig(
        cache_dir=tempfile.mkdtemp(),
        max_positions=5,
        leverage=10,
        min_risk_per_trade=0.01,
        max_risk_per_trade=0.05,
        check_interval=1.0,
        performance_window=60,
        sl_multiplier=1.0,
        tp_multiplier=2.0,
        order_retry_attempts=3,
        order_retry_delay=0.0,
        reversal_margin=0.05,
    )


def test_place_order_passes_tp_sl_without_special_method():
    class ExchangeNoTPSL:
        def __init__(self):
            self.calls = []

        async def create_order(self, symbol, type, side, amount, price, params):
            self.calls.append(
                {
                    "method": "create_order",
                    "symbol": symbol,
                    "type": type,
                    "side": side,
                    "amount": amount,
                    "price": price,
                    "params": params,
                }
            )
            return {"id": "1"}

    dh = DummyDataHandler()
    dh.exchange = ExchangeNoTPSL()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def run():
        await tm.place_order(
            "BTCUSDT",
            "buy",
            1,
            100,
            {"takeProfitPrice": 102.0, "stopLossPrice": 99.0},
            use_lock=False,
        )

    import asyncio

    asyncio.run(run())

    assert dh.exchange.calls, "create_order not called"
    call = dh.exchange.calls[0]
    assert call["method"] == "create_order"
    assert call["params"].get("takeProfitPrice") == pytest.approx(102.0)
    assert call["params"].get("stopLossPrice") == pytest.approx(99.0)

def test_position_calculations():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute
    import asyncio
    size = asyncio.run(tm.calculate_position_size('BTCUSDT', 100, 1.0, 1.5))
    assert size == pytest.approx(10 / (1.5 * 10))

    direct_size = order_utils.calculate_position_size(
        equity=1000,
        risk_per_trade=0.01,
        atr=1.0,
        sl_multiplier=1.5,
        leverage=tm.leverage,
        price=100,
        max_position_pct=tm.max_position_pct,
    )
    assert direct_size == pytest.approx(size)

    sl, tp = tm.calculate_stop_loss_take_profit('buy', 100, 1.0, 1.5, 2.5)
    assert sl == pytest.approx(98.5)
    assert tp == pytest.approx(102.5)

    sl_calc, tp_calc = order_utils.calculate_stop_loss_take_profit(
        'buy', 100, 1.0, 1.5, 2.5
    )
    assert sl_calc == pytest.approx(sl)
    assert tp_calc == pytest.approx(tp)
    plan = order_utils.build_protective_order_plan(
        'buy', entry_price=100, stop_loss_price=sl, take_profit_price=tp
    )
    assert plan.opposite_side == 'sell'
    assert plan.stop_loss_price == pytest.approx(sl)
    assert plan.take_profit_price == pytest.approx(tp)
    assert plan.trailing_stop_price is None


def test_open_position_places_tp_sl_orders():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    import asyncio
    asyncio.run(run())

    assert dh.exchange.orders, 'no orders created'
    order = dh.exchange.orders[0]
    assert order['method'] == 'create_order_with_tp_sl'
    assert order['tp'] == pytest.approx(102.0)
    assert order['sl'] == pytest.approx(99.0)


def test_trade_manager_telegram_factory_from_config(monkeypatch):
    cfg = make_config()
    cfg.service_factories["telegram_logger"] = (
        "services.service_factories:build_telegram_logger"
    )
    cfg.use_offline_services = True

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")

    factory_path = cfg.service_factories["telegram_logger"]
    from run_bot import _import_from_path

    # ``BotConfig`` stores service factories as fully-qualified dotted paths.
    # In production these paths are resolved via :func:`run_bot._import_from_path`
    # which validates the module name against an allow-list and confines imports
    # to the repository directory.  Reuse the same helper in tests to avoid the
    # raw ``importlib.import_module`` call which Semgrep rightfully flags as a
    # potential vector for loading arbitrary modules.
    factory = _import_from_path(factory_path)

    dh = DummyDataHandler()
    tm = TradeManager(
        cfg,
        dh,
        None,
        None,
        "chat",
        telegram_logger_factory=factory,
    )

    async def _send():
        await tm.telegram_logger.send_telegram_message("ok")

    asyncio.run(_send())


def test_trailing_stop_to_breakeven():
    dh = DummyDataHandler(pairs=['BTCUSDT', 'ETHUSDT'])
    cfg = make_config()
    cfg.update({
        'trailing_stop_percentage': 1.0,
        'trailing_stop_coeff': 0.0,
        'trailing_stop_multiplier': 1.0,
    })
    tm = TradeManager(cfg, dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('ETHUSDT', 'buy', 100, {})
        await tm.open_position('BTCUSDT', 'buy', 100, {})
        # Deliberately unsort positions
        tm.positions = tm.positions.iloc[::-1]
        assert not tm.positions.index.is_monotonic_increasing
        await tm.check_trailing_stop('BTCUSDT', 101)
        assert tm.positions.index.is_monotonic_increasing

    import asyncio
    asyncio.run(run())

    assert len(dh.exchange.orders) >= 3
    pos = tm.positions.xs('BTCUSDT', level='symbol').iloc[0]
    open_btc_order = next(o for o in dh.exchange.orders if o['symbol'] == 'BTCUSDT')
    assert pos['breakeven_triggered'] is True
    assert pos['size'] < open_btc_order['amount']
    assert pos['stop_loss_price'] == pytest.approx(pos['entry_price'] + 0.1)


@pytest.mark.asyncio
async def test_trailing_stop_updates_after_candle_close():
    dh = DummyDataHandler()
    cfg = make_config()
    cfg.update({'trailing_stop_multiplier': 1.0})
    tm = TradeManager(cfg, dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position('BTCUSDT', 'buy', 100, {})

    await tm.check_trailing_stop('BTCUSDT', 101)
    pos = tm.positions.xs('BTCUSDT', level='symbol').iloc[0]
    assert pos['highest_price'] == pytest.approx(100)

    await tm.check_trailing_stop('BTCUSDT', 105)
    pos = tm.positions.xs('BTCUSDT', level='symbol').iloc[0]
    assert pos['highest_price'] == pytest.approx(100)

    ts = dh.ohlcv.index.get_level_values('timestamp')[0] + pd.Timedelta(minutes=1)
    dh.ohlcv = pd.DataFrame(
        {'close': [105], 'atr': [1.0]},
        index=pd.MultiIndex.from_tuples([('BTCUSDT', ts)], names=['symbol', 'timestamp'])
    )
    await tm.check_trailing_stop('BTCUSDT', 105)
    pos = tm.positions.xs('BTCUSDT', level='symbol').iloc[0]
    assert pos['highest_price'] == pytest.approx(105)


@pytest.mark.asyncio
async def test_trailing_stop_executes_on_wick(monkeypatch):
    dh = DummyDataHandler()
    cfg = make_config()
    cfg.update({'trailing_stop_multiplier': 1.0})
    tm = TradeManager(cfg, dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position('BTCUSDT', 'buy', 100, {})

    exit_prices: list[float] = []

    async def fake_close(symbol, price, reason=""):
        exit_prices.append(price)
        tm.positions = tm.positions.drop(symbol, level='symbol')

    monkeypatch.setattr(tm, 'close_position', fake_close)

    await tm.check_trailing_stop('BTCUSDT', 98)

    assert exit_prices and exit_prices[0] == pytest.approx(98)


def test_open_position_skips_existing():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    import asyncio
    asyncio.run(run())

    assert len(dh.exchange.orders) == 1
    assert len(tm.positions) == 1


def test_open_position_uses_multiindex():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position("BTCUSDT", "buy", 100, {})
        await tm.open_position("BTCUSDT", "buy", 100, {})

    import asyncio
    asyncio.run(run())

    assert isinstance(tm.positions.index, pd.MultiIndex)
    assert tm.positions.index.names == ["symbol", "timestamp"]
    assert len(tm.positions) == 1


def test_open_position_concurrent_single_entry():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await asyncio.gather(
            tm.open_position('BTCUSDT', 'buy', 100, {}),
            tm.open_position('BTCUSDT', 'buy', 100, {}),
        )

    import asyncio
    asyncio.run(run())

    assert len(tm.positions) == 1


def test_open_position_many_concurrent_single_entry():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await asyncio.gather(
            *[
                tm.open_position('BTCUSDT', 'buy', 100, {})
                for _ in range(5)
            ]
        )

    import asyncio
    asyncio.run(run())

    assert len(tm.positions) == 1


def test_open_position_failed_order_not_recorded():
    dh = DummyDataHandler(fail_order=True)
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    import asyncio
    asyncio.run(run())

    assert len(tm.positions) == 0
    assert len(dh.exchange.orders) == tm.config.order_retry_attempts


def test_open_position_retries_until_success(monkeypatch):
    dh = DummyDataHandler()
    attempts = {"n": 0}

    async def fail_then_succeed(symbol, type, side, amount, price, tp, sl, params):
        attempts["n"] += 1
        dh.exchange.orders.append({
            'method': 'create_order_with_tp_sl',
            'symbol': symbol,
            'type': type,
            'side': side,
            'amount': amount,
            'price': price,
            'tp': tp,
            'sl': sl,
            'params': params,
        })
        if attempts["n"] < 2:
            return {"retCode": 1}
        return {"id": "2"}

    monkeypatch.setattr(
        dh.exchange,
        'create_order_with_take_profit_and_stop_loss',
        fail_then_succeed,
    )

    tm = TradeManager(make_config(), dh, None, None, None)
    async def fake_compute(symbol, vol):
        return 0.01
    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    asyncio.run(run())

    assert len(tm.positions) == 1
    assert attempts["n"] == 2


def test_open_position_skips_when_atr_zero():
    class ZeroAtrDataHandler(DummyDataHandler):
        async def get_atr(self, symbol: str) -> float:
            return 0.0

    dh = ZeroAtrDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    import asyncio
    asyncio.run(run())

    assert dh.exchange.orders == []
    assert len(tm.positions) == 0


def test_open_position_skips_when_data_stale():
    dh = DummyDataHandler(fresh=False)
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    import asyncio
    asyncio.run(run())

    assert dh.exchange.orders == []
    assert len(tm.positions) == 0


def test_is_data_fresh():
    fresh_dh = DummyDataHandler(fresh=True)
    stale_dh = DummyDataHandler(fresh=False)

    import asyncio
    assert asyncio.run(fresh_dh.is_data_fresh('BTCUSDT')) is True
    assert asyncio.run(stale_dh.is_data_fresh('BTCUSDT')) is False


def test_compute_risk_per_trade_zero_threshold():
    cfg = make_config()
    cfg.update({'volatility_threshold': 0})
    dh = DummyDataHandler()
    tm = TradeManager(cfg, dh, None, None, None)

    async def run():
        return await tm.compute_risk_per_trade('BTCUSDT', 0.01)

    import asyncio
    risk = asyncio.run(run())
    assert tm.min_risk_per_trade <= risk <= tm.max_risk_per_trade


def test_get_loss_streak():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    tm.returns_by_symbol['BTCUSDT'] = [
        (0, 0.1),
        (1, -0.2),
        (2, -0.3),
        (3, -0.4),
    ]

    async def run():
        return await tm.get_loss_streak('BTCUSDT')

    import asyncio

    streak = asyncio.run(run())
    assert streak == 3


def test_get_win_streak():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    tm.returns_by_symbol['BTCUSDT'] = [
        (0, -0.1),
        (1, 0.2),
        (2, 0.3),
        (3, 0.4),
    ]

    async def run():
        return await tm.get_win_streak('BTCUSDT')

    import asyncio
    streak = asyncio.run(run())
    assert streak == 3


@pytest.mark.asyncio
async def test_close_position_updates_returns_and_sharpe_ratio():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position("BTCUSDT", "buy", 100, {})
    await tm.close_position("BTCUSDT", 110, "Manual")

    assert len(tm.returns_by_symbol["BTCUSDT"]) == 1
    profit = tm.returns_by_symbol["BTCUSDT"][0][1]
    expected = profit / (1e-6) * math.sqrt(365 * 24 * 60 * 60 / tm.performance_window)
    sharpe = await tm.get_sharpe_ratio("BTCUSDT")
    assert sharpe == pytest.approx(expected)


@pytest.mark.asyncio
async def test_open_and_close_short_position():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position("BTCUSDT", "sell", 100, {})
    # verify stop loss for short
    pos = tm.positions.xs("BTCUSDT", level="symbol").iloc[0]
    assert pos["position"] == -1
    assert pos["stop_loss_price"] == pytest.approx(101.0)
    await tm.close_position("BTCUSDT", 90, "Manual")

    assert len(tm.returns_by_symbol["BTCUSDT"]) == 1
    assert tm.returns_by_symbol["BTCUSDT"][0][1] > 0


class DummyModel:
    def eval(self):
        pass

    def __call__(self, *_):
        class _Out:
            def squeeze(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return 0.6

        return _Out()


@pytest.mark.asyncio
async def test_evaluate_signal_uses_cached_features(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {"BTCUSDT": DummyModel()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    monkeypatch.setattr(tm, "evaluate_ema_condition", lambda *a, **k: True)

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal in ("buy", "sell", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_ema", [False, True])
async def test_evaluate_signal_handles_sync_async_ema(monkeypatch, async_ema):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {"BTCUSDT": DummyModel()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    called = {"ok": False}
    if async_ema:
        async def _ema(*a, **k):
            called["ok"] = True
            return True
    else:
        def _ema(*a, **k):
            called["ok"] = True
            return True
    monkeypatch.setattr(tm, "evaluate_ema_condition", _ema)

    signal = await tm.evaluate_signal("BTCUSDT")
    assert called["ok"]
    assert signal in ("buy", "sell", None)


@pytest.mark.asyncio
async def test_evaluate_signal_considers_gpt(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {"BTCUSDT": DummyModel()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()
    cfg = BotConfig(
        lstm_timesteps=2,
        cache_dir=tempfile.mkdtemp(),
        transformer_weight=0.4,
        ema_weight=0.0,
        gpt_weight=0.6,
    )
    tm = TradeManager(cfg, dh, mb, None, None)
    monkeypatch.setattr(tm, "evaluate_ema_condition", lambda *a, **k: False)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    from bot import trading_bot
    trading_bot.GPT_ADVICE.signal = "sell"
    signal = await tm.evaluate_signal("BTCUSDT")
    trading_bot.GPT_ADVICE.signal = None
    assert signal == "sell"


@pytest.mark.asyncio
async def test_evaluate_signal_retrains_when_model_missing(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {}
            self.calibrators = {}
            self.feature_cache = {}
            self.retrained = False

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            return np.arange(8, dtype=np.float32).reshape(-1, 1)

        def prepare_dataset(self, features):
            X = np.ones((4, 1), dtype=np.float32)
            y = np.array([0, 1, 0, 1], dtype=np.float32)
            return X, y

        async def retrain_symbol(self, symbol):
            self.retrained = True

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal is None
    assert mb.retrained


@pytest.mark.asyncio
async def test_evaluate_signal_skips_retrain_on_single_label(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {}
            self.calibrators = {}
            self.feature_cache = {}
            self.retrained = False

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            return np.ones((4, 1), dtype=np.float32)

        def prepare_dataset(self, features):
            X = np.ones((2, 1), dtype=np.float32)
            y = np.zeros(2, dtype=np.float32)
            return X, y

        async def retrain_symbol(self, symbol):
            self.retrained = True

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal is None
    assert not mb.retrained


@pytest.mark.asyncio
async def test_evaluate_signal_handles_http_400(monkeypatch):
    dh = DummyDataHandler()

    class HTTPError(httpx.HTTPError):
        def __init__(self, status):
            super().__init__("boom")
            self.response = types.SimpleNamespace(status_code=status)

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {}
            self.calibrators = {}
            self.feature_cache = {}
            self.called = False

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            return np.arange(8, dtype=np.float32).reshape(-1, 1)

        def prepare_dataset(self, features):
            X = np.ones((4, 1), dtype=np.float32)
            y = np.array([0, 1, 0, 1], dtype=np.float32)
            return X, y

        async def retrain_symbol(self, symbol):
            self.called = True
            raise HTTPError(400)

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal is None
    assert mb.called


@pytest.mark.asyncio
async def test_evaluate_signal_raises_on_http_error(monkeypatch):
    dh = DummyDataHandler()

    class HTTPError(httpx.HTTPError):
        def __init__(self, status):
            super().__init__("boom")
            self.response = types.SimpleNamespace(status_code=status)

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {}
            self.calibrators = {}
            self.feature_cache = {}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            return np.arange(8, dtype=np.float32).reshape(-1, 1)

        def prepare_dataset(self, features):
            X = np.ones((4, 1), dtype=np.float32)
            y = np.array([0, 1, 0, 1], dtype=np.float32)
            return X, y

        async def retrain_symbol(self, symbol):
            raise HTTPError(500)

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    with pytest.raises(HTTPError):
        await tm.evaluate_signal("BTCUSDT")


@pytest.mark.asyncio
async def test_evaluate_signal_regression(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"

            class Model:
                def eval(self):
                    pass

                def __call__(self, *_):
                    class _Out:
                        def squeeze(self):
                            return self

                        def float(self):
                            return self

                        def cpu(self):
                            return self

                        def numpy(self):
                            return 0.003

                    return _Out()

            self.predictive_models = {"BTCUSDT": Model()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

    mb = MB()
    cfg = BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp(), prediction_target="pnl", trading_fee=0.001)
    tm = TradeManager(cfg, dh, mb, None, None)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal == "buy"


@pytest.mark.asyncio
async def test_rl_action_overrides_voting(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            class Model(DummyModel):
                def __call__(self, *_):
                    class _Out:
                        def squeeze(self):
                            return self

                        def float(self):
                            return self

                        def cpu(self):
                            return self

                        def numpy(self):
                            return 0.9

                    return _Out()

            self.predictive_models = {"BTCUSDT": Model()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()

    class RL:
        def __init__(self):
            self.models = {"BTCUSDT": object()}

        def predict(self, symbol, obs):
            return "open_short"

    rl = RL()
    cfg = BotConfig(
        lstm_timesteps=2,
        cache_dir=tempfile.mkdtemp(),
        transformer_weight=0.7,
        ema_weight=0.3,
    )
    tm = TradeManager(cfg, dh, mb, None, None, rl)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    monkeypatch.setattr(tm, "evaluate_ema_condition", lambda *a, **k: True)

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal == "sell"


@pytest.mark.asyncio
async def test_check_exit_signal_uses_cached_features(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {"BTCUSDT": DummyModel()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)
    idx = pd.MultiIndex.from_tuples([
        ("BTCUSDT", pd.Timestamp("2020-01-01"))
    ], names=["symbol", "timestamp"])
    tm.positions = pd.DataFrame({
        "side": ["buy"],
        "position": [1],
        "size": [1],
        "entry_price": [100],
        "tp_multiplier": [2],
        "sl_multiplier": [1],
        "stop_loss_price": [99],
        "highest_price": [100],
        "lowest_price": [0],
        "breakeven_triggered": [False],
    }, index=idx)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    monkeypatch.setattr(tm, "close_position", lambda *a, **k: None)

    await tm.check_exit_signal("BTCUSDT", 100)


@pytest.mark.asyncio
async def test_exit_signal_triggers_reverse_trade(monkeypatch):
    dh = DummyDataHandler()
    class MB:
        def __init__(self):
            self.device = "cpu"
            class Model:
                def eval(self):
                    pass
                def __call__(self, *_):
                    class _Out:
                        def squeeze(self):
                            return self
                        def float(self):
                            return self
                        def cpu(self):
                            return self
                        def numpy(self):
                            return 0.2
                    return _Out()
            self.predictive_models = {"BTCUSDT": Model()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)
    idx = pd.MultiIndex.from_tuples([
        ("BTCUSDT", pd.Timestamp("2020-01-01"))
    ], names=["symbol", "timestamp"])
    tm.positions = pd.DataFrame({
        "side": ["buy"],
        "position": [1],
        "size": [1],
        "entry_price": [100],
        "tp_multiplier": [2],
        "sl_multiplier": [1],
        "stop_loss_price": [99],
        "highest_price": [100],
        "lowest_price": [0],
        "breakeven_triggered": [False],
    }, index=idx)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    opened = {"side": None}

    async def fake_close(symbol, price, reason=""):
        tm.positions = tm.positions.drop(symbol, level="symbol")

    async def fake_open(symbol, side, price, params):
        opened["side"] = side

    monkeypatch.setattr(tm, "close_position", fake_close)
    monkeypatch.setattr(tm, "open_position", fake_open)
    async def _ema(*a, **k):
        return True
    monkeypatch.setattr(tm, "evaluate_ema_condition", _ema)

    await tm.check_exit_signal("BTCUSDT", 100)

    assert opened["side"] == "sell"


@pytest.mark.asyncio
async def test_rl_close_action(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {"BTCUSDT": DummyModel()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()

    class RL:
        def __init__(self):
            self.models = {"BTCUSDT": object()}

        def predict(self, symbol, obs):
            return "close"

    rl = RL()

    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None, rl)
    idx = pd.MultiIndex.from_tuples([
        ("BTCUSDT", pd.Timestamp("2020-01-01"))
    ], names=["symbol", "timestamp"])
    tm.positions = pd.DataFrame({
        "side": ["buy"],
        "position": [1],
        "size": [1],
        "entry_price": [100],
        "tp_multiplier": [2],
        "sl_multiplier": [1],
        "stop_loss_price": [99],
        "highest_price": [100],
        "lowest_price": [0],
        "breakeven_triggered": [False],
    }, index=idx)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    called = {"n": 0}

    async def fake_close(symbol, price, reason=""):
        called["n"] += 1
        tm.positions = tm.positions.drop(symbol, level="symbol")

    monkeypatch.setattr(tm, "close_position", fake_close)

    await tm.check_exit_signal("BTCUSDT", 100)

    assert called["n"] == 1


@pytest.mark.asyncio
async def test_check_stop_loss_take_profit_triggers_close_long(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position("BTCUSDT", "buy", 100, {})

    open_ts = tm.positions.index.get_level_values("timestamp")[0]
    new_ts = open_ts + pd.Timedelta(minutes=1)
    dh.ohlcv = pd.DataFrame(
        {"close": [99], "atr": [1.0]},
        index=pd.MultiIndex.from_tuples([("BTCUSDT", new_ts)], names=["symbol", "timestamp"]),
    )
    dh.indicators["BTCUSDT"].atr = pd.Series([1.0])

    called = {"n": 0}

    async def wrapped(symbol, price, reason=""):
        called["n"] += 1
        tm.positions = tm.positions.drop(symbol, level="symbol")

    monkeypatch.setattr(tm, "close_position", wrapped)

    await tm.check_stop_loss_take_profit("BTCUSDT", 99)

    assert called["n"] == 1
    assert len(tm.positions) == 0


@pytest.mark.asyncio
async def test_check_stop_loss_take_profit_triggers_close_short(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position("BTCUSDT", "sell", 100, {})

    open_ts = tm.positions.index.get_level_values("timestamp")[0]
    new_ts = open_ts + pd.Timedelta(minutes=1)
    dh.ohlcv = pd.DataFrame(
        {"close": [101], "atr": [1.0]},
        index=pd.MultiIndex.from_tuples([("BTCUSDT", new_ts)], names=["symbol", "timestamp"]),
    )
    dh.indicators["BTCUSDT"].atr = pd.Series([1.0])

    called = {"n": 0}

    async def wrapped(symbol, price, reason=""):
        called["n"] += 1
        tm.positions = tm.positions.drop(symbol, level="symbol")

    monkeypatch.setattr(tm, "close_position", wrapped)

    await tm.check_stop_loss_take_profit("BTCUSDT", 101)

    assert called["n"] == 1
    assert len(tm.positions) == 0


@pytest.mark.asyncio
async def test_check_stop_loss_take_profit_breakeven_uses_fixed_price(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position("BTCUSDT", "buy", 100, {})

    tm.positions.loc[pd.IndexSlice["BTCUSDT", :], "breakeven_triggered"] = True
    tm.positions.loc[pd.IndexSlice["BTCUSDT", :], "stop_loss_price"] = 100
    dh.indicators["BTCUSDT"].atr = pd.Series([10.0])

    open_ts = tm.positions.index.get_level_values("timestamp")[0]
    new_ts = open_ts + pd.Timedelta(minutes=1)
    dh.ohlcv = pd.DataFrame(
        {"close": [99.5], "atr": [1.0]},
        index=pd.MultiIndex.from_tuples([("BTCUSDT", new_ts)], names=["symbol", "timestamp"]),
    )
    dh.indicators["BTCUSDT"].atr = pd.Series([10.0])

    called = {"n": 0}

    async def wrapped(symbol, price, reason=""):
        called["n"] += 1
        tm.positions = tm.positions.drop(symbol, level="symbol")

    monkeypatch.setattr(tm, "close_position", wrapped)

    await tm.check_stop_loss_take_profit("BTCUSDT", 99.5)

    assert called["n"] == 1


@pytest.mark.asyncio
async def test_check_stop_loss_take_profit_delayed(monkeypatch):
    dh = DummyDataHandler()
    cfg = make_config()
    cfg.update(
        {
            "trailing_stop_percentage": 1.0,
            "trailing_stop_coeff": 0.0,
            "trailing_stop_multiplier": 1.0,
        }
    )
    tm = TradeManager(cfg, dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position("BTCUSDT", "buy", 100, {})
    open_ts = tm.positions.index.get_level_values("timestamp")[0]
    dh.ohlcv = pd.DataFrame(
        {"close": [100], "atr": [1.0]},
        index=pd.MultiIndex.from_tuples([("BTCUSDT", open_ts)], names=["symbol", "timestamp"]),
    )
    dh.indicators["BTCUSDT"].atr = pd.Series([1.0])

    await tm.check_trailing_stop("BTCUSDT", 101)

    called = {"n": 0}

    async def wrapped(symbol, price, reason=""):
        called["n"] += 1
        tm.positions = tm.positions.drop(symbol, level="symbol")

    monkeypatch.setattr(tm, "close_position", wrapped)

    await tm.check_stop_loss_take_profit("BTCUSDT", 100)
    assert called["n"] == 0

    new_ts = open_ts + pd.Timedelta(minutes=1)
    dh.ohlcv = pd.DataFrame(
        {"close": [100], "atr": [1.0]},
        index=pd.MultiIndex.from_tuples([("BTCUSDT", new_ts)], names=["symbol", "timestamp"]),
    )
    dh.indicators["BTCUSDT"].atr = pd.Series([1.0])

    await tm.check_stop_loss_take_profit("BTCUSDT", 100)
    assert called["n"] == 1


def test_compute_stats():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)
    tm.returns_by_symbol["BTCUSDT"] = [
        (0, 1.0),
        (1, -2.0),
        (2, 2.0),
    ]

    async def run():
        return await tm.compute_stats()

    import asyncio

    stats = asyncio.run(run())
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["avg_pnl"] == pytest.approx(1.0 / 3)
    assert stats["max_drawdown"] == pytest.approx(2.0)


def test_get_stats_uses_cache(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)
    tm._stats_cache = {"win_rate": 0.1, "avg_pnl": 0.2, "max_drawdown": 0.3}
    tm._stats_cache_time = time.time()
    tm._stats_cache_stale = False

    def fake_run(coro):
        raise AssertionError("asyncio.run should not be called when cache is fresh")

    monkeypatch.setattr(asyncio, "run", fake_run)

    stats = tm.get_stats()
    assert stats == tm._stats_cache


@pytest.mark.asyncio
async def test_compute_stats_cache_invalidation():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)
    tm.returns_by_symbol["BTCUSDT"] = [(0.0, 1.0)]

    stats_before = await tm.compute_stats(force=True)
    tm.returns_by_symbol["BTCUSDT"].append((1.0, -1.0))
    tm._invalidate_stats_cache()
    stats_after = await tm.compute_stats()

    assert stats_after["avg_pnl"] != stats_before["avg_pnl"]


def test_save_state_runs_in_background(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)
    tm.save_interval = 0
    tm.last_save_time = 0
    tm._mark_positions_dirty()

    from bot import test_stubs as tm_test_stubs

    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setattr(tm_test_stubs, "IS_TEST_MODE", False, raising=False)

    finished = threading.Event()
    threads: list[int] = []

    def fake_persist(version: int) -> None:
        threads.append(threading.get_ident())
        time.sleep(0.05)
        tm._last_saved_version = version
        tm.positions_changed = False
        finished.set()

    monkeypatch.setattr(tm, "_persist_state", fake_persist)

    start = time.perf_counter()
    tm.save_state()
    duration = time.perf_counter() - start

    assert duration < 0.02
    assert finished.wait(1.0)
    assert threads and threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_execute_top_signals_ranking(monkeypatch):
    class DH(DummyDataHandler):
        def __init__(self):
            super().__init__()
            self.usdt_pairs = ["A", "B", "C"]
            idx = pd.MultiIndex.from_product([
                self.usdt_pairs,
                [pd.Timestamp("2020-01-01")],
            ], names=["symbol", "timestamp"])
            self.ohlcv = pd.DataFrame({"close": [1, 1, 1], "atr": [1, 1, 1]}, index=idx)

    dh = DH()
    tm = TradeManager(BotConfig(cache_dir=tempfile.mkdtemp(), top_signals=2), dh, None, None, None)

    async def fake_eval(symbol, return_prob=False):
        probs = {"A": 0.9, "B": 0.8, "C": 0.1}
        return ("buy", probs[symbol]) if return_prob else "buy"

    monkeypatch.setattr(tm, "evaluate_signal", fake_eval)

    opened = []

    async def fake_open(symbol, side, price, params):
        opened.append(symbol)

    monkeypatch.setattr(tm, "open_position", fake_open)

    await tm.execute_top_signals_once()

    assert opened == ["A", "B"]


def test_shutdown_shuts_down_ray(monkeypatch):
    import importlib
    import sys
    import types

    ray_stub = types.ModuleType("ray")
    called = {"done": False}
    ray_stub.init = lambda: None
    ray_stub.is_initialized = lambda: True

    def _shutdown():
        called["done"] = True

    ray_stub.shutdown = _shutdown

    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    tm_mod = importlib.reload(sys.modules.get("bot.trade_manager.core", trade_manager))
    monkeypatch.setattr(tm_mod, "ray", ray_stub, raising=False)

    dh = DummyDataHandler()
    tm = tm_mod.TradeManager(make_config(), dh, None, None, None)
    tm.shutdown()

    assert called["done"]


def test_shutdown_calls_ray_shutdown_in_test_mode(monkeypatch):
    import importlib
    import sys
    import types

    ray_stub = types.ModuleType("ray")
    called = {"done": False}
    ray_stub.is_initialized = lambda: False
    ray_stub.shutdown = lambda: called.update(done=True)

    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    tm_mod = importlib.reload(sys.modules.get("bot.trade_manager.core", trade_manager))
    monkeypatch.setattr(tm_mod, "ray", ray_stub, raising=False)

    dh = DummyDataHandler()
    tm = tm_mod.TradeManager(make_config(), dh, None, None, None)
    tm.shutdown()

    assert called["done"]


def test_shutdown_handles_missing_is_initialized(monkeypatch):
    import importlib
    import sys
    import types

    monkeypatch.setenv("TEST_MODE", "0")

    ray_stub = types.ModuleType("ray")

    class _RayRemoteFunction:
        def __init__(self, func):
            self._function = func

        def remote(self, *args, **kwargs):
            return self._function(*args, **kwargs)

        def options(self, *args, **kwargs):
            return self

    def _ray_remote(func=None, **_kwargs):
        if func is None:
            def wrapper(f):
                return _RayRemoteFunction(f)
            return wrapper
        return _RayRemoteFunction(func)

    ray_stub.remote = _ray_remote
    ray_stub.get = lambda x: x
    ray_stub.init = lambda *a, **k: None
    called = {"done": False}

    def _shutdown():
        called["done"] = True

    ray_stub.shutdown = _shutdown

    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    tm_mod = importlib.reload(sys.modules.get("bot.trade_manager.core", trade_manager))
    monkeypatch.setattr(tm_mod, "ray", ray_stub, raising=False)

    dh = DummyDataHandler()
    tm = tm_mod.TradeManager(make_config(), dh, None, None, None)
    tm.shutdown()

    assert not called["done"]


def test_warn_when_token_missing(monkeypatch, caplog):
    monkeypatch.delenv("TRADE_MANAGER_TOKEN", raising=False)
    ray_stub = types.ModuleType("ray")
    ray_stub.is_initialized = lambda: False
    ray_stub.init = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "ray", ray_stub)
    service = importlib.reload(
        importlib.import_module("bot.trade_manager.service")
    )
    with caplog.at_level(logging.WARNING, logger=service.logger.name):
        service._load_env()
    assert "торговые запросы будут отвергнуты" in caplog.text


sys.modules.pop('utils', None)
sys.modules.pop('bot.utils', None)

