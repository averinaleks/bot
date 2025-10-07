import asyncio
import builtins
import importlib
import logging
import sys

import pytest

from bot import utils


class DummyExchange:
    def __init__(self):
        self.calls = 0
        self.last_http_status = 200

    async def fail(self):
        self.calls += 1
        return {'retCode': 1}


@pytest.mark.asyncio
async def test_safe_api_call_retries(monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)
    exch = DummyExchange()

    sleep_calls = {'n': 0}
    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        sleep_calls['n'] += 1
        await orig_sleep(0)

    monkeypatch.setattr(utils.asyncio, 'sleep', fast_sleep)

    with pytest.raises(RuntimeError):
        await utils.safe_api_call(exch, 'fail')

    assert exch.calls == 5
    assert sleep_calls['n'] == 4


@pytest.mark.asyncio
async def test_safe_api_call_test_mode():
    exch = DummyExchange()

    result = await utils.safe_api_call(exch, 'fail', test_mode=True)

    assert result == {'retCode': 1}
    assert exch.calls == 1


@pytest.mark.asyncio
async def test_safe_api_call_env_var_false(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "0")
    exch = DummyExchange()

    with pytest.raises(RuntimeError):
        await utils.safe_api_call(exch, 'fail')

    assert exch.calls == 5


@pytest.mark.asyncio
async def test_safe_api_call_unhandled(monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)

    class DummyExchange:
        last_http_status = 200

        async def boom(self):
            raise ValueError("boom")

    exch = DummyExchange()

    with pytest.raises(ValueError):
        await utils.safe_api_call(exch, 'boom')


def test_logging_not_duplicated_on_reimport(monkeypatch, tmp_path, capsys):
    import sys
    import importlib
    import logging

    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    # Reset logger to simulate first import
    logger = logging.getLogger("TradingBot")
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    monkeypatch.delitem(sys.modules, "bot.utils", raising=False)
    capsys.readouterr()  # clear captured output

    utils_mod = importlib.import_module("bot.utils")
    captured = capsys.readouterr()
    assert "Logging configured" not in captured.err

    utils_mod.configure_logging()
    captured = capsys.readouterr()
    assert captured.err.count("Logging configured") == 1

    utils_mod.logger.info("first")
    captured = capsys.readouterr()
    assert captured.err.count("first") == 1

    utils_mod = importlib.reload(utils_mod)
    captured = capsys.readouterr()
    assert "Logging configured" not in captured.err

    utils_mod.logger.info("second")
    captured = capsys.readouterr()
    assert captured.err.count("second") == 1


def test_configure_logging_level_update(monkeypatch, tmp_path):
    import logging

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    logger = logging.getLogger("TradingBot")
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    utils.configure_logging()
    assert logger.level == logging.WARNING
    handler_count = len(logger.handlers)

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    utils.configure_logging()
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == handler_count

    for h in logger.handlers[:]:
        logger.removeHandler(h)


def test_configure_logging_invalid_level(monkeypatch, tmp_path, caplog):
    import logging

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_LEVEL", "NOPE")

    logger = logging.getLogger("TradingBot")
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    with caplog.at_level(logging.WARNING):
        utils.configure_logging()

    assert "LOG_LEVEL 'NOPE' недопустим, используется INFO" in caplog.text
    assert logger.level == logging.INFO

    for h in logger.handlers[:]:
        logger.removeHandler(h)


def test_configure_logging_updates_level(monkeypatch, tmp_path):
    import logging

    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    logger = logging.getLogger("TradingBot")
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    utils.configure_logging()
    assert logger.level == logging.WARNING
    handler_count = len(logger.handlers)

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    utils.configure_logging()
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == handler_count

    for h in logger.handlers[:]:
        logger.removeHandler(h)

def test_jit_stub_preserves_metadata(monkeypatch):
    if not hasattr(utils, "_numba_missing"):
        pytest.skip("numba is installed")

    @utils.jit(nopython=True)
    def sample_func(x):
        """example docstring"""
        return x

    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "example docstring"

    with pytest.raises(ImportError):
        sample_func(1)


def test_validate_host_default(monkeypatch, caplog):
    monkeypatch.delenv('HOST', raising=False)
    with caplog.at_level('INFO'):
        host = utils.validate_host()
    assert host == '127.0.0.1'
    assert 'HOST не установлен' in caplog.text

def test_validate_host_accepts_loopback(monkeypatch):
    monkeypatch.setenv('HOST', '127.0.0.1')
    assert utils.validate_host() == '127.0.0.1'


def test_filter_outliers_zscore_fallback_without_scipy(monkeypatch):
    import pandas as pd

    utils._reset_zscore_cache_for_tests()

    def missing_scipy():
        raise ImportError("scipy unavailable")

    monkeypatch.setattr(utils, "_import_scipy_stats", lambda: missing_scipy())

    df = pd.DataFrame({"close": [1.0, 2.0, 100.0, 2.0, 1.0]})
    result = utils.filter_outliers_zscore(df, "close", threshold=1.5)

    assert result["close"].isna().sum() == 1
    assert result["close"].isna().idxmax() == 2


def test_filter_outliers_zscore_handles_zero_variance(monkeypatch):
    import pandas as pd

    utils._reset_zscore_cache_for_tests()
    monkeypatch.setattr(utils, "_import_scipy_stats", lambda: (_ for _ in ()).throw(ImportError("missing")))

    df = pd.DataFrame({"close": [5.0, 5.0, 5.0, 5.0]})
    result = utils.filter_outliers_zscore(df, "close", threshold=1.0)

    assert result["close"].tolist() == [5.0, 5.0, 5.0, 5.0]


def test_validate_host_accepts_localhost(monkeypatch, caplog):
    monkeypatch.setenv('HOST', 'localhost')
    with caplog.at_level('INFO'):
        host = utils.validate_host()
    assert host == '127.0.0.1'
    assert 'localhost' in caplog.text


def test_validate_host_strips_and_normalizes(monkeypatch, caplog):
    monkeypatch.setenv('HOST', '  LOCALHOST  ')
    with caplog.at_level('INFO'):
        host = utils.validate_host()
    assert host == '127.0.0.1'
    assert 'localhost' in caplog.text


def test_validate_host_empty_string(monkeypatch, caplog):
    monkeypatch.setenv('HOST', '')
    with caplog.at_level('INFO'):
        host = utils.validate_host()
    assert host == '127.0.0.1'
    assert 'HOST не установлен' in caplog.text


@pytest.mark.parametrize('host', ['192.0.2.1', '256.0.0.1', 'example.com'])
def test_validate_host_rejects_invalid(host, monkeypatch):
    monkeypatch.setenv('HOST', host)
    with pytest.raises(ValueError):
        utils.validate_host()


def test_validate_host_ignores_port(monkeypatch):
    monkeypatch.setenv('HOST', '127.0.0.1:8080')
    assert utils.validate_host() == '127.0.0.1'


def test_validate_host_ipv6_port(monkeypatch):
    monkeypatch.setenv('HOST', '[::1]:9000')
    assert utils.validate_host() == '::1'


def test_validate_host_strips_whitespace_around_host_and_port(monkeypatch):
    monkeypatch.setenv('HOST', ' 127.0.0.1 : 8080 ')
    assert utils.validate_host() == '127.0.0.1'

    monkeypatch.setenv('HOST', '[ ::1 ] : 9000')
    assert utils.validate_host() == '::1'


def test_validate_host_invalid_port(monkeypatch):
    monkeypatch.setenv('HOST', '127.0.0.1:notaport')
    with pytest.raises(ValueError):
        utils.validate_host()


def test_validate_host_port_out_of_range(monkeypatch):
    monkeypatch.setenv('HOST', '127.0.0.1:70000')
    with pytest.raises(ValueError):
        utils.validate_host()


@pytest.mark.parametrize('host', ['127.0.0.1:', '[::1]:'])
def test_validate_host_missing_port_value(monkeypatch, host):
    monkeypatch.setenv('HOST', host)
    with pytest.raises(ValueError):
        utils.validate_host()
def _reload_utils_with_blocked(monkeypatch, blocked_names):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked_names:
            raise ImportError(f"blocked import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as context:
        for mod_name in blocked_names:
            context.delitem(sys.modules, mod_name, raising=False)
        context.setattr(builtins, "__import__", fake_import)
        module = sys.modules.get("bot.utils") or sys.modules.get("utils") or utils
        return importlib.reload(module)


def test_numba_warning_emitted_once(monkeypatch, caplog):
    utils._NUMBA_IMPORT_WARNED = False
    caplog.set_level(logging.WARNING, logger="TradingBot")

    _reload_utils_with_blocked(monkeypatch, {"numba"})
    numba_messages = [
        record.message for record in caplog.records if "Numba" in record.message
    ]
    assert any("Numba import failed" in message for message in numba_messages)

    caplog.clear()

    _reload_utils_with_blocked(monkeypatch, {"numba"})
    assert not any("Numba import failed" in record.message for record in caplog.records)

    importlib.reload(sys.modules["bot.utils"])  # restore actual imports


def test_telegram_logger_warnings_emitted_once(monkeypatch, caplog):
    utils._TELEGRAMLOGGER_IMPORT_WARNED = False
    utils._TELEGRAMLOGGER_STUB_INIT_WARNED = False
    caplog.set_level(logging.WARNING, logger="TradingBot")

    blocked = {"bot.telegram_logger", "telegram_logger"}
    _reload_utils_with_blocked(monkeypatch, blocked)

    assert sum(
        1
        for record in caplog.records
        if "Failed to import TelegramLogger via package" in record.message
    ) == 1

    caplog.clear()

    _reload_utils_with_blocked(monkeypatch, blocked)
    assert not any(
        "Failed to import TelegramLogger via package" in record.message
        for record in caplog.records
    )

    caplog.clear()

    utils.TelegramLogger()
    assert "TelegramLogger is unavailable; notifications disabled" in caplog.text

    caplog.clear()

    utils.TelegramLogger()
    assert "TelegramLogger is unavailable; notifications disabled" not in caplog.text

    importlib.import_module("bot.telegram_logger")
    importlib.reload(sys.modules["bot.telegram_logger"])
    importlib.reload(sys.modules["bot.utils"])  # restore actual imports

