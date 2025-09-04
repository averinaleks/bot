import asyncio
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
async def test_safe_api_call_test_mode(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    exch = DummyExchange()

    result = await utils.safe_api_call(exch, 'fail')

    assert result == {'retCode': 1}
    assert exch.calls == 1


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
    assert "Logging initialized" not in captured.err

    utils_mod.configure_logging()
    captured = capsys.readouterr()
    assert captured.err.count("Logging initialized") == 1

    utils_mod.logger.info("first")
    captured = capsys.readouterr()
    assert captured.err.count("first") == 1

    utils_mod = importlib.reload(utils_mod)
    captured = capsys.readouterr()
    assert "Logging initialized" not in captured.err

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
    handlers = list(logger.handlers)

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    utils.configure_logging()
    assert logger.level == logging.DEBUG
    assert logger.handlers == handlers

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


def test_validate_host_accepts_localhost(monkeypatch, caplog):
    monkeypatch.setenv('HOST', 'localhost')
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


@pytest.mark.parametrize('host', ['0.0.0.0', '256.0.0.1', 'example.com'])  # nosec B104
def test_validate_host_rejects_invalid(host, monkeypatch):
    monkeypatch.setenv('HOST', host)
    with pytest.raises(ValueError):
        utils.validate_host()
