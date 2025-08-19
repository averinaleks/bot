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
