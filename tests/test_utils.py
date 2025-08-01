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
