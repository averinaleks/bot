import pytest

from bot.utils import retry


def test_retry_success_sync():
    calls = {"count": 0, "delays": []}

    def delay(base):
        calls["delays"].append(base)
        return 0

    @retry(3, delay)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("boom")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3
    assert calls["delays"] == [1, 2]


def test_retry_exhaust_sync():
    calls = {"count": 0}

    @retry(3, lambda _: 0)
    def always_fail():
        calls["count"] += 1
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        always_fail()
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_retry_success_async():
    calls = {"count": 0, "delays": []}

    def delay(base):
        calls["delays"].append(base)
        return 0

    @retry(3, delay)
    async def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("boom")
        return "ok"

    assert await flaky() == "ok"
    assert calls["count"] == 3
    assert calls["delays"] == [1, 2]


@pytest.mark.asyncio
async def test_retry_exhaust_async():
    calls = {"count": 0}

    @retry(3, lambda _: 0)
    async def always_fail():
        calls["count"] += 1
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        await always_fail()
    assert calls["count"] == 3
