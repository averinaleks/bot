import os
import sys
import types
import pytest

os.environ.setdefault("TEST_MODE", "1")

# Stub out data_handler to avoid circular import during trading_bot import.
stub_data_handler = types.ModuleType("data_handler")
stub_data_handler.get_settings = lambda: None
sys.modules.setdefault("data_handler", stub_data_handler)

from bot.trading_bot import get_http_client, close_http_client


@pytest.mark.asyncio
async def test_http_client_cleanup():
    client1 = await get_http_client()
    await close_http_client()
    assert client1.is_closed
    with pytest.raises(RuntimeError):
        await client1.get("http://example.com")

    client2 = await get_http_client()
    assert client2 is not client1

    await close_http_client()
