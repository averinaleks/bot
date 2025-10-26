import os
import sys
import types

import pytest

os.environ.setdefault("TEST_MODE", "1")


def _import_trading_bot() -> tuple[object, object]:
    """Import trading bot helpers while temporarily stubbing ``data_handler``."""

    original_data_handler = sys.modules.get("data_handler")
    stub_data_handler = types.ModuleType("data_handler")
    stub_data_handler.get_settings = lambda: None
    try:
        sys.modules["data_handler"] = stub_data_handler
        from bot.trading_bot import close_http_client, get_http_client  # noqa: E402
    finally:
        if original_data_handler is None:
            sys.modules.pop("data_handler", None)
        else:
            sys.modules["data_handler"] = original_data_handler
    return get_http_client, close_http_client


get_http_client, close_http_client = _import_trading_bot()


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
