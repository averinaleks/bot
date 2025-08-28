import os
import pytest

os.environ.setdefault("TEST_MODE", "1")

from bot import data_handler, trade_manager


@pytest.mark.asyncio
async def test_shared_http_client_cleanup():
    client1 = await data_handler.get_http_client()
    client2 = await trade_manager.get_http_client()
    assert client1 is client2
    await data_handler.close_http_client()
    client3 = await data_handler.get_http_client()
    assert client3 is not client1
    await data_handler.close_http_client()
