import importlib.machinery
import os
import sys
import types
from pathlib import Path
import pytest

os.environ.setdefault("TEST_MODE", "1")

# Stub out data_handler to avoid circular import during trading_bot import.
stub_data_handler = types.ModuleType("data_handler")
stub_data_handler.get_settings = lambda: None
_package_path = Path(__file__).resolve().parent.parent / "data_handler"
stub_data_handler.__path__ = [str(_package_path)]
stub_spec = importlib.machinery.ModuleSpec("data_handler", loader=None, is_package=True)
stub_spec.submodule_search_locations = [str(_package_path)]
stub_spec.origin = str(_package_path / "__init__.py")
stub_data_handler.__spec__ = stub_spec
sys.modules.setdefault("data_handler", stub_data_handler)
from bot.trading_bot import get_http_client, close_http_client  # noqa: E402


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
