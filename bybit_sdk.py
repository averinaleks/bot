import os
import asyncio
from typing import Any, Callable
from pybit.unified_trading import HTTP

class AsyncBybit:
    """Simple async wrapper around pybit's HTTP client."""

    def __init__(self, testnet: bool = False):
        self.client = HTTP(
            testnet=testnet,
            api_key=os.environ.get("BYBIT_API_KEY"),
            api_secret=os.environ.get("BYBIT_API_SECRET"),
        )

    async def _call(self, func: Callable[..., Any], *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    async def __getattr__(self, name: str):  # type: ignore[override]
        func = getattr(self.client, name)
        if callable(func):
            async def wrapper(*args, **kwargs):
                return await self._call(func, *args, **kwargs)
            return wrapper
        raise AttributeError(name)
