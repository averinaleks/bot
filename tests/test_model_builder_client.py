from __future__ import annotations

import asyncio
import importlib
import builtins
import sys


def test_model_builder_client_uses_httpx_stub(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "httpx":
            raise ImportError("httpx not installed for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module_name = "model_builder_client"
    original_module = sys.modules.pop(module_name, None)

    try:
        module = importlib.import_module(module_name)
        assert getattr(module.httpx, "__offline_stub__", False) is True

        client = module.httpx.Client()
        response = client.post("https://example.test", json={"stub": True})
        assert response.json() == {"stub": True}
        assert response.status_code == 200

        async def _exercise_async_client() -> None:
            async with module.httpx.AsyncClient() as async_client:
                async_response = await async_client.get(
                    "https://example.test", json={"stub": "async"}
                )
                assert async_response.json() == {"stub": "async"}

        asyncio.run(_exercise_async_client())
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
