"""Lightweight stubs for optional heavy dependencies in tests.

When the environment variable ``TEST_MODE`` is set to ``"1"`` this module
injects simplified stand-ins for external libraries such as :mod:`ray` and
``httpx``.  Importing this module has no effect outside of test mode.
"""

from __future__ import annotations

import os
import sys
import types
import logging
from types import ModuleType
from typing import Any, Protocol, cast


class RayModule(Protocol):
    def remote(self, func: Any | None = None, **kwargs: Any) -> Any:
        ...

    def get(self, obj: Any) -> Any:
        ...

    def init(self, *args: Any, **kwargs: Any) -> None:
        ...

    def is_initialized(self) -> bool:
        ...

    def shutdown(self, *args: Any, **kwargs: Any) -> None:
        ...


class HTTPXResponse(Protocol):
    status_code: int

    def json(self) -> Any:
        ...

    @property
    def text(self) -> str:
        ...


class HTTPXModule(Protocol):
    HTTPError: type[Exception]
    Response: type[HTTPXResponse]

    def get(self, url: str, *args: Any, **kwargs: Any) -> HTTPXResponse:
        ...

    def post(self, url: str, *args: Any, **kwargs: Any) -> HTTPXResponse:
        ...

    class AsyncClient:  # pragma: no cover - minimal placeholder
        ...


class PyBitUTModule(Protocol):
    HTTP: type


class PyBitModule(Protocol):
    unified_trading: PyBitUTModule


class A2WSGIModule(Protocol):
    def WSGIMiddleware(self, app: Any) -> Any:
        ...


class UvicornWSGIModule(Protocol):
    def WSGIMiddleware(self, app: Any) -> Any:
        ...


class UvicornMiddlewareModule(Protocol):
    wsgi: UvicornWSGIModule


class UvicornModule(Protocol):
    middleware: UvicornMiddlewareModule


class FlaskWithASGI(Protocol):
    wsgi_app: Any

    @property
    def asgi_app(self) -> Any:  # pragma: no cover - simple property
        ...


IS_TEST_MODE = False


def apply() -> None:
    """Patch heavy dependencies with lightweight stubs in test mode."""
    global IS_TEST_MODE
    IS_TEST_MODE = os.getenv("TEST_MODE") == "1"
    if not IS_TEST_MODE:
        return

    # ------------------------------------------------------------------ Ray
    ray_mod = cast(RayModule, types.ModuleType("ray"))

    class _RayRemoteFunction:
        def __init__(self, func):
            self._function = func

        def remote(self, *args, **kwargs):
            return self._function(*args, **kwargs)

        def options(self, *args, **kwargs):  # pragma: no cover - simple chain
            return self

    def _ray_remote(func=None, **_kwargs):
        if func is None:
            def wrapper(f):
                return _RayRemoteFunction(f)

            return wrapper
        return _RayRemoteFunction(func)

    _ray_state = {"initialized": False}

    def _init(*_a, **_k):
        _ray_state["initialized"] = True

    def _is_initialized() -> bool:
        return _ray_state["initialized"]

    def _shutdown(*_a, **_k):
        _ray_state["initialized"] = False

    setattr(ray_mod, "remote", _ray_remote)
    setattr(ray_mod, "get", lambda x: x)
    setattr(ray_mod, "init", _init)
    setattr(ray_mod, "is_initialized", _is_initialized)
    setattr(ray_mod, "shutdown", _shutdown)
    sys.modules["ray"] = cast(ModuleType, ray_mod)

    # ----------------------------------------------------------------- HTTPX
    try:
        import httpx as _real_httpx  # noqa: F401
    except Exception:
        httpx_mod = cast(HTTPXModule, types.ModuleType("httpx"))

        class _HTTPXResponse:
            def __init__(self, status_code: int = 200, text: str = "", json_data: Any | None = None):
                self.status_code = status_code
                self._text = text
                self._json = json_data

            def json(self) -> Any:
                return self._json

            @property
            def text(self) -> str:
                return self._text

        def _return_response(*_a: Any, **_k: Any) -> _HTTPXResponse:
            return _HTTPXResponse()

        class _AsyncClient:
            """Lightweight stand in for :class:`httpx.AsyncClient`."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.trust_env = kwargs.get("trust_env", False)

            async def __aenter__(self) -> "_AsyncClient":  # pragma: no cover - simple
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - simple
                return None

            async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover - patched in tests
                raise NotImplementedError

            def close(self) -> None:  # pragma: no cover - simple no-op
                return None

            async def aclose(self) -> None:  # pragma: no cover - simple no-op

        class _HTTPXClient:  # pragma: no cover - minimal placeholder
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.trust_env = kwargs.get("trust_env", False)
                self.cookies = _CookieJar()

            def request(self, *args: Any, **kwargs: Any) -> _HTTPXResponse:
                return _return_response()

            get = post = request

            def close(self) -> None:  # pragma: no cover - simple no-op
                return None

        class _HTTPXBaseTransport:  # pragma: no cover - minimal placeholder
            ...

        class _TimeoutException(Exception):  # pragma: no cover - simple subclass
            ...

        class _ConnectError(Exception):  # pragma: no cover - simple subclass
            ...

        _client_mod = types.SimpleNamespace(
            UseClientDefault=object,  # pragma: no cover - minimal
            USE_CLIENT_DEFAULT=object(),
        )

        setattr(httpx_mod, "Response", _HTTPXResponse)
        setattr(httpx_mod, "get", _return_response)
        setattr(httpx_mod, "post", _return_response)
        setattr(httpx_mod, "AsyncClient", _AsyncClient)
        setattr(httpx_mod, "HTTPError", Exception)
        setattr(httpx_mod, "TimeoutException", _TimeoutException)
        setattr(httpx_mod, "ConnectError", _ConnectError)
        setattr(httpx_mod, "Client", _HTTPXClient)
        setattr(httpx_mod, "BaseTransport", _HTTPXBaseTransport)
        setattr(httpx_mod, "_client", _client_mod)
        sys.modules["httpx"] = cast(ModuleType, httpx_mod)

    # ---------------------------------------------------------------- websockets
    ws_mod = cast(ModuleType, types.ModuleType("websockets"))

    class _WSConnectionClosed(Exception):
        ...

    class _WebSocket:
        async def __aenter__(self) -> "_WebSocket":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - simple
            return None

        async def send(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            return None

        async def recv(self) -> str:  # pragma: no cover - simple default
            return ""

    async def _ws_connect(*_a: Any, **_k: Any) -> _WebSocket:
        return _WebSocket()

    setattr(ws_mod, "connect", _ws_connect)
    setattr(ws_mod, "exceptions", types.SimpleNamespace(ConnectionClosed=_WSConnectionClosed))
    sys.modules["websockets"] = cast(ModuleType, ws_mod)

    # ------------------------------------------------------------------- PyBit
    pybit_mod = cast(PyBitModule, types.ModuleType("pybit"))
    ut_mod = cast(PyBitUTModule, types.ModuleType("unified_trading"))
    setattr(ut_mod, "HTTP", object)
    setattr(pybit_mod, "unified_trading", ut_mod)
    sys.modules["pybit"] = cast(ModuleType, pybit_mod)
    sys.modules["pybit.unified_trading"] = cast(ModuleType, ut_mod)

    # ------------------------------------------------------------------ a2wsgi
    a2wsgi_mod = cast(A2WSGIModule, types.ModuleType("a2wsgi"))
    setattr(a2wsgi_mod, "WSGIMiddleware", lambda app: app)
    sys.modules["a2wsgi"] = cast(ModuleType, a2wsgi_mod)

    # ------------------------------------------------------------------ uvicorn
    uvicorn_mod = cast(UvicornModule, types.ModuleType("uvicorn"))
    setattr(
        uvicorn_mod,
        "middleware",
        types.SimpleNamespace(
            wsgi=types.SimpleNamespace(WSGIMiddleware=lambda app: app)
        ),
    )
    sys.modules["uvicorn"] = cast(ModuleType, uvicorn_mod)

    # ------------------------------------------------------------------- torch
    try:
        import torch  # type: ignore  # pragma: no cover - use real torch if available
    except Exception:
        # If torch is not installed, allow modules to handle its absence
        # individually. The model builder will fall back to lightweight stubs.
        pass

    # ------------------------------------------------------------------- Flask
    try:  # pragma: no cover - best effort
        from flask import Flask as _Flask

        Flask = cast(type[FlaskWithASGI], _Flask)
        if not hasattr(Flask, "asgi_app"):
            setattr(Flask, "asgi_app", property(lambda self: self.wsgi_app))
    except Exception as exc:  # pragma: no cover - best effort
        logging.debug("Failed to patch Flask for ASGI support: %s", exc)


apply()


__all__ = ["IS_TEST_MODE", "apply"]

