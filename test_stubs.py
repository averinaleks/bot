"""Lightweight stubs for optional heavy dependencies in tests.

When the environment variable ``TEST_MODE`` is set to ``"1"`` this module
injects simplified stand-ins for external libraries such as :mod:`ray` and
``httpx``.  Importing this module has no effect outside of test mode.
"""

from __future__ import annotations

import os
import sys
import types
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

    ray_mod.remote = _ray_remote
    ray_mod.get = lambda x: x
    ray_mod.init = _init
    ray_mod.is_initialized = _is_initialized
    ray_mod.shutdown = _shutdown
    sys.modules["ray"] = cast(ModuleType, ray_mod)

    # ----------------------------------------------------------------- HTTPX
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

    httpx_mod.Response = _HTTPXResponse
    httpx_mod.get = _return_response
    httpx_mod.post = _return_response
    httpx_mod.AsyncClient = object
    httpx_mod.HTTPError = Exception
    sys.modules["httpx"] = cast(ModuleType, httpx_mod)

    # ------------------------------------------------------------------- PyBit
    pybit_mod = cast(PyBitModule, types.ModuleType("pybit"))
    ut_mod = cast(PyBitUTModule, types.ModuleType("unified_trading"))
    ut_mod.HTTP = object
    pybit_mod.unified_trading = ut_mod
    sys.modules["pybit"] = cast(ModuleType, pybit_mod)
    sys.modules["pybit.unified_trading"] = cast(ModuleType, ut_mod)

    # ------------------------------------------------------------------ a2wsgi
    a2wsgi_mod = cast(A2WSGIModule, types.ModuleType("a2wsgi"))
    a2wsgi_mod.WSGIMiddleware = lambda app: app
    sys.modules["a2wsgi"] = cast(ModuleType, a2wsgi_mod)

    # ------------------------------------------------------------------ uvicorn
    uvicorn_mod = cast(UvicornModule, types.ModuleType("uvicorn"))
    uvicorn_mod.middleware = types.SimpleNamespace(
        wsgi=types.SimpleNamespace(WSGIMiddleware=lambda app: app)
    )
    sys.modules["uvicorn"] = cast(ModuleType, uvicorn_mod)

    # ------------------------------------------------------------------- Flask
    try:  # pragma: no cover - best effort
        from flask import Flask as _Flask

        Flask = cast(type[FlaskWithASGI], _Flask)
        if not hasattr(Flask, "asgi_app"):
            Flask.asgi_app = property(lambda self: self.wsgi_app)
    except Exception:
        pass


apply()


__all__ = ["IS_TEST_MODE", "apply"]

