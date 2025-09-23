"""Lightweight stubs for optional heavy dependencies in tests.

When the environment variable ``TEST_MODE`` is set to ``"1"`` this module
injects simplified stand-ins for external libraries such as :mod:`ray` and
``httpx``.  Importing this module has no effect outside of test mode.
"""

from __future__ import annotations

import os
import sys
import http.client
import json as _json
import logging
import types
from importlib.machinery import ModuleSpec
from types import ModuleType, TracebackType
from typing import Any, Literal, Protocol, cast
from urllib import parse as _urllib_parse


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


def _create_module(name: str) -> ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None)  # type: ignore[attr-defined]
    return module


class CsrfProtectError(Exception):
    pass


class CsrfProtect:
    @classmethod
    def load_config(cls, func):
        return func

    def __init__(self, *args, **kwargs):
        pass

    def generate_csrf_token(self) -> str:
        return "test-token"

    def generate_csrf_tokens(self):
        token = self.generate_csrf_token()
        return token, token

    async def validate_csrf(self, request) -> None:  # pragma: no cover - simple stub
        return


def apply() -> None:
    """Patch heavy dependencies with lightweight stubs in test mode."""
    global IS_TEST_MODE
    IS_TEST_MODE = os.getenv("TEST_MODE") == "1"
    if not IS_TEST_MODE:
        return

    # ------------------------------------------------------------------ Ray
    ray_mod = cast(RayModule, _create_module("ray"))

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
        httpx_mod = cast(HTTPXModule, _create_module("httpx"))

        class _HTTPXResponse:
            def __init__(
                self,
                status_code: int = 200,
                text: str = "",
                json: Any | None = None,
                request: Any | None = None,
                content: bytes | None = None,
                headers: dict[str, str] | None = None,
            ) -> None:
                self.status_code = status_code
                self._text = text
                self._json = json
                self.request = request
                self.content = content or b""
                self.headers = headers or {}

            def json(self) -> Any:
                return self._json

            @property
            def text(self) -> str:
                return self._text

            async def aread(self) -> bytes:
                return self.content

            def raise_for_status(self) -> None:
                """Mimic :meth:`httpx.Response.raise_for_status`."""

                if 400 <= self.status_code:
                    raise Exception(f"HTTP error {self.status_code}")

        def _return_response(method: str, url: str, *, timeout: Any | None = None, **kwargs: Any) -> _HTTPXResponse:
            """Fallback network client using :mod:`http.client`."""

            parsed = _urllib_parse.urlsplit(url)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

            connection_cls = (
                http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            )
            host = parsed.hostname or "localhost"
            port = parsed.port

            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"

            headers = {"Accept": "application/json"}
            headers.update(kwargs.pop("headers", {}) or {})

            body = kwargs.pop("data", None)
            if "json" in kwargs:
                body = _json.dumps(kwargs.pop("json")).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")

            if body is not None and not isinstance(body, (bytes, bytearray)):
                body = str(body).encode("utf-8")

            connection = connection_cls(host, port, timeout=timeout)
            try:
                connection.request(method.upper(), path, body=body, headers=headers)
                response = connection.getresponse()
                content = response.read()
            finally:
                connection.close()

            text = content.decode("utf-8", errors="replace")
            try:
                parsed_json: Any | None = _json.loads(text)
            except ValueError:
                parsed_json = None

            return _HTTPXResponse(
                status_code=response.status,
                text=text,
                json=parsed_json,
                content=content,
                headers=dict(response.headers),
            )

        class _CookieJar(dict):  # pragma: no cover - minimal cookie jar
            """Simplified cookie jar supporting assignment via ``set``.

            The real :mod:`httpx` cookie jar exposes a ``set`` method which is
            used by tests to store CSRF tokens.  Implementing the method here
            allows the stub client to mimic that behaviour when the actual
            dependency is not installed.
            """

            def set(self, key: str, value: str, **_kwargs: Any) -> None:
                self[key] = value

        class _AsyncClient:
            """Lightweight stand in for :class:`httpx.AsyncClient`."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.trust_env = kwargs.get("trust_env", False)
                self.is_closed = False

            async def __aenter__(self) -> "_AsyncClient":  # pragma: no cover - simple
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - simple
                await self.aclose()
                return None

            async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover - patched in tests
                raise NotImplementedError

            async def get(self, url: str, **kwargs: Any) -> _HTTPXResponse:
                if self.is_closed:
                    raise RuntimeError("Client closed")
                return _return_response("GET", url, **kwargs)

            post = get

            async def aclose(self) -> None:  # pragma: no cover - simple no-op
                self.is_closed = True

            def close(self) -> None:  # pragma: no cover - simple no-op
                self.is_closed = True
        class _HTTPXClient:  # pragma: no cover - minimal placeholder
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.trust_env = kwargs.get("trust_env", False)
                self.cookies = _CookieJar()
                self._timeout = kwargs.get("timeout")

            def request(self, method: str, url: str, **kwargs: Any) -> _HTTPXResponse:
                if "timeout" not in kwargs:
                    kwargs["timeout"] = self._timeout
                return _return_response(method, url, **kwargs)

            def get(self, url: str, **kwargs: Any) -> _HTTPXResponse:
                return self.request("GET", url, **kwargs)

            def post(self, url: str, **kwargs: Any) -> _HTTPXResponse:
                return self.request("POST", url, **kwargs)

            def __enter__(self) -> "_HTTPXClient":  # pragma: no cover - simple helper
                return self

            def close(self) -> None:  # pragma: no cover - simple no-op
                return None

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> Literal[False]:
                self.close()
                return False

        class _HTTPXBaseTransport:  # pragma: no cover - minimal placeholder
            ...

        class _HTTPXRequest:  # pragma: no cover - minimal placeholder
            def __init__(self, method: str, url: str) -> None:
                self.method = method
                self.url = url

        class _TimeoutException(Exception):  # pragma: no cover - simple subclass
            ...

        class _ConnectError(Exception):  # pragma: no cover - simple subclass
            ...

        _client_mod = types.SimpleNamespace(
            UseClientDefault=object,  # pragma: no cover - minimal
            USE_CLIENT_DEFAULT=object(),
        )

        setattr(httpx_mod, "Response", _HTTPXResponse)
        setattr(httpx_mod, "get", lambda url, *a, **k: _return_response("GET", url, *a, **k))
        setattr(httpx_mod, "post", lambda url, *a, **k: _return_response("POST", url, *a, **k))
        setattr(httpx_mod, "AsyncClient", _AsyncClient)
        setattr(httpx_mod, "HTTPError", Exception)
        setattr(httpx_mod, "TimeoutException", _TimeoutException)
        setattr(httpx_mod, "ConnectError", _ConnectError)
        setattr(httpx_mod, "Request", _HTTPXRequest)
        setattr(httpx_mod, "Client", _HTTPXClient)
        setattr(httpx_mod, "BaseTransport", _HTTPXBaseTransport)
        setattr(httpx_mod, "_client", _client_mod)
        sys.modules["httpx"] = cast(ModuleType, httpx_mod)

    # ---------------------------------------------------------------- websockets
    ws_mod = cast(ModuleType, _create_module("websockets"))

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
    pybit_mod = cast(PyBitModule, _create_module("pybit"))
    ut_mod = cast(PyBitUTModule, _create_module("pybit.unified_trading"))
    setattr(ut_mod, "HTTP", object)
    setattr(pybit_mod, "unified_trading", ut_mod)
    sys.modules["pybit"] = cast(ModuleType, pybit_mod)
    sys.modules["pybit.unified_trading"] = cast(ModuleType, ut_mod)

    # ------------------------------------------------------------------ a2wsgi
    a2wsgi_mod = cast(A2WSGIModule, _create_module("a2wsgi"))
    setattr(a2wsgi_mod, "WSGIMiddleware", lambda app: app)
    sys.modules["a2wsgi"] = cast(ModuleType, a2wsgi_mod)

    # ------------------------------------------------------------------ uvicorn
    uvicorn_mod = cast(UvicornModule, _create_module("uvicorn"))
    setattr(
        uvicorn_mod,
        "middleware",
        types.SimpleNamespace(
            wsgi=types.SimpleNamespace(WSGIMiddleware=lambda app: app)
        ),
    )
    sys.modules["uvicorn"] = cast(ModuleType, uvicorn_mod)

    # ------------------------------------------------ fastapi_csrf_protect
    csrf_mod = cast(ModuleType, _create_module("fastapi_csrf_protect"))
    setattr(csrf_mod, "CsrfProtect", CsrfProtect)
    setattr(csrf_mod, "CsrfProtectError", CsrfProtectError)
    sys.modules["fastapi_csrf_protect"] = csrf_mod

    # ------------------------------------------------------------------- torch
    # Importing the real ``torch`` package is expensive and unnecessary during
    # tests.  Instead, provide a very small stub that exposes the attributes
    # commonly checked by the codebase.  This keeps test startup fast and
    # ensures modules can safely call ``torch.cuda.is_available()`` without
    # pulling in heavy dependencies.
    if "torch" not in sys.modules:
        torch = cast(ModuleType, _create_module("torch"))
        torch.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
        torch.nn = types.SimpleNamespace(Module=type("Module", (), {}))  # type: ignore[attr-defined]
        torch.distributed = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
        sys.modules["torch"] = torch

    # ------------------------------------------------------------- safetensors
    if "safetensors" not in sys.modules:
        safetensors = _create_module("safetensors")
        safetensors_torch = _create_module("safetensors.torch")

        def _raise_not_implemented(*_a: Any, **_k: Any) -> None:
            raise NotImplementedError("safetensors is not available in test mode")

        safetensors.deserialize = _raise_not_implemented  # type: ignore[attr-defined]
        safetensors.serialize = _raise_not_implemented  # type: ignore[attr-defined]
        safetensors.serialize_file = _raise_not_implemented  # type: ignore[attr-defined]
        safetensors.safe_open = _raise_not_implemented  # type: ignore[attr-defined]

        safetensors_torch.save_file = _raise_not_implemented  # type: ignore[attr-defined]
        safetensors_torch.load_file = _raise_not_implemented  # type: ignore[attr-defined]

        safetensors.torch = safetensors_torch  # type: ignore[attr-defined]
        sys.modules["safetensors"] = safetensors
        sys.modules["safetensors.torch"] = safetensors_torch

    # ------------------------------------------------------------------- Flask
    try:  # pragma: no cover - best effort
        from flask import Flask as _Flask

        Flask = cast(type[FlaskWithASGI], _Flask)
        if not hasattr(Flask, "asgi_app"):
            setattr(Flask, "asgi_app", property(lambda self: self.wsgi_app))
    except Exception as exc:  # pragma: no cover - best effort
        logging.debug("Failed to patch Flask for ASGI support: %s", exc)

apply()


__all__ = ["IS_TEST_MODE", "apply", "CsrfProtect", "CsrfProtectError"]

