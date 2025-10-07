"""Lightweight stubs for optional third-party dependencies."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any, Dict


def is_offline_env() -> bool:
    """Return ``True`` when the environment requests offline mode."""

    import os

    return (os.getenv("OFFLINE_MODE", "0").strip().lower() in {"1", "true", "yes", "on"})


def create_httpx_stub() -> SimpleNamespace:
    """Return a minimal stub emulating the :mod:`httpx` API."""

    class HTTPError(Exception):
        """Base HTTPX error placeholder."""

    class TimeoutException(HTTPError):
        """Timeout error placeholder."""

    class ConnectError(HTTPError):
        """Connection error placeholder."""

    class Headers(dict):
        """Simple headers mapping."""

    class Request:
        """Simplified HTTP request representation."""

        def __init__(self, method: str, url: str):
            self.method = method
            self.url = url

    class Response:
        """Simplified HTTP response object."""

        def __init__(
            self,
            status_code: int = 200,
            *,
            headers: Dict[str, str] | None = None,
            content: bytes | str | None = None,
            request: Request | None = None,
            json_data: Any | None = None,
        ) -> None:
            self.status_code = status_code
            self.headers: Headers = Headers(headers or {})
            self.request = request or Request("GET", "offline://stub")
            self._closed = False
            if content is None:
                body = b""
            elif isinstance(content, bytes):
                body = content
            else:
                body = str(content).encode("utf-8", errors="ignore")
            self._content = body
            self.content = body
            self._json_data = json_data

        async def aread(self) -> bytes:
            return self._content

        def read(self) -> bytes:
            return self._content

        def json(self) -> Any:
            if self._json_data is not None:
                return self._json_data
            try:
                return json.loads(self._content.decode("utf-8")) if self._content else {}
            except json.JSONDecodeError:
                return {}

        @property
        def text(self) -> str:
            try:
                return self._content.decode("utf-8")
            except Exception:  # pragma: no cover - defensive fallback
                return ""

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise HTTPError(f"Offline stub status {self.status_code}")

        def close(self) -> None:
            self._closed = True

        async def aclose(self) -> None:
            self._closed = True

        async def aiter_bytes(self):
            if not self._content:
                return
            yield self._content

        def iter_bytes(self):
            if not self._content:
                yield from ()
                return
            yield self._content

    def _build_response(method: str, url: str, **kwargs: Any) -> Response:
        json_payload = kwargs.get("json")
        data_payload = kwargs.get("data")
        content_payload = kwargs.get("content")
        if json_payload is not None:
            content = json.dumps(json_payload).encode("utf-8")
            json_data = json_payload
        elif data_payload is not None:
            if isinstance(data_payload, (dict, list)):
                content = json.dumps(data_payload).encode("utf-8")
            else:
                content = str(data_payload).encode("utf-8")
            json_data = data_payload
        else:
            if content_payload is None:
                content = b""
            elif isinstance(content_payload, bytes):
                content = content_payload
            else:
                content = str(content_payload).encode("utf-8")
            json_data = None
        return Response(
            200,
            headers=Headers(),
            content=content,
            request=Request(method, url),
            json_data=json_data,
        )

    class Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._closed = False

        def request(self, method: str, url: str, **kwargs: Any) -> Response:
            return _build_response(method, url, **kwargs)

        def get(self, url: str, **kwargs: Any) -> Response:
            return self.request("GET", url, **kwargs)

        def post(self, url: str, **kwargs: Any) -> Response:
            return self.request("POST", url, **kwargs)

        def close(self) -> None:
            self._closed = True

        def __enter__(self) -> "Client":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

    class AsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._closed = False

        async def request(self, method: str, url: str, **kwargs: Any) -> Response:
            return _build_response(method, url, **kwargs)

        async def get(self, url: str, **kwargs: Any) -> Response:
            return await self.request("GET", url, **kwargs)

        async def post(self, url: str, **kwargs: Any) -> Response:
            return await self.request("POST", url, **kwargs)

        def stream(self, method: str, url: str, **kwargs: Any):
            response = _build_response(method, url, **kwargs)

            class _StreamContext:
                def __init__(self, resp: Response) -> None:
                    self._response = resp

                async def __aenter__(self) -> Response:
                    return self._response

                async def __aexit__(self, exc_type, exc, tb) -> None:
                    # Mirror httpx behaviour by closing the response when leaving
                    # the streaming context.  ``aclose`` is available on the real
                    # response object while ``close`` is defined on this stub.
                    close_async = getattr(self._response, "aclose", None)
                    if callable(close_async):
                        result = close_async()
                        if inspect.isawaitable(result):
                            await result
                    close_sync = getattr(self._response, "close", None)
                    if callable(close_sync):
                        close_sync()
                    return None

            return _StreamContext(response)

        async def aclose(self) -> None:
            self._closed = True

        async def __aenter__(self) -> "AsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            await self.aclose()

    stub = SimpleNamespace(
        HTTPError=HTTPError,
        TimeoutException=TimeoutException,
        ConnectError=ConnectError,
        Request=Request,
        Response=Response,
        Headers=Headers,
        AsyncClient=AsyncClient,
        Client=Client,
        get=lambda url, *args, **kwargs: _build_response("GET", url, **kwargs),
        post=lambda url, *args, **kwargs: _build_response("POST", url, **kwargs),
        __offline_stub__=True,
    )
    return stub


def create_pydantic_stub():
    """Return minimal stand-ins for :mod:`pydantic` classes."""

    class ValidationError(ValueError):
        """Pydantic validation error placeholder."""

    class BaseModel:
        """Simplified Pydantic ``BaseModel`` replacement."""

        model_config: dict[str, Any] = {}

        def __init__(self, **data: Any) -> None:
            annotations = getattr(type(self), "__annotations__", {})
            for field in annotations:
                if hasattr(type(self), field):
                    setattr(self, field, getattr(type(self), field))
                else:
                    setattr(self, field, None)
            for key, value in data.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data: Any) -> "BaseModel":
            if not isinstance(data, dict):
                raise ValidationError("Expected mapping for model validation")
            return cls(**data)

        def model_dump(self) -> dict[str, Any]:
            annotations = getattr(type(self), "__annotations__", {})
            return {field: getattr(self, field, None) for field in annotations}

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:
        return dict(**kwargs)

    BaseModel.__offline_stub__ = True  # type: ignore[attr-defined]
    return BaseModel, ConfigDict, ValidationError
