"""Простая заглушка для пакета :mod:`flask` в офлайн-режиме.

Эта реализация покрывает минимальный API, который используют сервисы бота
и тесты заглушек: ``Flask``, ``Response``, ``jsonify`` и ``request``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Iterable


class Response:
    def __init__(self, json: Any = None, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self._json = json
        self.status_code = int(status)
        self.headers = headers or {}

    def get_json(self) -> Any:
        return self._json


request = SimpleNamespace(endpoint=None, json=None, args={}, form={}, headers={}, data=b"", method=None)


def jsonify(obj: Any = None, **kwargs: Any) -> Response:
    payload = obj if obj is not None else kwargs
    return Response(payload, status=200)


class _TestClient:
    def __init__(self, routes: dict[tuple[str, str], Callable[..., Any]]) -> None:
        self._routes = routes

    def _dispatch(self, method: str, path: str, json: Any = None) -> Response:
        handler = self._routes.get((method.upper(), path))
        request.method = method.upper()
        request.endpoint = path
        request.json = json
        if handler is None:
            return Response({"error": "not found"}, status=404)
        result = handler()
        status_code = 200
        body = result
        if isinstance(result, tuple):
            if len(result) >= 1:
                body = result[0]
            if len(result) >= 2:
                status_code = result[1]
        if isinstance(body, Response):
            body.status_code = status_code if body.status_code == 200 else body.status_code
            return body
        return Response(body, status=status_code)

    def get(self, path: str, **_kwargs: Any) -> Response:
        return self._dispatch("GET", path)

    def post(self, path: str, json: Any = None, **_kwargs: Any) -> Response:
        return self._dispatch("POST", path, json=json)


class Flask:
    def __init__(self, name: str) -> None:
        self.name = name
        self._routes: dict[tuple[str, str], Callable[..., Any]] = {}
        self.testing = False

    def route(self, rule: str, methods: Iterable[str] | None = None, **_kwargs: Any):  # noqa: ANN001
        allowed = list(methods) if methods is not None else ["GET"]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            for method in allowed:
                self._routes[(method.upper(), rule)] = func
            return func

        return decorator

    def test_client(self) -> _TestClient:
        return _TestClient(self._routes)


current_app = None

__all__ = ["Flask", "Response", "jsonify", "request", "current_app"]
