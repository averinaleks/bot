"""Простая заглушка для пакета :mod:`flask` в офлайн-режиме.

Эта реализация покрывает минимальный API, который используют сервисы бота
и тесты заглушек: ``Flask``, ``Response``, ``jsonify`` и ``request``.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


def _normalize_result(result: Any) -> Response:
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


class _TestClient:
    def __init__(self, app: "Flask") -> None:
        self._app = app

    def _dispatch(self, method: str, path: str, json: Any = None) -> Response:
        return self._app._handle_request(method, path, json=json)

    def get(self, path: str, **_kwargs: Any) -> Response:
        return self._dispatch("GET", path)

    def post(self, path: str, json: Any = None, **_kwargs: Any) -> Response:
        return self._dispatch("POST", path, json=json)


class Flask:
    def __init__(self, name: str) -> None:
        self.name = name
        self._routes: dict[tuple[str, str], Callable[..., Any]] = {}
        self.testing = False
        self._before_request_handlers: list[Callable[..., Any]] = []
        self._server: HTTPServer | None = None
        self._server_thread: threading.Thread | None = None

    def before_request(self, func: Callable[..., Any]):  # noqa: ANN001
        self._before_request_handlers.append(func)
        return func

    def route(self, rule: str, methods: Iterable[str] | None = None, **_kwargs: Any):  # noqa: ANN001
        allowed = list(methods) if methods is not None else ["GET"]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            for method in allowed:
                self._routes[(method.upper(), rule)] = func
            return func

        return decorator

    def test_client(self) -> _TestClient:
        return _TestClient(self)

    def _handle_request(self, method: str, path: str, json: Any = None) -> Response:
        handler = self._routes.get((method.upper(), path))
        request.method = method.upper()
        request.endpoint = path
        request.json = json

        for hook in self._before_request_handlers:
            hook_result = hook()
            if hook_result is not None:
                return _normalize_result(hook_result)

        if handler is None:
            return Response({"error": "not found"}, status=404)

        result = handler()
        return _normalize_result(result)

    def run(self, host: str = "127.0.0.1", port: int = 5000, **_kwargs: Any) -> None:  # noqa: ANN001
        """Minimal HTTP server to mimic :meth:`Flask.run` in tests."""

        app = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN001,D401 - suppress noisy logs
                return

            def _write_response(self, resp: Response) -> None:
                body = resp.get_json()
                payload = json.dumps(body).encode()
                self.send_response(resp.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                for key, value in resp.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802 - signature follows BaseHTTPRequestHandler
                response = app._handle_request("GET", self.path)
                self._write_response(response)

            def do_POST(self) -> None:  # noqa: N802 - signature follows BaseHTTPRequestHandler
                length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length) if length > 0 else b""
                try:
                    body = json.loads(raw_body.decode()) if raw_body else None
                except json.JSONDecodeError:
                    body = None
                response = app._handle_request("POST", self.path, json=body)
                self._write_response(response)

        self._server = HTTPServer((host, port), _Handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        try:
            self._server_thread.join()
        finally:
            self._server.shutdown()
            self._server.server_close()


current_app = None

__all__ = ["Flask", "Response", "jsonify", "request", "current_app"]
