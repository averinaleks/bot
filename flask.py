from __future__ import annotations

import json as _json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from urllib.parse import unquote

_current_request: _Request | None = None


class _Request:
    def __init__(
        self,
        json_data: Any,
        raw_data: bytes | None = None,
        method: str = "",
        path: str = "",
        headers: Optional[Dict[str, str]] | None = None,
    ) -> None:
        self._json = json_data
        self._raw = raw_data
        self.method = method
        self.path = path
        self.headers: Dict[str, str] = headers or {}

    def get_json(self, force: bool = False) -> Any:
        if self._json is not None:
            return self._json
        if not force or self._raw is None:
            return None
        try:
            self._json = _json.loads(self._raw.decode())
        except Exception:
            return None
        return self._json


class _RequestProxy:
    def __getattr__(self, name: str) -> Any:
        if _current_request is None:
            raise RuntimeError("No request context")
        return getattr(_current_request, name)


request = _RequestProxy()


class Response:
    def __init__(self, data: Any = None, status: int = 200,
                 headers: Optional[Dict[str, str]] = None) -> None:
        self.status_code = status
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)
        self._json = None
        if isinstance(data, (dict, list)):
            self._json = data
            self.data = _json.dumps(data).encode()
        elif isinstance(data, bytes):
            self.data = data
            try:
                self._json = _json.loads(data.decode())
            except Exception:
                pass
        elif isinstance(data, str):
            self.data = data.encode()
            try:
                self._json = _json.loads(data)
            except Exception:
                pass
        elif data is None:
            self.data = b""
        else:
            self.data = str(data).encode()

    def get_json(self) -> Any:
        return self._json

    @property
    def json(self) -> Any:
        return self._json


def jsonify(obj: Any) -> Response:
    return Response(obj)


class Flask:
    def __init__(self, name: str) -> None:
        self.name = name
        self.config: Dict[str, Any] = {}
        self._routes: list[Tuple[str, Callable[..., Any]]] = []
        self._error_handlers: Dict[int, Callable[..., Any]] = {}
        self._before_request: list[Callable[[], None]] = []
        self._before_first: list[Callable[[], None]] = []
        self._teardown: list[Callable[[BaseException | None], None]] = []
        self._first_done = False

    def route(self, rule: str, methods: Iterable[str] | None = None) -> Callable:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes.append((rule, func))
            return func
        return decorator

    def errorhandler(self, code: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._error_handlers[code] = func
            return func
        return decorator

    def before_request(self, func: Callable[[], None]) -> Callable[[], None]:
        self._before_request.append(func)
        return func

    def before_first_request(self, func: Callable[[], None]) -> Callable[[], None]:
        self._before_first.append(func)
        return func

    def teardown_appcontext(self, func: Callable[[BaseException | None], None]) -> Callable[[BaseException | None], None]:
        self._teardown.append(func)
        return func

    def _find_handler(self, path: str) -> Tuple[Callable[..., Any] | None, Dict[str, str]]:
        for rule, func in self._routes:
            if rule == path:
                return func, {}
            if "<" in rule:
                prefix, var = rule.split("<", 1)
                var = var.rstrip(">")
                if path.startswith(prefix):
                    return func, {var: unquote(path[len(prefix):])}
        return None, {}

    def _dispatch(
        self,
        path: str,
        json_data: Any,
        raw: bytes | None,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
    ) -> Response:
        global _current_request
        handler, kwargs = self._find_handler(path)
        if handler is None:
            return Response({"error": "not found"}, status=404)
        _current_request = _Request(json_data, raw, method, path, headers)
        try:
            for func in self._before_request:
                func()
            rv = handler(**kwargs)
        except Exception as exc:
            err_handler = self._error_handlers.get(500)
            if err_handler:
                rv = err_handler(exc)
            else:
                raise
        finally:
            _current_request = None
        status = 200
        if isinstance(rv, tuple):
            rv, status = rv
        if isinstance(rv, Response):
            rv.status_code = status
            return rv
        return Response(rv, status=status)

    def test_client(self):
        app = self

        class _Client:
            def __enter__(self) -> "_Client":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                for func in app._teardown:
                    func(None)

            def _prep_first(self) -> None:
                if not app._first_done:
                    for func in app._before_first:
                        func()
                    app._first_done = True

            def get(self, path: str, headers: Optional[Dict[str, str]] = None) -> Response:
                self._prep_first()
                return app._dispatch(path, None, None, "GET", headers)

            def post(
                self,
                path: str,
                json: Any | None = None,
                data: Any | None = None,
                content: bytes | None = None,
                headers: Optional[Dict[str, str]] = None,
            ) -> Response:
                self._prep_first()
                raw = b""
                jdata = None
                if json is not None:
                    jdata = json
                    raw = json_dump_bytes(json)
                elif content is not None:
                    raw = content
                    try:
                        jdata = _json.loads(content)
                    except Exception:
                        jdata = None
                elif data is not None:
                    raw = data if isinstance(data, bytes) else str(data).encode()
                    try:
                        jdata = _json.loads(raw)
                    except Exception:
                        jdata = None
                return app._dispatch(path, jdata, raw, "POST", headers)

        return _Client()

    def run(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _call(self, method: str) -> None:
                if not app._first_done:
                    for func in app._before_first:
                        func()
                    app._first_done = True
                length = self.headers.get("Content-Length")
                raw = b""
                if length:
                    raw = self.rfile.read(int(length))
                elif self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    while True:
                        line = self.rfile.readline().strip()
                        if not line:
                            continue
                        size = int(line, 16)
                        if size == 0:
                            break
                        raw += self.rfile.read(size)
                        self.rfile.readline()
                try:
                    jdata = _json.loads(raw) if raw else None
                except Exception:
                    jdata = None
                resp = app._dispatch(self.path, jdata, raw, method, dict(self.headers))
                self.send_response(resp.status_code)
                for k, v in resp.headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.data)

            do_GET = lambda self: self._call("GET")  # type: ignore
            do_POST = lambda self: self._call("POST")  # type: ignore

        httpd = HTTPServer((host, port), Handler)
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()


def json_dump_bytes(obj: Any) -> bytes:
    return _json.dumps(obj).encode()


__all__ = ["Flask", "jsonify", "request", "Response"]
