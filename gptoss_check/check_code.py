import json
import logging
import os
import socket
import time
from contextlib import contextmanager
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from ipaddress import ip_address
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urljoin, urlparse, urlunparse

from services.logging_utils import sanitize_log_value


logger = logging.getLogger(__name__)

_http_client_error: Exception | None = None
try:  # pragma: no cover - exercised via docker compose integration
    import http_client as _http_client
except Exception as import_error:  # pragma: no cover - fallback for CI container
    _http_client_error = import_error
else:
    _httpx_candidate = getattr(_http_client, "httpx", None)
    _uses_stub = bool(getattr(_httpx_candidate, "__offline_stub__", False))
    if _httpx_candidate is None or _uses_stub:
        reason = "HTTPX stub" if _uses_stub else "missing httpx client"
        _http_client_error = RuntimeError(
            f"Unavailable HTTP client for GPT-OSS check: {reason}"
        )
    else:  # pragma: no cover - executed when a real httpx client is available
        get_httpx_client = _http_client.get_httpx_client
        HTTPError = _httpx_candidate.HTTPError  # type: ignore[attr-defined]
        httpx = _httpx_candidate  # type: ignore[assignment]
        _fallback_reason = None

if _http_client_error is not None:
    try:
        import httpx as _httpx  # type: ignore[import-not-found]
    except Exception as httpx_error:  # pragma: no cover - fallback when httpx missing
        _httpx = None
        _httpx_error = httpx_error
    else:  # pragma: no cover - executed when httpx import succeeds
        _httpx_error = None

    if _httpx is not None:  # pragma: no cover - executed when httpx is installed

        @contextmanager
        def get_httpx_client(timeout: float = 10.0, **kwargs):
            """Provide the project-wide HTTPX client helper."""

            kwargs.setdefault("timeout", timeout)
            kwargs.setdefault("trust_env", False)
            client = _httpx.Client(**kwargs)
            try:
                yield client
            finally:
                client.close()

        HTTPError = _httpx.HTTPError
        httpx = _httpx
        _fallback_reason = _http_client_error
    else:  # pragma: no cover - executed when httpx is unavailable

        class HTTPError(RuntimeError):
            """Base error raised by the lightweight HTTP client."""

        class _SimpleResponse:
            """Lightweight response object mimicking :class:`httpx.Response`."""

            def __init__(self, status_code: int, headers: dict[str, str], body: bytes) -> None:
                self.status_code = status_code
                self.headers = headers
                self._body = body

            def json(self) -> object:
                if not self._body:
                    return {}
                try:
                    return json.loads(self._body.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValueError("Response is not valid JSON") from exc

            def close(self) -> None:  # pragma: no cover - no resources to release
                return

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise HTTPError(f"HTTP request failed with status {self.status_code}")

        class _SimpleClient:
            """Minimal HTTP client that uses the standard library."""

            def __init__(self, timeout: float = 10.0) -> None:
                self.timeout = timeout

            @staticmethod
            def _is_local_hostname(hostname: str | None) -> bool:
                """Return ``True`` when *hostname* refers to a local address."""

                if not hostname:
                    return False

                lowered = hostname.lower()
                if lowered == "localhost":
                    return True

                try:
                    parsed_ip = ip_address(lowered)
                except ValueError:
                    pass
                else:
                    return parsed_ip.is_loopback or parsed_ip.is_private

                try:
                    addrinfo = socket.getaddrinfo(hostname, None)
                except socket.gaierror:
                    return False

                for _family, _socktype, _proto, _canonname, sockaddr in addrinfo:
                    address = sockaddr[0]
                    try:
                        parsed = ip_address(address)
                    except ValueError:
                        continue
                    if parsed.is_loopback or parsed.is_private:
                        return True
                return False

            def _request(
                self,
                method: str,
                url: str,
                *,
                body: bytes | None,
                headers: dict[str, str],
            ) -> _SimpleResponse:
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"}:
                    raise HTTPError(f"Unsupported URL scheme: {parsed.scheme}")

                if parsed.scheme == "http" and not self._is_local_hostname(parsed.hostname):
                    raise HTTPError(
                        "Insecure HTTP connections are only allowed for localhost"
                    )

                connection_cls = HTTPConnection if parsed.scheme == "http" else HTTPSConnection
                host = parsed.hostname or ""
                port = parsed.port
                path = parsed.path or "/"
                if parsed.query:
                    path = f"{path}?{parsed.query}"

                connection = None
                try:
                    connection = connection_cls(host, port, timeout=self.timeout)  # type: ignore[arg-type]
                    connection.request(method.upper(), path, body=body, headers=headers)
                    response = connection.getresponse()
                    payload = response.read()
                    header_map = {key: value for key, value in response.getheaders()}
                    status = response.status
                except (HTTPException, OSError, ValueError, socket.error) as exc:
                    raise HTTPError(str(exc)) from exc
                finally:
                    if connection is not None:
                        try:
                            connection.close()  # type: ignore[has-type]
                        except Exception as close_exc:  # pragma: no cover - best effort cleanup
                            logger.debug(
                                "Failed to close HTTP connection: %s",
                                sanitize_log_value(str(close_exc)),
                            )

                return _SimpleResponse(status, header_map, payload)

            def request(self, method: str, url: str, **kwargs) -> _SimpleResponse:
                headers = {str(k): str(v) for k, v in (kwargs.get("headers") or {}).items()}
                json_payload = kwargs.get("json")
                data_payload = kwargs.get("data")
                body: bytes | None = None
                if json_payload is not None:
                    body = json.dumps(json_payload).encode("utf-8")
                    headers.setdefault("Content-Type", "application/json")
                elif data_payload is not None:
                    if isinstance(data_payload, bytes):
                        body = data_payload
                    else:
                        body = str(data_payload).encode("utf-8")
                return self._request(method, url, body=body, headers=headers)

            def get(self, url: str, **kwargs) -> _SimpleResponse:
                return self.request("GET", url, **kwargs)

            def post(self, url: str, **kwargs) -> _SimpleResponse:
                return self.request("POST", url, **kwargs)

            def close(self) -> None:  # pragma: no cover - nothing to close
                return

            def __enter__(self) -> "_SimpleClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                self.close()

        @contextmanager
        def get_httpx_client(timeout: float = 10.0, **kwargs):
            """Provide a minimal HTTP client backed by the standard library."""

            client = _SimpleClient(timeout=timeout)
            try:
                yield client
            finally:
                client.close()

        httpx = SimpleNamespace(HTTPError=HTTPError)
        _fallback_reason = _http_client_error

else:  # pragma: no cover - import succeeds in fully configured environments
    _fallback_reason = None

try:  # pragma: no cover - exercised via docker compose integration
    from gpt_client import GPTClientError as _GPTClientError, query_gpt as _query_gpt
except Exception as import_error_gpt:  # pragma: no cover - fallback for CI container

    class GPTClientError(RuntimeError):
        """Fallback GPT client error used when trading bot modules are unavailable."""

    def _extract_choice_text(choice: object) -> str | None:
        """Return textual content from a single GPT-OSS choice payload."""

        if not isinstance(choice, dict):
            return None

        text = choice.get("text")
        if isinstance(text, str) and text:
            return text

        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                pieces: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        fragment = item.get("text")
                    else:
                        fragment = item.get("content")
                    if isinstance(fragment, str) and fragment:
                        pieces.append(fragment)
                if pieces:
                    return "".join(pieces)

        content = choice.get("content")
        if isinstance(content, str) and content:
            return content

        return None

    def query_gpt(prompt: str) -> str:
        api_url = os.getenv("GPT_OSS_API")
        if not api_url:
            raise RuntimeError("Переменная окружения GPT_OSS_API не установлена")

        completions_url = _build_completions_url(api_url)
        payload = {"prompt": prompt}
        model = os.getenv("GPT_OSS_MODEL")
        if model:
            payload["model"] = model

        with get_httpx_client(timeout=30, trust_env=False) as client:
            response = client.post(completions_url, json=payload)
            try:
                response.raise_for_status()
                data = response.json()
            except ValueError as exc:  # pragma: no cover - unexpected API response
                raise RuntimeError("Некорректный JSON-ответ от GPT-OSS") from exc
            finally:
                response.close()

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Некорректный ответ GPT-OSS: {data!r}")

        text = _extract_choice_text(choices[0])
        if text is None:
            raise RuntimeError(f"Некорректный ответ GPT-OSS: {data!r}")
        return text

    logger.debug("Using fallback GPT client for GPT-OSS check: %s", import_error_gpt)
else:  # pragma: no cover - import succeeds in fully configured environments
    GPTClientError = _GPTClientError
    query_gpt = _query_gpt


def _normalize_api_base(api_url: str) -> str:
    """Return the base GPT-OSS URL stripped from any ``/v1`` suffix."""

    parsed = urlparse(api_url)
    path = parsed.path
    v1_index = path.find("/v1")
    if v1_index != -1:
        path = path[:v1_index]
    base = urlunparse(parsed._replace(path=path.rstrip("/")))
    return base.rstrip("/") + "/"


def _build_completions_url(api_url: str) -> str:
    """Normalize the GPT-OSS URL so requests always target ``/v1/completions``."""

    return urljoin(_normalize_api_base(api_url), "v1/completions")


def _build_health_url(api_url: str) -> str:
    """Normalize the GPT-OSS URL so requests target ``/v1/health``."""

    return urljoin(_normalize_api_base(api_url), "v1/health")


def wait_for_api(api_url: str, timeout: int | None = None) -> None:
    """Ожидать готовности сервера GPT-OSS."""

    if timeout is None:
        try:
            timeout = int(os.getenv("GPT_OSS_WAIT_TIMEOUT", "300"))
        except ValueError:
            timeout = 300

    deadline = time.time() + timeout
    health_url = _build_health_url(api_url)
    completions_url = _build_completions_url(api_url)

    while time.time() < deadline:
        try:
            with get_httpx_client(timeout=5, trust_env=False) as client:
                response = client.get(health_url)
                try:
                    response.raise_for_status()
                finally:
                    response.close()
            return
        except HTTPError:
            pass

        payload = {"prompt": "ping"}
        model = os.getenv("GPT_OSS_MODEL")
        if model:
            payload["model"] = model

        try:
            with get_httpx_client(timeout=5, trust_env=False) as client:
                response = client.post(completions_url, json=payload)
                try:
                    status_code = getattr(response, "status_code", None)
                    if isinstance(status_code, int) and status_code < 500:
                        return
                    response.raise_for_status()
                finally:
                    response.close()
        except HTTPError:
            time.sleep(1)

    raise RuntimeError(f"Сервер GPT-OSS по адресу {api_url} не отвечает")



def query(prompt: str) -> str:
    """Отправить текст на сервер GPT-OSS и вернуть полученный ответ."""
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        raise RuntimeError("Переменная окружения GPT_OSS_API не установлена")

    max_retries = 3
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            return query_gpt(prompt)
        except GPTClientError as err:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Ошибка запроса к GPT-OSS API: {err}"
                ) from err
            time.sleep(backoff)
            backoff *= 2


def send_telegram(msg: str) -> None:
    """Отправить сообщение в Telegram, если заданы токен и chat_id."""
    if os.getenv("TEST_MODE") == "1":
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            with get_httpx_client(timeout=15, trust_env=False) as client:
                response = client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={"chat_id": chat_id, "text": msg[:4000]},
                )
                try:
                    response.raise_for_status()
                finally:
                    response.close()
        except HTTPError as err:
            logger.warning("⚠️ Failed to send Telegram message: %s", err)


def run() -> None:
    """Run GPT-OSS analysis for configured files."""
    paths_env = os.getenv("CHECK_CODE_PATH", "trading_bot.py")
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        warning = "Переменная окружения GPT_OSS_API не установлена, проверка пропущена"
        logger.warning(warning)
        send_telegram(warning)
        return
    try:
        wait_for_api(api_url)
    except RuntimeError as err:
        warning = f"{err}, проверка пропущена"
        logger.warning(warning)
        send_telegram(warning)
        return
    repo_root = Path(__file__).resolve().parent.parent
    for filename in (p.strip() for p in paths_env.split(",") if p.strip()):
        path = repo_root / filename
        if not path.exists():
            warning = f"⚠️ {filename} not found, skipping"
            logger.warning(warning)
            send_telegram(warning)
            continue

        try:
            code = path.read_text(encoding="utf-8")
        except OSError as exc:
            warning = (
                f"⚠️ Не удалось прочитать {filename}: "
                f"{exc}".replace(str(path), sanitize_log_value(str(path)))
            )
            logger.warning(warning)
            send_telegram(warning)
            continue

        prompt = (
            "Проанализируй код Python. Выяви ошибки, уязвимости, улучшения. "
            "Объясни сигналы стратегии:\n" + code
        )
        try:
            result = query(prompt)
        except RuntimeError as err:
            logger.error("\n📄 %s\n%s\n", filename, err)
            send_telegram(f"📄 {filename}\n{err}")
            continue
        except Exception as err:  # pragma: no cover - unexpected GPT errors
            message = (
                f"Непредвиденная ошибка GPT-OSS: {err}".replace(
                    os.getenv("GPT_OSS_API", ""), "<redacted>"
                )
            )
            logger.error("\n📄 %s\n%s\n", filename, message)
            send_telegram(f"📄 {filename}\n{message}")
            continue

        logger.info("\n📄 %s\n%s\n", filename, result)
        send_telegram(f"📄 {filename}\n{result}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    run()
