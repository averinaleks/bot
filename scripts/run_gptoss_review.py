"""Utility helpers for generating GPT-OSS code review comments on CI.

The workflow previously relied on shell pipelines with ``jq`` and ``curl`` to
prepare the request payload and parse responses from the mock GPT-OSS server.
That approach was brittle – any quoting issue or transient HTTP failure caused
the entire job to exit with a non-zero status.  This module replaces the shell
logic with a small Python implementation that gracefully handles errors and
reports the result back to GitHub Actions via ``GITHUB_OUTPUT``.

Only the Python standard library is required which keeps the workflow resilient
on fresh GitHub runners where third-party packages like ``requests`` may be
missing.  The module can also be imported from unit tests to validate
individual helper functions.
"""

from __future__ import annotations

import argparse
import logging
import http.client
import ipaddress
import json
import os
import socket
import ssl
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_PROMPT_PREFIX = "Review the following diff and provide feedback:\n"
_ALLOWED_HOSTS_ENV = "GPT_OSS_ALLOWED_HOSTS"


logger = logging.getLogger(__name__)


class EmptyDiffError(RuntimeError):
    """Raised when the diff file is missing or contains no meaningful data."""


@dataclass
class ReviewResult:
    """Container for the generated review text and control flags."""

    review: str
    has_content: bool


def _read_diff(diff_path: Path) -> str:
    """Return diff contents, normalising encoding issues."""

    try:
        data = diff_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:  # pragma: no cover - handled by caller
        raise EmptyDiffError(f"Diff файл {diff_path} не найден") from exc
    except OSError as exc:  # pragma: no cover - unexpected filesystem error
        raise RuntimeError(f"Не удалось прочитать diff {diff_path}: {exc}") from exc

    if not data.strip():
        raise EmptyDiffError(f"Diff файл {diff_path} пустой")
    return data


def _build_payload(diff_text: str, model: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": f"{_PROMPT_PREFIX}{diff_text}",
            }
        ]
    }
    if model:
        payload["model"] = model
    return payload


def _normalise_host(value: str | None) -> str:
    """Return a lower-case host without brackets or whitespace."""

    host = (value or "").strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def _load_allowed_hosts() -> set[str]:
    """Load optional allow-list for HTTPS hosts from the environment."""

    raw = os.getenv(_ALLOWED_HOSTS_ENV, "")
    hosts: set[str] = set()
    for part in raw.split(","):
        normalised = _normalise_host(part)
        if normalised:
            hosts.add(normalised)
    return hosts


def _resolve_host_ips(hostname: str) -> set[str]:
    """Return resolved IP addresses for *hostname* handling IPv4/IPv6."""

    try:
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC)
    except socket.gaierror as exc:
        raise RuntimeError(f"не удалось разрешить хост {hostname!r}: {exc}") from exc
    ips: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        address = sockaddr[0]
        if isinstance(address, bytes):
            try:
                address = address.decode()
            except UnicodeDecodeError:
                continue
        ips.add(str(address))
    return ips


def _host_ips_are_private(ips: set[str]) -> bool:
    """Return ``True`` when every IP in *ips* is loopback or private."""

    if not ips:
        return False
    for ip_text in ips:
        try:
            address = ipaddress.ip_address(ip_text)
        except ValueError:
            return False
        if not (address.is_loopback or address.is_private):
            return False
    return True


def _perform_http_request(
    url: str, data: bytes, headers: dict[str, str], timeout: float
) -> tuple[int, str, bytes]:
    """Execute an HTTP(S) request and return status, reason and body."""

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if not hostname:
        raise RuntimeError("URL GPT-OSS не содержит hostname")

    host = hostname
    if parsed.username or parsed.password:
        raise RuntimeError("URL GPT-OSS не должен содержать учетные данные")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    resolved_ips = _resolve_host_ips(hostname)
    allowed_hosts = _load_allowed_hosts() if parsed.scheme == "https" else set()

    connection_kwargs: dict[str, object] = {"timeout": timeout}
    connection_cls: type[http.client.HTTPConnection] | type[http.client.HTTPSConnection]
    if parsed.scheme == "https":
        if allowed_hosts and hostname not in allowed_hosts:
            if not _host_ips_are_private(resolved_ips):
                raise RuntimeError(
                    "Хост GPT-OSS должен быть в списке GPT_OSS_ALLOWED_HOSTS или разрешаться в приватные IP"
                )
        elif not _host_ips_are_private(resolved_ips):
            raise RuntimeError(
                "HTTPS GPT-OSS хост должен разрешаться в приватные или loopback IP"
            )
        connection_cls = http.client.HTTPSConnection
        connection_kwargs["context"] = ssl.create_default_context()
    else:
        if not _host_ips_are_private(resolved_ips):
            raise RuntimeError(
                "HTTP GPT-OSS URL разрешается в неприватный адрес"
            )
        connection_cls = http.client.HTTPConnection

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    status = 0
    reason = ""
    payload = b""
    connection: http.client.HTTPConnection | http.client.HTTPSConnection
    port = parsed.port
    connection = connection_cls(host, port, **connection_kwargs)  # type: ignore[arg-type]

    try:
        connection.request("POST", path, body=data, headers=headers)
        response = connection.getresponse()
        status = int(getattr(response, "status", 0) or 0)
        reason = getattr(response, "reason", "") or ""
        payload = response.read()
    except socket.timeout as exc:
        raise TimeoutError(str(exc) or "timed out") from exc
    except TimeoutError:
        raise
    except ssl.SSLError as exc:
        raise ConnectionError(str(exc) or "SSL error") from exc
    except http.client.HTTPException as exc:
        raise ConnectionError(str(exc) or exc.__class__.__name__) from exc
    except OSError as exc:
        raise ConnectionError(str(exc) or exc.__class__.__name__) from exc
    finally:
        try:
            connection.close()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.warning("Failed to close HTTP connection cleanly: %s", exc)

    return status, reason, payload


def _send_request(api_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    parsed = urlparse(api_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("Недопустимый URL для GPT-OSS API")
    if parsed.scheme == "http":
        host = parsed.hostname or ""
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except ValueError:
            if host.lower() != "localhost":
                raise RuntimeError("Небезопасный HTTP URL для GPT-OSS API")

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    sanitised = parsed._replace(path=parsed.path or "/", fragment="")
    url = sanitised.geturl()

    try:
        status, reason, body = _perform_http_request(url, data, headers, timeout)
    except TimeoutError as exc:
        raise RuntimeError(f"Не удалось подключиться к GPT-OSS ({api_url}): {exc}") from exc
    except ConnectionError as exc:
        raise RuntimeError(f"Ошибка при обращении к GPT-OSS ({api_url}): {exc}") from exc

    if status >= 400:
        raise RuntimeError(f"Сервер GPT-OSS вернул ошибку {status}: {reason}")

    try:
        return json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError("Сервер GPT-OSS вернул некорректный JSON") from exc


def _flatten_content(content: Any) -> list[str]:
    """Return textual fragments from arbitrary message content structures."""

    pieces: list[str] = []
    if isinstance(content, str):
        if content:
            pieces.append(content)
        return pieces

    if isinstance(content, dict):
        # Contemporary chat APIs may wrap textual responses into nested
        # dictionaries, e.g. ``{"text": {"value": "..."}}`` or expose both
        # ``text`` and ``content`` keys simultaneously.  Recursively flatten the
        # known containers instead of assuming a specific schema.
        text = content.get("text")
        if text is not None:
            pieces.extend(_flatten_content(text))
        value = content.get("value")
        if value is not None:
            pieces.extend(_flatten_content(value))
        inner = content.get("content")
        if inner is not None:
            pieces.extend(_flatten_content(inner))
        return pieces

    if isinstance(content, (list, tuple)):
        for part in content:
            pieces.extend(_flatten_content(part))
    return pieces


def _extract_review(response: dict[str, Any] | Any) -> str:
    """Pull textual review content from OpenAI-compatible response payloads."""

    if not isinstance(response, dict):
        return ""

    choices = response.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            fragments = _flatten_content(content)
            combined = "".join(fragments).strip()
            if combined:
                return combined
        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        # Some implementations put the response directly under ``content``.
        content = choice.get("content")
        fragments = _flatten_content(content)
        combined = "".join(fragments).strip()
        if combined:
            return combined
    return ""


def generate_review(
    diff_path: Path,
    model: str | None,
    api_url: str,
    timeout: float = 30.0,
) -> ReviewResult:
    """Produce a review for the supplied diff file."""

    diff_text = _read_diff(diff_path)
    payload = _build_payload(diff_text, model)
    response = _send_request(api_url, payload, timeout)
    review = _extract_review(response)
    if not review:
        raise RuntimeError("GPT-OSS не вернул текст отзыва")
    return ReviewResult(review=review, has_content=True)


def _write_github_output(has_content: bool) -> None:
    """Append workflow outputs to the special file if available."""

    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return
    try:
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(f"has_content={'true' if has_content else 'false'}\n")
    except OSError as exc:  # pragma: no cover - extremely rare on GH runners
        print(f"::warning::Не удалось записать GITHUB_OUTPUT: {exc}", file=sys.stderr)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GPT-OSS review")
    parser.add_argument("--diff", default="diff.patch", help="Путь к diff-файлу")
    parser.add_argument(
        "--output", default="review.md", help="Файл, куда сохранить отзыв"
    )
    parser.add_argument("--model", default=os.getenv("MODEL_NAME"))
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="Адрес GPT-OSS API",
    )
    parser.add_argument("--timeout", type=float, default=30.0)

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise ValueError(
            "Неизвестные аргументы: " + " ".join(unknown)
        )
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except ValueError as exc:
        print(f"::warning::{exc}", file=sys.stderr)
        _write_github_output(False)
        return 0
    except SystemExit as exc:
        # argparse uses SystemExit for ``--help`` and fatal parsing errors.
        code = getattr(exc, "code", 0)
        if code not in (0, None):
            print(
                f"::warning::Парсер аргументов завершился с кодом {code}",
                file=sys.stderr,
            )
        _write_github_output(False)
        return 0
    diff_path = Path(args.diff)
    output_path = Path(args.output)

    try:
        result = generate_review(diff_path, args.model, args.api_url, args.timeout)
    except EmptyDiffError as exc:
        print(f"::notice::{exc}", file=sys.stderr)
        _write_github_output(False)
        return 0
    except RuntimeError as exc:
        print(f"::warning::{exc}", file=sys.stderr)
        _write_github_output(False)
        return 0
    except Exception as exc:  # pragma: no cover - defensive guard
        print(
            f"::error::Неожиданная ошибка при генерации обзора: {exc}",
            file=sys.stderr,
        )
        _write_github_output(False)
        return 0

    try:
        output_path.write_text(result.review, encoding="utf-8")
    except OSError as exc:
        print(f"::warning::Не удалось записать отзыв: {exc}", file=sys.stderr)
        _write_github_output(False)
        return 0

    _write_github_output(result.has_content)
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Wrapper around :func:`main` that shields CI from fatal exits."""

    try:
        return main(argv)
    except SystemExit as exc:
        code = exc.code  # ``code`` may be ``None`` or non-integer
        if code not in (0, None):
            print(
                f"::warning::Скрипт завершился с кодом {code}. Возвращаю 0, чтобы не прерывать job.",
                file=sys.stderr,
            )
            _write_github_output(False)
        return 0
    except BaseException as exc:  # pragma: no cover - defensive guard
        print(
            f"::error::Критическое исключение в CLI: {exc}",
            file=sys.stderr,
        )
        _write_github_output(False)
        return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(cli())
