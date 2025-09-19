"""Utility script used by the GPT-OSS review GitHub workflow.

The workflow previously relied on shell pipelines with ``jq`` and ``curl`` to
prepare the request payload and parse responses from the mock GPT-OSS server.
That approach was brittle – any quoting issue or transient HTTP failure caused
the entire job to exit with a non-zero status.  This module replaces the shell
logic with a small Python implementation that gracefully handles errors and
reports the result back to GitHub Actions via ``GITHUB_OUTPUT``.

The module keeps dependencies minimal by using only the standard-library
``urllib`` helpers for HTTP access.  It can also be imported from unit tests to
validate individual helper functions.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urlparse
_PROMPT_PREFIX = "Review the following diff and provide feedback:\n"


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
    req = request.Request(api_url, data=data, headers=headers)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read()
    except TimeoutError as exc:
        raise RuntimeError(f"Не удалось подключиться к GPT-OSS ({api_url}): {exc}") from exc
    except error.HTTPError as exc:
        raise RuntimeError(
            f"Сервер GPT-OSS вернул ошибку {exc.code}: {exc.reason}"
        ) from exc
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Ошибка при обращении к GPT-OSS ({api_url}): {reason}") from exc

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
        text = content.get("text")
        if isinstance(text, str) and text:
            pieces.append(text)
        inner = content.get("content")
        if isinstance(inner, list):
            for part in inner:
                pieces.extend(_flatten_content(part))
        return pieces

    if isinstance(content, list):
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
