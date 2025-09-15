"""Mock GPT-OSS server used in CI to emulate chat completions.

The implementation intentionally relies only on the Python standard library so
that it can be executed in GitHub Actions without installing extra
dependencies.  It exposes a minimal subset of the OpenAI-compatible API:

* ``GET /v1/models`` – returns a single mock model identifier.
* ``POST /v1/chat/completions`` – accepts the diff inside the chat messages
  and produces a deterministic textual review.

The responses are deliberately lightweight yet informative.  They include basic
statistics about the diff and highlight a few common red flags (e.g. leftover
``print`` calls or ``TODO`` markers).  This keeps the "GPT-OSS Code Review"
workflow functional even when real LLM resources are unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable, Sequence

_MODEL_NAME = os.getenv("MODEL_NAME", "gptoss-mock")
_RESPONSE_LIMIT = 4000  # characters


@dataclass
class DiffStats:
    """Summary of parsed diff information."""

    files: int
    additions: int
    deletions: int
    issues: list[str]


_DIFF_HEADER_RE = re.compile(r"^\+\+\+ b/(.+)$")


def _iter_messages_content(messages: Sequence[dict]) -> Iterable[str]:
    """Yield message content strings in the order they appear."""

    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            # Some chat APIs provide structured content; concatenate text blocks.
            pieces: list[str] = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    text = chunk.get("text", "")
                    if isinstance(text, str):
                        pieces.append(text)
            if pieces:
                yield "".join(pieces)


def _extract_diff(messages: Sequence[dict]) -> str:
    """Return diff text from the latest user message."""

    for content in reversed(list(_iter_messages_content(messages))):
        # We expect the diff to follow an instruction header.
        if "---" in content or "+++" in content:
            return content
    # Fall back to the last message verbatim.
    return next(reversed(list(_iter_messages_content(messages))), "")


def _analyze_diff(diff: str) -> DiffStats:
    files: set[str] = set()
    additions = 0
    deletions = 0
    issues: list[str] = []
    current_file = ""

    for raw_line in diff.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("+++ "):
            match = _DIFF_HEADER_RE.match(line)
            if match:
                current_file = match.group(1)
                files.add(current_file)
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue

        if line.startswith("+"):
            additions += 1
            content = line[1:].strip()
            if "TODO" in content or "FIXME" in content:
                issues.append(
                    f"В файле {current_file or 'неопределённом'} найден незавершённый комментарий: {content!r}."
                )
            if content.startswith("print("):
                issues.append(
                    f"В файле {current_file or 'неопределённом'} остался отладочный print: {content!r}."
                )
            if content.startswith("import ") and " as " not in content and "," in content:
                issues.append(
                    f"Проверьте импорты в {current_file or 'неопределённом'} – множественный import в одной строке сложнее поддерживать."
                )
            continue
        if line.startswith("-"):
            deletions += 1

    return DiffStats(files=len(files) or 0, additions=additions, deletions=deletions, issues=issues)


def _build_review(stats: DiffStats) -> str:
    summary = [
        "## Автоматический обзор",
        f"- Файлов затронуто: {stats.files}.",
        f"- Добавлено строк: {stats.additions}, удалено строк: {stats.deletions}.",
    ]

    if stats.issues:
        summary.append("### Возможные проблемы")
        for issue in stats.issues[:10]:
            summary.append(f"* {issue}")
    else:
        summary.append(
            "⚙️ Я не обнаружил явных проблем. Проверьте, соответствует ли код внутренним стандартам и протестируйте изменения."
        )

    review = "\n".join(summary)
    if len(review) > _RESPONSE_LIMIT:
        review = review[: _RESPONSE_LIMIT - 3] + "..."
    return review


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = "GptossMock/1.0"

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path.rstrip("/") == "/v1/models":
            self._send_json({"data": [{"id": _MODEL_NAME}]})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path.rstrip("/") not in {"/v1/chat/completions", "/v1/completions"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        length_header = self.headers.get("Content-Length")
        if not length_header:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing Content-Length header")
            return
        try:
            length = int(length_header)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length header")
            return

        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON payload")
            return

        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            self.send_error(HTTPStatus.BAD_REQUEST, "messages must be a list")
            return

        diff = _extract_diff(messages)
        stats = _analyze_diff(diff)
        review = _build_review(stats)

        response = {
            "object": "chat.completion",
            "model": payload.get("model", _MODEL_NAME),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": review},
                }
            ],
        }
        self._send_json(response)

    def log_message(self, fmt: str, *args) -> None:  # pragma: no cover - reduce noise
        # Silence default request logging to keep CI output tidy.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mock GPT-OSS server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), _RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
