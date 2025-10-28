# -*- coding: utf-8 -*-
"""Mock GPT-OSS server used by the GitHub Actions workflow.

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
import signal
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts._filesystem import write_secure_text


_MODEL_NAME = os.getenv("MODEL_NAME", "gptoss-mock")
_RESPONSE_LIMIT = 4000  # characters


@dataclass
class DiffStats:
    """Summary of parsed diff information."""

    files: int
    additions: int
    deletions: int
    issues: list[str]


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


def _parse_diff_header(line: str) -> str | None:
    """Return a sanitised path from a unified diff header line."""

    prefix = "+++ b/"
    if not line.startswith(prefix):
        return None

    candidate = line[len(prefix) :]
    # Remove diff metadata such as tabs or trailing spaces
    candidate = candidate.split("\t", 1)[0].rstrip()
    if not candidate or candidate == "/dev/null":
        return None

    # Strip optional quoting and redundant leading ./ segments
    if candidate.startswith("\"") and candidate.endswith("\""):
        candidate = candidate[1:-1]
    while candidate.startswith("./"):
        candidate = candidate[2:]

    if not candidate:
        return None

    # Reject absolute paths and traversal attempts
    if candidate.startswith(("/", "\\")):
        return None
    if candidate.startswith(".."):  # blocks ../ and ..\ styles
        return None

    # Avoid control characters or excessive paths that could exhaust memory
    if any(ord(ch) < 32 for ch in candidate):
        return None
    if len(candidate) > 512:
        return None

    return candidate


def _record_line_issue(content: str, current_file: str, issues: list[str]) -> None:
    """Append a human readable warning for ``content`` if needed."""

    location = current_file or "неопределённом"
    if "TODO" in content or "FIXME" in content:
        issues.append(
            f"В файле {location} найден незавершённый комментарий: {content!r}."
        )
    if content.startswith("print("):
        issues.append(
            f"В файле {location} остался отладочный print: {content!r}."
        )
    if content.startswith("import ") and " as " not in content and "," in content:
        issues.append(
            f"Проверьте импорты в {location} – множественный import в одной строке сложнее поддерживать."
        )


def _analyze_diff(diff: str) -> DiffStats:
    files: set[str] = set()
    additions = 0
    deletions = 0
    issues: list[str] = []
    current_file = ""

    for raw_line in diff.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("+++ "):
            header_path = _parse_diff_header(line)
            if header_path is not None:
                current_file = header_path
                files.add(header_path)
            else:
                current_file = ""
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue

        if line.startswith("+"):
            additions += 1
            content = line[1:].strip()
            if content:
                _record_line_issue(content, current_file, issues)
            continue
        if line.startswith("-"):
            deletions += 1

    return DiffStats(files=len(files) or 0, additions=additions, deletions=deletions, issues=issues)


def _analyze_prompt(prompt: str) -> DiffStats:
    """Return pseudo diff stats for a raw prompt body."""

    issues: list[str] = []
    additions = 0
    for raw_line in prompt.splitlines():
        content = raw_line.strip()
        if not content:
            continue
        additions += 1
        _record_line_issue(content, "представленном коде", issues)

    files = 1 if prompt.strip() else 0
    return DiffStats(files=files, additions=additions, deletions=0, issues=issues)


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
            "⚙️ Я не обнаружил явных проблем. Проверьте, соответствует ли код "
            "внутренним стандартам и протестируйте изменения."
        )

    review = "\n".join(summary)
    if len(review) > _RESPONSE_LIMIT:
        review = review[: _RESPONSE_LIMIT - 3] + "..."
    return review


class _Server(ThreadingHTTPServer):
    """HTTP server with sane defaults for ephemeral port binding."""

    allow_reuse_address = True
    daemon_threads = True


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
        endpoint = self.path.rstrip("/") or "/"
        if endpoint in {"/health", "/v1/health"}:
            self._send_json({"status": "ok"})
        elif endpoint == "/v1/models":
            self._send_json({"data": [{"id": _MODEL_NAME}]})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        endpoint = self.path.rstrip("/")
        if endpoint not in {"/v1/chat/completions", "/v1/completions"}:
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

        if endpoint == "/v1/chat/completions":
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
        else:
            prompt = payload.get("prompt", "")
            if prompt is None:
                prompt = ""
            if not isinstance(prompt, str):
                self.send_error(HTTPStatus.BAD_REQUEST, "prompt must be a string")
                return
            stats = _analyze_prompt(prompt)
            review = _build_review(stats)
            response = {
                "object": "text_completion",
                "model": payload.get("model", _MODEL_NAME),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "text": review,
                    }
                ],
            }
        self._send_json(response)

    def log_message(self, fmt: str, *args) -> None:  # pragma: no cover - reduce noise
        # Silence default request logging to keep CI output tidy.
        pass


def _install_signal_handlers(server: ThreadingHTTPServer) -> None:
    """Configure basic signal handlers for long-running CI jobs."""

    if not hasattr(signal, "signal"):
        return

    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def _shutdown(_signum: int, _frame: Any | None) -> None:
        # ``shutdown`` is idempotent and safe to call multiple times.
        server.shutdown()

    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _shutdown)


def _write_port_file(path: Path | None, port: int) -> None:
    """Persist the selected ``port`` so other steps can reuse it."""

    if path is None:
        return

    try:
        # ``Path.write_text`` truncates the file before writing, so we only
        # need to ensure the persisted value is newline-terminated.  The GitHub
        # workflow reads the file with ``cat`` and a trailing newline keeps the
        # subsequent shell prompt tidy while remaining backwards compatible
        # with previous behaviour.
        write_secure_text(path, f"{port}\n")
    except OSError as exc:  # pragma: no cover - best-effort logging
        print(
            f"::warning::Не удалось записать номер порта в {path}: {exc}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mock GPT-OSS server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--port-file", type=Path)
    args = parser.parse_args()

    server = _Server((args.host, args.port), _RequestHandler)
    _write_port_file(args.port_file, server.server_address[1])
    _install_signal_handlers(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
