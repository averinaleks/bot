import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import request as urllib_request

import pytest

from scripts import gptoss_mock_server
from scripts import run_gptoss_review


@pytest.fixture()
def gptoss_server_port():
    server = ThreadingHTTPServer(("127.0.0.1", 0), gptoss_mock_server._RequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    for _ in range(50):
        try:
            with urllib_request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=0.2
            ):
                break
        except Exception:
            time.sleep(0.05)
    else:  # pragma: no cover - extremely unlikely when server starts correctly
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()
        pytest.fail("mock GPT-OSS server did not start in time")

    try:
        yield port
    finally:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()


def _write_diff(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "diff --git a/test.py b/test.py",
                "+++ b/test.py",
                "@@",
                "+print('debug')",
            ]
        ),
        encoding="utf-8",
    )


def test_generate_review_success(tmp_path: Path, gptoss_server_port: int) -> None:
    diff_path = tmp_path / "diff.patch"
    _write_diff(diff_path)
    api_url = f"http://127.0.0.1:{gptoss_server_port}/v1/chat/completions"

    result = run_gptoss_review.generate_review(diff_path, "dummy", api_url)

    assert result.has_content
    assert "Автоматический обзор" in result.review


def test_main_writes_review_and_output(tmp_path: Path, gptoss_server_port: int, monkeypatch):
    diff_path = tmp_path / "diff.patch"
    review_path = tmp_path / "review.md"
    github_output = tmp_path / "gh_output.txt"
    _write_diff(diff_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    exit_code = run_gptoss_review.main(
        [
            "--diff",
            str(diff_path),
            "--output",
            str(review_path),
            "--model",
            "dummy-model",
            "--api-url",
            f"http://127.0.0.1:{gptoss_server_port}/v1/chat/completions",
        ]
    )

    assert exit_code == 0
    assert review_path.read_text(encoding="utf-8")
    assert "has_content=true" in github_output.read_text(encoding="utf-8")


def test_main_handles_missing_diff(tmp_path: Path, monkeypatch) -> None:
    diff_path = tmp_path / "missing.patch"
    review_path = tmp_path / "review.md"
    github_output = tmp_path / "gh_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    exit_code = run_gptoss_review.main(
        [
            "--diff",
            str(diff_path),
            "--output",
            str(review_path),
            "--api-url",
            "http://127.0.0.1:9999/v1/chat/completions",
        ]
    )

    assert exit_code == 0
    assert not review_path.exists()
    assert "has_content=false" in github_output.read_text(encoding="utf-8")


def test_extract_review_fallback_to_text() -> None:
    response = {"choices": [{"text": " result "}]}
    assert run_gptoss_review._extract_review(response) == "result"


def test_extract_review_handles_unexpected_payload() -> None:
    assert run_gptoss_review._extract_review(["unexpected"]) == ""
    assert run_gptoss_review._extract_review({"choices": "invalid"}) == ""


def test_main_handles_unexpected_exception(monkeypatch, tmp_path):
    diff_path = tmp_path / "diff.patch"
    diff_path.write_text("dummy", encoding="utf-8")
    github_output = tmp_path / "gh_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    def _boom(*_args, **_kwargs):  # pragma: no cover - executed via test
        raise ValueError("boom")

    monkeypatch.setattr(run_gptoss_review, "generate_review", _boom)

    exit_code = run_gptoss_review.main(["--diff", str(diff_path)])

    assert exit_code == 0
    assert "has_content=false" in github_output.read_text(encoding="utf-8")
