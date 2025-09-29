from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_pr_status


def _build_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "state": "open",
        "head": {
            "sha": "deadbeef" * 5,
            "repo": {"full_name": "owner/repo"},
        },
    }
    payload.update(overrides)
    return payload


def test_evaluate_payload_happy_path() -> None:
    status = check_pr_status._evaluate_payload(_build_payload(), "owner/repo")
    assert status.skip is False
    assert status.head_sha == "deadbeef" * 5
    assert status.notices == []


def test_evaluate_payload_detects_closed_pr() -> None:
    payload = _build_payload(state="closed")
    status = check_pr_status._evaluate_payload(payload, "owner/repo")
    assert status.skip is True
    assert any("closed" in notice for notice in status.notices)


def test_evaluate_payload_detects_repo_mismatch() -> None:
    payload = _build_payload()
    status = check_pr_status._evaluate_payload(payload, "other/repo")
    assert status.skip is True
    assert any("other/repo" in notice for notice in status.notices)


def test_evaluate_payload_handles_missing_data() -> None:
    status = check_pr_status._evaluate_payload({}, "owner/repo")
    assert status.skip is True
    assert any("SHA" in notice for notice in status.notices)


def test_evaluate_payload_handles_unexpected_format() -> None:
    status = check_pr_status._evaluate_payload([], "owner/repo")  # type: ignore[arg-type]
    assert status.skip is True
    assert "неожиданный формат" in status.notices[0]


def test_main_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _build_payload()

    expected_token = "dummy" + "_value"

    def fake_fetch(url: str, token: str, timeout: float) -> dict[str, object]:
        assert "3163" in url
        assert token == expected_token
        assert timeout == 5.0
        return payload

    monkeypatch.setattr(check_pr_status, "_fetch_pull_request", fake_fetch)

    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    result = check_pr_status.main([
        "--repo",
        "owner/repo",
        "--pr-number",
        "3163",
        "--token",
        expected_token,
        "--timeout",
        "5.0",
    ])

    assert result == 0
    assert output_file.read_text(encoding="utf-8") == "skip=false\nhead_sha={}\n".format("deadbeef" * 5)


def test_main_handles_http_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_fetch(url: str, token: str, timeout: float) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(check_pr_status, "_fetch_pull_request", fake_fetch)
    monkeypatch.setenv("GITHUB_OUTPUT", "")

    result = check_pr_status.main([
        "--repo",
        "owner/repo",
        "--pr-number",
        "42",
        "--token",
        "",
    ])

    captured = capsys.readouterr()
    assert result == 0
    assert "::warning::boom" in captured.err


def test_cli_swallows_system_exit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(check_pr_status, "main", lambda argv=None: (_ for _ in ()).throw(SystemExit(3)))
    monkeypatch.setenv("GITHUB_OUTPUT", "")

    result = check_pr_status.cli([])
    captured = capsys.readouterr()
    assert result == 0
    assert "кодом 3" in captured.err


def test_cli_logs_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _raise(*_args, **_kwargs):
        raise KeyboardInterrupt("boom")

    monkeypatch.setattr(check_pr_status, "main", _raise)
    monkeypatch.setenv("GITHUB_OUTPUT", "")

    result = check_pr_status.cli([])
    captured = capsys.readouterr()
    assert result == 0
    assert "::warning::Критическое исключение" in captured.err

