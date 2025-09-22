from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest

from scripts import prepare_gptoss_diff


@pytest.fixture()
def temp_repo(tmp_path: Path) -> Iterator[Path]:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)

    workdir = tmp_path / "work"
    subprocess.run(["git", "clone", str(remote), str(workdir)], check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=workdir, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=workdir, check=True)

    (workdir / "example.py").write_text("print('hi')\n", encoding="utf-8")
    subprocess.run(["git", "add", "example.py"], cwd=workdir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=workdir, check=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=workdir, check=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=workdir, check=True)

    subprocess.run(["git", "checkout", "-b", "feature"], cwd=workdir, check=True)
    (workdir / "example.py").write_text("print('hi')\nprint('bye')\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "update"], cwd=workdir, check=True)

    yield workdir


def test_compute_diff_returns_content(temp_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(temp_repo)
    base_sha = (
        subprocess.run(["git", "rev-parse", "origin/main"], cwd=temp_repo, check=True, capture_output=True, text=True)
        .stdout.strip()
    )

    result = prepare_gptoss_diff._compute_diff(base_sha, [":(glob)**/*.py"], truncate=10_000)

    assert result.has_diff is True
    assert "print('bye')" in result.content


def test_prepare_diff_reads_remote_metadata(monkeypatch: pytest.MonkeyPatch, temp_repo: Path) -> None:
    monkeypatch.chdir(temp_repo)

    pull_payload = {
        "base": {
            "sha": subprocess.run(
                ["git", "rev-parse", "origin/main"],
                cwd=temp_repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "ref": "main",
        }
    }

    def _fake_request(url, headers, timeout):
        _ = (headers, timeout)
        payload = json.dumps(pull_payload).encode("utf-8")
        return 200, "OK", payload

    with mock.patch.object(prepare_gptoss_diff, "_perform_https_request", _fake_request):
        result = prepare_gptoss_diff.prepare_diff(
            "example/repo",
            "123",
            token=None,
            truncate=50_000,
        )

    assert result.has_diff is True
    assert "print('bye')" in result.content


def test_api_request_adds_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_headers: dict[str, str] = {}

    def _fake_request(url: str, headers: dict[str, str], timeout: float) -> tuple[int, str, bytes]:
        _ = (url, timeout)
        recorded_headers.update(headers)
        return 200, "OK", b"{}"

    monkeypatch.setattr(prepare_gptoss_diff, "_perform_https_request", _fake_request)

    prepare_gptoss_diff._api_request("https://api.github.com/repos/test/repo", token=None)

    assert recorded_headers.get("User-Agent")


def test_main_writes_outputs(monkeypatch: pytest.MonkeyPatch, temp_repo: Path, tmp_path: Path) -> None:
    monkeypatch.chdir(temp_repo)
    base_sha = (
        subprocess.run(["git", "rev-parse", "origin/main"], cwd=temp_repo, check=True, capture_output=True, text=True)
        .stdout.strip()
    )

    gh_output = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_output))

    exit_code = prepare_gptoss_diff.main(
        [
            "--repo",
            "example/repo",
            "--pr-number",
            "1",
            "--base-sha",
            base_sha,
            "--base-ref",
            "main",
            "--output",
            str(tmp_path / "diff.patch"),
        ]
    )

    assert exit_code == 0
    assert "has_diff=true" in gh_output.read_text(encoding="utf-8")


def test_main_handles_unknown_args(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_output = Path("nonexistent")
    monkeypatch.setenv("GITHUB_OUTPUT", str(dummy_output))
    exit_code = prepare_gptoss_diff.main(["--unknown"])
    assert exit_code == 0


def test_prepare_diff_rejects_invalid_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare_gptoss_diff, "_ensure_base_available", lambda ref: None)
    with pytest.raises(RuntimeError):
        prepare_gptoss_diff.prepare_diff(
            "example/repo",
            "123",
            token=None,
            base_sha="not-a-sha",
            base_ref="main",
        )


def test_prepare_diff_rejects_invalid_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare_gptoss_diff, "_compute_diff", lambda sha, paths, truncate: None)
    with pytest.raises(RuntimeError):
        prepare_gptoss_diff.prepare_diff(
            "example/repo",
            "123",
            token=None,
            base_sha="a" * 40,
            base_ref="bad ref",
        )

