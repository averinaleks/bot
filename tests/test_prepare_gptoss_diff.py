from __future__ import annotations

import json
import subprocess  # nosec B404
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest

from scripts import prepare_gptoss_diff


@pytest.fixture()
def temp_repo(tmp_path: Path) -> Iterator[Path]:
    remote = tmp_path / "remote.git"
    # Bandit: the fixture initialises a temporary git repository with trusted commands.
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)  # nosec

    workdir = tmp_path / "work"
    subprocess.run(["git", "clone", str(remote), str(workdir)], check=True)  # nosec
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=workdir, check=True)  # nosec
    subprocess.run(["git", "config", "user.name", "CI"], cwd=workdir, check=True)  # nosec

    (workdir / "example.py").write_text("print('hi')\n", encoding="utf-8")
    subprocess.run(["git", "add", "example.py"], cwd=workdir, check=True)  # nosec
    subprocess.run(["git", "commit", "-m", "init"], cwd=workdir, check=True)  # nosec
    subprocess.run(["git", "branch", "-M", "main"], cwd=workdir, check=True)  # nosec
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=workdir, check=True)  # nosec

    subprocess.run(["git", "checkout", "-b", "feature"], cwd=workdir, check=True)  # nosec
    (workdir / "example.py").write_text("print('hi')\nprint('bye')\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "update"], cwd=workdir, check=True)  # nosec

    yield workdir


def test_compute_diff_returns_content(temp_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(temp_repo)
    # Bandit: git commands operate on the temporary repository created above.
    base_sha = (
        subprocess.run(["git", "rev-parse", "origin/main"], cwd=temp_repo, check=True, capture_output=True, text=True)  # nosec
        .stdout.strip()
    )

    result = prepare_gptoss_diff._compute_diff(base_sha, [":(glob)**/*.py"], truncate=10_000)

    assert result.has_diff is True
    assert "print('bye')" in result.content


def test_prepare_diff_reads_remote_metadata(monkeypatch: pytest.MonkeyPatch, temp_repo: Path) -> None:
    monkeypatch.chdir(temp_repo)

    pull_payload = {
        "base": {
            # Bandit: git invocations here also target the controlled temporary repository.
            "sha": subprocess.run(  # nosec
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
        subprocess.run(["git", "rev-parse", "origin/main"], cwd=temp_repo, check=True, capture_output=True, text=True)  # nosec
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
    monkeypatch.setattr(
        prepare_gptoss_diff, "_ensure_base_available", lambda ref, sha: None
    )
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


@pytest.mark.parametrize(
    "path",
    [
        "src/module.py",
        "nested/dir/file.txt",
        ":(glob)**/*.py",
    ],
)
def test_validate_path_argument_allows_safe_inputs(path: str) -> None:
    assert prepare_gptoss_diff._validate_path_argument(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "dir/../secret.txt",
        "..\\windows\\system32",
        "C:/absolute/path",
        "C\\windows\\system32",
        "\x00bad",
        ":(literal)danger",  # unsupported pathspec prefix
    ],
)
def test_validate_path_argument_rejects_unsafe_inputs(path: str) -> None:
    with pytest.raises(RuntimeError):
        prepare_gptoss_diff._validate_path_argument(path)


def test_ensure_base_available_skips_fetch_when_commit_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], bool]] = []

    def _fake_run_git(args, *, capture_output: bool = False):
        calls.append((tuple(args), capture_output))
        if args[1] == "cat-file":
            return prepare_gptoss_diff.GitCompletedProcess(tuple(args), 0)
        pytest.fail(f"unexpected git command: {args}")

    monkeypatch.setattr(prepare_gptoss_diff, "_run_git", _fake_run_git)

    prepare_gptoss_diff._ensure_base_available("main", "a" * 40)

    assert any(cmd[0][1] == "cat-file" for cmd in calls)
    assert not any(cmd[0][1] == "fetch" for cmd in calls)


def test_ensure_base_available_fetches_when_missing_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], bool]] = []

    def _fake_run_git(args, *, capture_output: bool = False):
        calls.append((tuple(args), capture_output))
        if args[1] == "cat-file":
            raise prepare_gptoss_diff.GitCommandError(1, args, None, None)
        if args[1] == "fetch":
            return prepare_gptoss_diff.GitCompletedProcess(tuple(args), 0)
        pytest.fail(f"unexpected git command: {args}")

    monkeypatch.setattr(prepare_gptoss_diff, "_run_git", _fake_run_git)

    prepare_gptoss_diff._ensure_base_available("main", "a" * 40)

    assert any(cmd[0][1] == "fetch" for cmd in calls)


def test_ensure_base_available_allows_fetch_failure_if_commit_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"cat": 0}

    def _fake_run_git(args, *, capture_output: bool = False):
        if args[1] == "cat-file":
            attempts["cat"] += 1
            if attempts["cat"] == 1:
                raise prepare_gptoss_diff.GitCommandError(1, args, None, None)
            return prepare_gptoss_diff.GitCompletedProcess(tuple(args), 0)
        if args[1] == "fetch":
            raise prepare_gptoss_diff.GitCommandError(128, args, None, None)
        pytest.fail(f"unexpected git command: {args}")

    monkeypatch.setattr(prepare_gptoss_diff, "_run_git", _fake_run_git)

    prepare_gptoss_diff._ensure_base_available("main", "a" * 40)

    assert attempts["cat"] == 2


def test_ensure_base_available_raises_if_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_git(args, *, capture_output: bool = False):
        if args[1] == "cat-file":
            raise prepare_gptoss_diff.GitCommandError(1, args, None, None)
        if args[1] == "fetch":
            raise prepare_gptoss_diff.GitCommandError(128, args, None, None)
        pytest.fail(f"unexpected git command: {args}")

    monkeypatch.setattr(prepare_gptoss_diff, "_run_git", _fake_run_git)

    with pytest.raises(RuntimeError):
        prepare_gptoss_diff._ensure_base_available("main", "a" * 40)


def test_compute_diff_rejects_negative_truncate(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_git(_args, *, capture_output: bool = False):
        _ = capture_output
        return prepare_gptoss_diff.GitCompletedProcess(tuple(_args), 0, stdout="diff")

    monkeypatch.setattr(prepare_gptoss_diff, "_run_git", _fake_run_git)

    with pytest.raises(RuntimeError):
        prepare_gptoss_diff._compute_diff("a" * 40, [":(glob)**/*.py"], truncate=-1)


def test_prepare_diff_validates_compute_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prepare_gptoss_diff,
        "_ensure_base_available",
        lambda ref, sha: None,
    )
    monkeypatch.setattr(prepare_gptoss_diff, "_validate_git_sha", lambda sha: "a" * 40)
    monkeypatch.setattr(prepare_gptoss_diff, "_compute_diff", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError):
        prepare_gptoss_diff.prepare_diff(
            "example/repo",
            "123",
            token=None,
            base_sha="a" * 40,
            base_ref="main",
        )

