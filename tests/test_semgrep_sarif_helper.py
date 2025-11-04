"""Tests for the Semgrep SARIF helper script used in CI workflows."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts import ensure_semgrep_sarif as semgrep_helper


# Bandit note: the helper interacts exclusively with repository-controlled
# executables.  The thin wrapper consolidates subprocess invocations so that the
# security rationale sits alongside the suppression comment.
def _run_helper(
    command: Sequence[str | os.PathLike[str]],
    **kwargs,
) -> subprocess.CompletedProcess[str]:
    converted = [str(part) for part in command]
    return subprocess.run(converted, **kwargs)  # nosec B603


def test_ensure_semgrep_sarif_creates_empty_report(tmp_path: Path) -> None:
    sarif_path = tmp_path / "report.sarif"

    created = semgrep_helper.ensure_semgrep_sarif(sarif_path)

    assert created == sarif_path
    assert sarif_path.exists()

    data = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert data["runs"]
    assert data["runs"][0]["results"] == []


def test_write_github_output_appends_expected_values(tmp_path: Path) -> None:
    output_path = tmp_path / "github_output.txt"

    semgrep_helper.write_github_output(
        output_path,
        upload=True,
        findings=2,
        sarif_path=Path("semgrep.sarif"),
    )

    semgrep_helper.write_github_output(
        output_path,
        upload=False,
        findings=0,
        sarif_path=Path("semgrep.sarif"),
    )

    content = output_path.read_text(encoding="utf-8").splitlines()
    assert content[0] == "upload=true"
    assert content[1] == "result_count=2"
    assert content[2] == "sarif_path=semgrep.sarif"
    assert content[3] == "upload=false"
    assert content[4] == "result_count=0"


def test_write_github_output_supports_named_pipes(tmp_path: Path) -> None:
    fifo_path = tmp_path / "github_output_fifo"
    os.mkfifo(fifo_path)

    received: list[str] = []

    def _reader() -> None:
        with fifo_path.open("r", encoding="utf-8") as handle:
            received.append(handle.read())

    reader = threading.Thread(target=_reader)
    reader.start()

    try:
        semgrep_helper.write_github_output(
            fifo_path,
            upload=True,
            findings=3,
            sarif_path=Path("semgrep.sarif"),
        )
    finally:
        reader.join(timeout=5)
    assert not reader.is_alive()

    assert received
    assert "upload=true" in received[0]
    assert "result_count=3" in received[0]


def test_write_github_output_supports_file_descriptors(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    received: list[str] = []

    def _reader() -> None:
        with os.fdopen(read_fd, "r", encoding="utf-8") as handle:
            received.append(handle.read())

    reader = threading.Thread(target=_reader)
    reader.start()

    try:
        semgrep_helper.write_github_output(
            write_fd,
            upload=True,
            findings=5,
            sarif_path=Path("semgrep.sarif"),
        )
    finally:
        os.close(write_fd)
        reader.join(timeout=5)

    assert not reader.is_alive()
    assert received
    assert "upload=true" in received[0]
    assert "result_count=5" in received[0]


def test_write_github_output_closes_acquired_descriptor(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    received: list[str] = []

    def _reader() -> None:
        with os.fdopen(read_fd, "r", encoding="utf-8") as handle:
            received.append(handle.read())

    reader = threading.Thread(target=_reader)
    reader.start()

    target = semgrep_helper.GithubOutputTarget(write_fd, close_after=True)

    try:
        semgrep_helper.write_github_output(
            target,
            upload=False,
            findings=0,
            sarif_path=Path("semgrep.sarif"),
        )
    finally:
        reader.join(timeout=5)

    assert not reader.is_alive()
    assert received
    assert "upload=false" in received[0]

    with pytest.raises(OSError):
        os.close(write_fd)


def test_cli_supports_symlink_github_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    target_output = workspace / "outputs.txt"
    symlink_path = workspace / "outputs-link.txt"
    symlink_path.symlink_to(target_output)

    script_path = repo_root / "scripts" / "ensure_semgrep_sarif.py"

    result = _run_helper(
        [sys.executable, script_path, "--github-output", symlink_path],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert target_output.exists()
    assert "upload=false" in target_output.read_text(encoding="utf-8")


def test_normalize_supports_pipe_style_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    read_fd, write_fd = os.pipe()
    github_output = workspace / "outputs-link.txt"
    github_output.symlink_to("pipe:[12345]")

    monkeypatch.setattr(
        semgrep_helper,
        "_open_descriptor",
        lambda path: write_fd,
    )

    target = semgrep_helper._normalize_github_output(github_output)
    assert isinstance(target, semgrep_helper.GithubOutputTarget)
    assert target.handle == write_fd
    assert target.close_after is True

    os.close(write_fd)
    os.close(read_fd)


def test_cli_supports_fd_github_output(tmp_path: Path) -> None:
    if sys.platform == "win32":  # pragma: no cover - Windows lacks pass_fds support
        pytest.skip("File descriptor propagation is not supported on Windows runners")

    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    read_fd, write_fd = os.pipe()
    received: list[str] = []

    def _reader() -> None:
        with os.fdopen(read_fd, "r", encoding="utf-8") as handle:
            received.append(handle.read())

    reader = threading.Thread(target=_reader)
    reader.start()

    script_path = repo_root / "scripts" / "ensure_semgrep_sarif.py"

    try:
        result = _run_helper(
            [sys.executable, script_path, "--github-output", f"fd:{write_fd}"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            pass_fds=(write_fd,),
        )
    finally:
        os.close(write_fd)
        reader.join(timeout=5)

    assert result.returncode == 0
    assert not reader.is_alive()
    assert received
    assert "upload=false" in received[0]
    assert (workspace / "semgrep.sarif").exists()


@pytest.mark.skipif(
    sys.platform == "win32",  # pragma: no cover - Windows lacks pass_fds support
    reason="File descriptor propagation is not supported on Windows runners",
)
def test_cli_supports_proc_fd_path(tmp_path: Path) -> None:
    if not Path("/proc/self/fd").exists():  # pragma: no cover - non-Linux environments
        pytest.skip("/proc/self/fd is not available on this platform")

    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    read_fd, write_fd = os.pipe()
    received: list[str] = []

    def _reader() -> None:
        with os.fdopen(read_fd, "r", encoding="utf-8") as handle:
            received.append(handle.read())

    reader = threading.Thread(target=_reader)
    reader.start()

    script_path = repo_root / "scripts" / "ensure_semgrep_sarif.py"
    proc_fd_path = Path(f"/proc/self/fd/{write_fd}")

    try:
        result = _run_helper(
            [
                sys.executable,
                script_path,
                "--github-output",
                proc_fd_path,
            ],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            pass_fds=(write_fd,),
        )
    finally:
        os.close(write_fd)
        reader.join(timeout=5)

    assert result.returncode == 0
    assert not reader.is_alive()
    assert received
    assert "upload=false" in received[0]
    assert (workspace / "semgrep.sarif").exists()


@pytest.mark.skipif(
    sys.platform == "win32",  # pragma: no cover - Windows lacks pass_fds support
    reason="File descriptor propagation is not supported on Windows runners",
)
def test_cli_supports_symlinked_proc_fd(tmp_path: Path) -> None:
    if not Path("/proc/self/fd").exists():  # pragma: no cover - non-Linux environments
        pytest.skip("/proc/self/fd is not available on this platform")

    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    read_fd, write_fd = os.pipe()
    received: list[str] = []

    def _reader() -> None:
        with os.fdopen(read_fd, "r", encoding="utf-8") as handle:
            received.append(handle.read())

    reader = threading.Thread(target=_reader)
    reader.start()

    proc_fd_path = Path(f"/proc/self/fd/{write_fd}")
    github_output = workspace / "gh-output"
    github_output.symlink_to(proc_fd_path)

    script_path = repo_root / "scripts" / "ensure_semgrep_sarif.py"

    try:
        result = _run_helper(
            [
                sys.executable,
                script_path,
                "--github-output",
                github_output,
            ],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            pass_fds=(write_fd,),
        )
    finally:
        os.close(write_fd)
        reader.join(timeout=5)

    assert result.returncode == 0
    assert not reader.is_alive()
    assert received
    assert "upload=false" in received[0]
    assert (workspace / "semgrep.sarif").exists()


def test_cli_execution_from_arbitrary_directory(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    github_output = workspace / "outputs.txt"
    script_path = repo_root / "scripts" / "ensure_semgrep_sarif.py"

    result = _run_helper(
        [sys.executable, script_path, "--github-output", github_output],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "No Semgrep findings detected" in result.stdout
    assert github_output.exists()
    assert "upload=false" in github_output.read_text(encoding="utf-8")

    sarif_path = workspace / "semgrep.sarif"
    assert sarif_path.exists()

    sarif_data = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif_data["runs"][0]["results"] == []


def test_cli_creates_parent_directories(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    github_output = workspace / "nested" / "outputs.txt"
    script_path = repo_root / "scripts" / "ensure_semgrep_sarif.py"

    result = _run_helper(
        [sys.executable, script_path, "--github-output", github_output],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert github_output.exists()
    assert "upload=false" in github_output.read_text(encoding="utf-8")


def test_cli_skips_empty_github_output_value(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    script_path = repo_root / "scripts" / "ensure_semgrep_sarif.py"

    result = _run_helper(
        [sys.executable, script_path, "--github-output", ""],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "skipping output export" in result.stderr.lower()
    assert "No Semgrep findings detected" in result.stdout

    sarif_path = workspace / "semgrep.sarif"
    assert sarif_path.exists()
    assert json.loads(sarif_path.read_text(encoding="utf-8"))["runs"][0]["results"] == []


def test_cli_skips_directory_github_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    script_path = repo_root / "scripts" / "ensure_semgrep_sarif.py"

    result = _run_helper(
        [sys.executable, script_path, "--github-output", workspace],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "skipping output export" in result.stderr.lower()
    assert "No Semgrep findings detected" in result.stdout


@pytest.mark.skipif(
    not Path("/dev/null").exists(),
    reason="Character devices are not available on this platform",
)
def test_cli_supports_character_device_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    script_path = repo_root / "scripts" / "ensure_semgrep_sarif.py"

    result = _run_helper(
        [sys.executable, script_path, "--github-output", "/dev/null"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "No Semgrep findings detected" in result.stdout
