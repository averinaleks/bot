"""Tests for the Semgrep SARIF helper script used in CI workflows."""

from __future__ import annotations

import json
import os
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pytest

from scripts import ensure_semgrep_sarif as semgrep_helper


@dataclass
class _CLIResult:
    """Container mirroring the subset of ``subprocess.CompletedProcess`` we assert on."""

    returncode: int
    stdout: str
    stderr: str


def _run_cli(monkeypatch: pytest.MonkeyPatch, cwd: Path, *args: str) -> _CLIResult:
    """Execute :func:`scripts.ensure_semgrep_sarif.main` as if invoked via CLI."""

    monkeypatch.chdir(cwd)
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        returncode = semgrep_helper.main(list(args))
    return _CLIResult(returncode=returncode, stdout=stdout_buffer.getvalue(), stderr=stderr_buffer.getvalue())


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


def test_cli_supports_symlink_github_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    target_output = workspace / "outputs.txt"
    symlink_path = workspace / "outputs-link.txt"
    symlink_path.symlink_to(target_output)

    result = _run_cli(
        monkeypatch,
        workspace,
        "--github-output",
        str(symlink_path.relative_to(workspace)),
    )

    assert result.returncode == 0
    assert target_output.exists()
    assert "upload=false" in target_output.read_text(encoding="utf-8")


def test_cli_execution_from_arbitrary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    github_output = workspace / "outputs.txt"

    result = _run_cli(
        monkeypatch,
        workspace,
        "--github-output",
        str(github_output.relative_to(workspace)),
    )

    assert "No Semgrep findings detected" in result.stdout
    assert github_output.exists()
    assert "upload=false" in github_output.read_text(encoding="utf-8")

    sarif_path = workspace / "semgrep.sarif"
    assert sarif_path.exists()

    sarif_data = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif_data["runs"][0]["results"] == []


def test_cli_creates_parent_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    github_output = workspace / "nested" / "outputs.txt"

    result = _run_cli(
        monkeypatch,
        workspace,
        "--github-output",
        str(github_output.relative_to(workspace)),
    )

    assert result.returncode == 0
    assert github_output.exists()
    assert "upload=false" in github_output.read_text(encoding="utf-8")


def test_cli_skips_empty_github_output_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = _run_cli(monkeypatch, workspace, "--github-output", "")

    assert result.returncode == 0
    assert "skipping output export" in result.stderr.lower()
    assert "No Semgrep findings detected" in result.stdout

    sarif_path = workspace / "semgrep.sarif"
    assert sarif_path.exists()
    assert json.loads(sarif_path.read_text(encoding="utf-8"))["runs"][0]["results"] == []


def test_cli_skips_directory_github_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = _run_cli(
        monkeypatch,
        workspace,
        "--github-output",
        str(workspace),
    )

    assert result.returncode == 0
    assert "skipping output export" in result.stderr.lower()
    assert "No Semgrep findings detected" in result.stdout
