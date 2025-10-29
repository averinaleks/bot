"""Tests for the Semgrep SARIF helper script used in CI workflows."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import ensure_semgrep_sarif as semgrep_helper


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


def test_cli_execution_from_arbitrary_directory(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    github_output = workspace / "outputs.txt"
    script_path = repo_root / "scripts" / "ensure_semgrep_sarif.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--github-output", str(github_output)],
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

    result = subprocess.run(
        [sys.executable, str(script_path), "--github-output", str(github_output)],
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

    result = subprocess.run(
        [sys.executable, str(script_path), "--github-output", ""],
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

    result = subprocess.run(
        [sys.executable, str(script_path), "--github-output", str(workspace)],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "skipping output export" in result.stderr.lower()
    assert "No Semgrep findings detected" in result.stdout
