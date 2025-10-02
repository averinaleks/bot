"""Ensure that the legacy ``import trade_manager`` alias is not reintroduced."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404: используется только для чтения списка файлов через git
from pathlib import Path

import pytest


GIT_EXECUTABLE = shutil.which("git")

if GIT_EXECUTABLE is None:  # pragma: no cover - защитный сценарий окружения
    pytest.skip("Для теста требуется установленный git", allow_module_level=True)


def _tracked_python_files(root: Path) -> list[Path]:
    command = [GIT_EXECUTABLE, "ls-files", "--", "*.py"]
    result = subprocess.run(  # nosec B603: команда git формируется из фиксированных аргументов
        command,
        check=True,
        cwd=root,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [(root / line.strip()) for line in result.stdout.splitlines() if line.strip()]


def test_no_legacy_trade_manager_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for file_path in _tracked_python_files(repo_root):
        if not file_path.exists():
            # A file can be scheduled for deletion in the working tree; skip it.
            continue
        for lineno, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("import trade_manager"):
                offenders.append(f"{file_path.relative_to(repo_root)}:{lineno}")

    assert not offenders, "Legacy TradeManager imports detected: " + ", ".join(offenders)
