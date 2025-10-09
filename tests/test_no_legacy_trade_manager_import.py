"""Ensure that the legacy ``import trade_manager`` alias is not reintroduced."""

from __future__ import annotations

from pathlib import Path


_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}


def _tracked_python_files(root: Path) -> list[Path]:
    """Return Python sources rooted under *root* excluding transient folders."""

    candidates: list[Path] = []
    for path in root.rglob("*.py"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            # Files outside the repository root are irrelevant.
            continue
        if any(part in _SKIP_DIRS for part in relative.parts):
            continue
        if not path.is_file():
            continue
        candidates.append(path)
    return candidates


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
