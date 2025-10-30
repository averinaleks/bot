from __future__ import annotations

from pathlib import Path

from packaging.version import Version
import pytest

MIN_STARLETTE_VERSION = Version("0.49.1")
STARLETTE_PINNED_FILES = [
    Path("requirements.txt"),
    Path("requirements-core.txt"),
    Path("requirements-dev.txt"),
]
STARLETTE_CONSTRAINT_FILE = Path("requirements-core.in")


def _extract_exact_version(path: Path) -> Version:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("starlette=="):
            return Version(line.split("==", 1)[1])
    pytest.fail(f"'starlette==<version>' pin not found in {path}")


def _extract_minimum_constraint(path: Path) -> Version:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line.lower().startswith("starlette>="):
            return Version(line.split(">=", 1)[1].split()[0])
    pytest.fail(f"'starlette>=<version>' constraint not found in {path}")


def test_starlette_pins_are_not_vulnerable() -> None:
    for requirement_path in STARLETTE_PINNED_FILES:
        version = _extract_exact_version(requirement_path)
        assert version >= MIN_STARLETTE_VERSION, (
            f"{requirement_path} pins starlette {version}, expected >= {MIN_STARLETTE_VERSION}"
        )


def test_starlette_constraint_prevents_regressions() -> None:
    minimum = _extract_minimum_constraint(STARLETTE_CONSTRAINT_FILE)
    assert minimum >= MIN_STARLETTE_VERSION, (
        f"{STARLETTE_CONSTRAINT_FILE} constrains starlette to {minimum},"
        f" expected >= {MIN_STARLETTE_VERSION}"
    )
