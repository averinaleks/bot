from __future__ import annotations

from pathlib import Path

from scripts.submit_dependency_snapshot import _parse_requirements


def _write_requirements(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text(contents)
    return path


def test_parse_requirements_strips_inline_comments(tmp_path: Path) -> None:
    path = _write_requirements(
        tmp_path,
        """\
        package==1.2.3  # pinned for compatibility
        other==4.5.6
        """,
    )

    parsed = _parse_requirements(path)

    assert "pkg:pypi/package@1.2.3" in parsed
    assert "pkg:pypi/other@4.5.6" in parsed


def test_parse_requirements_handles_hash_block(tmp_path: Path) -> None:
    path = _write_requirements(
        tmp_path,
        """\
        sample==0.1.0 \
            --hash=sha256:deadbeef
        # a comment that should be ignored
        """,
    )

    parsed = _parse_requirements(path)

    assert list(parsed) == ["pkg:pypi/sample@0.1.0"]
