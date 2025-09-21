from __future__ import annotations

from pathlib import Path

from scripts.submit_dependency_snapshot import (
    _auth_schemes,
    _job_metadata,
    _parse_requirements,
)


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


def test_auth_schemes_prefers_bearer_for_github_tokens(monkeypatch) -> None:
    monkeypatch.delenv("DEPENDENCY_SNAPSHOT_AUTH_SCHEME", raising=False)
    assert _auth_schemes("ghs_example") == ["Bearer", "token"]


def test_auth_schemes_allows_override(monkeypatch) -> None:
    monkeypatch.setenv("DEPENDENCY_SNAPSHOT_AUTH_SCHEME", "token")
    assert _auth_schemes("anything") == ["token"]


def test_job_metadata_adds_html_url_when_run_id_numeric(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://example.com")
    job = _job_metadata("owner/repo", "12345", "corr")
    assert job["html_url"] == "https://example.com/owner/repo/actions/runs/12345"
    job_no_url = _job_metadata("owner/repo", "run-1", "corr")
    assert "html_url" not in job_no_url
