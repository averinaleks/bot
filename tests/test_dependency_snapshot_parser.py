from __future__ import annotations

import json
from pathlib import Path

from scripts.submit_dependency_snapshot import (
    _auth_schemes,
    _build_manifests,
    _job_metadata,
    _normalise_run_attempt,
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
    assert (
        parsed["pkg:pypi/package@1.2.3"]["package_url"]
        == "pkg:pypi/package@1.2.3"
    )
    assert "pkg:pypi/other@4.5.6" in parsed
    assert (
        parsed["pkg:pypi/other@4.5.6"]["package_url"]
        == "pkg:pypi/other@4.5.6"
    )


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


def test_normalise_run_attempt_handles_invalid_values(capsys) -> None:
    assert _normalise_run_attempt(None) == 1
    assert _normalise_run_attempt("3") == 3

    result_invalid = _normalise_run_attempt("not-a-number")
    assert result_invalid == 1
    captured = capsys.readouterr()
    assert "Invalid GITHUB_RUN_ATTEMPT" in captured.err

    result_negative = _normalise_run_attempt("0")
    assert result_negative == 1
    captured = capsys.readouterr()
    assert "must be >= 1" in captured.err


def test_build_manifests_supports_multiple_patterns(tmp_path: Path) -> None:
    files = {
        "requirements.txt": "package==1.0.0",
        "requirements-dev.in": "devpkg==2.0.0",
        "requirements-full.out": "fullpkg==3.0.0",
    }

    for name, contents in files.items():
        (tmp_path / name).write_text(contents)

    manifests = _build_manifests(tmp_path)

    assert set(manifests) == {
        "requirements.txt",
        "requirements-dev.in",
        "requirements-full.out",
    }

    for name, manifest in manifests.items():
        assert manifest["file"]["source_location"] == name


def test_job_metadata_adds_html_url_when_run_id_numeric(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://example.com")
    job = _job_metadata("owner/repo", "12345", "corr")
    assert job["html_url"] == "https://example.com/owner/repo/actions/runs/12345"
    job_no_url = _job_metadata("owner/repo", "run-1", "corr")
    assert "html_url" not in job_no_url


def test_submit_snapshot_skips_on_unauthorised_token(monkeypatch, capsys) -> None:
    from scripts import submit_dependency_snapshot as module

    manifests = {
        "requirements.txt": {
            "name": "requirements.txt",
            "file": {"source_location": "requirements.txt"},
            "resolved": {
                "pkg:pypi/sample@1.0.0": {
                    "package_url": "pkg:pypi/sample@1.0.0",
                    "relationship": "direct",
                    "scope": "runtime",
                    "dependencies": [],
                }
            },
        }
    }

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_dummy")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")

    monkeypatch.setattr(module, "_build_manifests", lambda root: manifests)

    def _raise(*args, **kwargs):
        raise module.DependencySubmissionError(401, "bad credentials")

    monkeypatch.setattr(module, "_submit_with_headers", _raise)

    module.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "ошибки авторизации" in captured.err


def test_submit_snapshot_skips_on_network_issue(monkeypatch, capsys) -> None:
    from scripts import submit_dependency_snapshot as module

    manifests = {
        "requirements.txt": {
            "name": "requirements.txt",
            "file": {"source_location": "requirements.txt"},
            "resolved": {
                "pkg:pypi/sample@1.0.0": {
                    "package_url": "pkg:pypi/sample@1.0.0",
                    "relationship": "direct",
                    "scope": "runtime",
                    "dependencies": [],
                }
            },
        }
    }

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_dummy")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")

    monkeypatch.setattr(module, "_build_manifests", lambda root: manifests)

    def _raise(*args, **kwargs):
        raise module.DependencySubmissionError(None, "network down")

    monkeypatch.setattr(module, "_submit_with_headers", _raise)

    module.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "сетевой ошибки" in captured.err


def test_submit_snapshot_skips_on_validation_issue(monkeypatch, capsys) -> None:
    from scripts import submit_dependency_snapshot as module

    manifests = {
        "requirements.txt": {
            "name": "requirements.txt",
            "file": {"source_location": "requirements.txt"},
            "resolved": {
                "pkg:pypi/sample@1.0.0": {
                    "package_url": "pkg:pypi/sample@1.0.0",
                    "relationship": "direct",
                    "scope": "runtime",
                    "dependencies": [],
                }
            },
        }
    }

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_dummy")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")

    monkeypatch.setattr(module, "_build_manifests", lambda root: manifests)

    def _raise(*args, **kwargs):
        raise module.DependencySubmissionError(422, "validation failed")

    monkeypatch.setattr(module, "_submit_with_headers", _raise)

    module.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "ошибки валидации" in captured.err


def test_submit_snapshot_uses_numeric_run_attempt(monkeypatch) -> None:
    from scripts import submit_dependency_snapshot as module

    manifests = {
        "requirements.txt": {
            "name": "requirements.txt",
            "file": {"source_location": "requirements.txt"},
            "resolved": {
                "pkg:pypi/sample@1.0.0": {
                    "package_url": "pkg:pypi/sample@1.0.0",
                    "relationship": "direct",
                    "scope": "runtime",
                    "dependencies": [],
                }
            },
        }
    }

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_dummy")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")

    monkeypatch.setattr(module, "_build_manifests", lambda root: manifests)

    captured_payload: dict | None = None

    def _capture(url: str, body: bytes, headers: dict[str, str]) -> None:
        nonlocal captured_payload
        captured_payload = json.loads(body.decode())

    monkeypatch.setattr(module, "_submit_with_headers", _capture)

    module.submit_dependency_snapshot()

    assert captured_payload is not None
    assert captured_payload["metadata"]["run_attempt"] == 3
