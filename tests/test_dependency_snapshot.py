from __future__ import annotations

from pathlib import Path

import pytest

from scripts import submit_dependency_snapshot as snapshot


@pytest.mark.parametrize(
    "filename",
    [
        "requirements.txt",
        "requirements.in",
        "requirements.out",
    ],
)
def test_build_manifests_includes_supported_patterns(tmp_path: Path, filename: str) -> None:
    requirement_file = tmp_path / filename
    requirement_file.write_text("requests==2.32.3\n")

    manifests = snapshot._build_manifests(tmp_path)

    assert set(manifests.keys()) == {filename}

    manifest_data = manifests[filename]
    assert manifest_data["file"]["source_location"] == filename
    assert manifest_data["resolved"]["requests"]["package_url"] == "pkg:pypi/requests@2.32.3"


def test_parse_requirements_skips_blocklisted_packages(tmp_path: Path) -> None:
    requirement_file = tmp_path / "requirements.txt"
    requirement_file.write_text("ccxtpro==1.0.1\nhttpx==0.27.2\n")

    resolved = snapshot._parse_requirements(requirement_file)

    assert "ccxtpro" not in resolved
    assert "httpx" in resolved
    assert resolved["httpx"]["package_url"] == "pkg:pypi/httpx@0.27.2"


def test_parse_requirements_encodes_versions_for_purl(tmp_path: Path) -> None:
    requirement_file = tmp_path / "requirements.txt"
    requirement_file.write_text("torch==2.8.0+cpu\n")

    resolved = snapshot._parse_requirements(requirement_file)

    assert resolved["torch"]["package_url"] == "pkg:pypi/torch@2.8.0%2Bcpu"


def test_submit_dependency_snapshot_skips_when_env_missing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "Missing required environment variable: GITHUB_REPOSITORY" in captured.err
    assert (
        "Dependency snapshot submission skipped из-за отсутствия переменных окружения." in captured.err
    )


def test_submit_dependency_snapshot_handles_manifest_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_dummy")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")

    def boom(_: Path) -> dict[str, snapshot.Manifest]:
        raise RuntimeError("manifest failure")

    monkeypatch.setattr(snapshot, "_build_manifests", boom)

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert (
        "Dependency snapshot submission skipped из-за непредвиденной ошибки." in captured.err
    )
    assert "manifest failure" in captured.err


def test_submit_dependency_snapshot_reports_submission_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_RUN_ID", "123456")
    monkeypatch.setenv("GITHUB_WORKFLOW", "dependency-graph")
    monkeypatch.setenv("GITHUB_JOB", "submit")

    manifest: snapshot.Manifest = {
        "name": "requirements.txt",
        "file": {"source_location": "requirements.txt"},
        "resolved": {
            "pkg:pypi/httpx@0.27.2": {
                "package_url": "pkg:pypi/httpx@0.27.2",
                "relationship": "direct",
                "scope": "runtime",
                "dependencies": [],
            }
        },
    }
    monkeypatch.setattr(snapshot, "_build_manifests", lambda _: {"requirements.txt": manifest})

    def raise_submission_error(*_: object, **__: object) -> None:
        raise snapshot.DependencySubmissionError(400, "Bad Request")

    monkeypatch.setattr(snapshot, "_submit_with_headers", raise_submission_error)

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "Dependency snapshot submission skipped из-за ошибки GitHub API." in captured.err
    assert "HTTP 400" in captured.err
