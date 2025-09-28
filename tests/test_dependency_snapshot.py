from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path

import pytest

from scripts import submit_dependency_snapshot as snapshot


httpx = pytest.importorskip("httpx")

try:
    HTTPX_VERSION = httpx.__version__
except AttributeError:  # pragma: no cover - fallback for older or metadata-less releases
    try:
        HTTPX_VERSION = metadata.version("httpx")
    except metadata.PackageNotFoundError:  # pragma: no cover - skip when dist metadata absent
        pytest.skip(
            "httpx distribution metadata is unavailable",
            allow_module_level=True,
        )

HTTPX_REQUIREMENT = f"httpx=={HTTPX_VERSION}"
HTTPX_PURL = f"pkg:pypi/httpx@{snapshot._encode_version_for_purl(HTTPX_VERSION)}"


@pytest.mark.parametrize(
    "filename",
    [
        "requirements.txt",
        "requirements.in",
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
    requirement_file.write_text(f"ccxtpro==1.0.1\n{HTTPX_REQUIREMENT}\n")

    resolved = snapshot._parse_requirements(requirement_file)

    assert "ccxtpro" not in resolved
    assert "httpx" in resolved
    assert resolved["httpx"]["package_url"] == HTTPX_PURL
    assert resolved[HTTPX_PURL]["package_url"] == HTTPX_PURL


def test_parse_requirements_encodes_versions_for_purl(tmp_path: Path) -> None:
    requirement_file = tmp_path / "requirements.txt"
    requirement_file.write_text("torch==2.8.0+cpu\n")

    resolved = snapshot._parse_requirements(requirement_file)

    assert resolved["torch"]["package_url"] == "pkg:pypi/torch@2.8.0%2Bcpu"


def test_parse_requirements_handles_decode_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "requirements.txt"
    manifest.write_bytes(b"\xff\xfeinvalid")

    resolved = snapshot._parse_requirements(manifest)

    assert not resolved

    captured = capsys.readouterr()
    assert "encoding error" in captured.err
    assert "requirements.txt" in captured.err


def test_parse_requirements_handles_filesystem_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "requirements.txt"
    manifest.write_text("requests==2.32.3\n")

    original_read_text = Path.read_text

    def failing_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == manifest:
            raise OSError("Permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", failing_read_text)

    resolved = snapshot._parse_requirements(manifest)

    assert not resolved

    captured = capsys.readouterr()
    assert "filesystem error" in captured.err
    assert "requirements.txt" in captured.err


def test_submit_dependency_snapshot_skips_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "Missing required environment variable: GITHUB_REPOSITORY" in captured.err
    expected_message = (
        "Dependency snapshot submission skipped "
        "из-за отсутствия переменных окружения."
    )
    assert expected_message in captured.err


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
    expected_message = (
        "Dependency snapshot submission skipped "
        "из-за непредвиденной ошибки."
    )
    assert expected_message in captured.err
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
            "httpx": {
                "package_url": HTTPX_PURL,
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


def test_submit_dependency_snapshot_reports_missing_requests(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")

    manifest: snapshot.Manifest = {
        "name": "requirements.txt",
        "file": {"source_location": "requirements.txt"},
        "resolved": {},
    }
    monkeypatch.setattr(snapshot, "_build_manifests", lambda _: {"requirements.txt": manifest})

    error = snapshot.DependencySubmissionError(
        None,
        (
            "Dependency snapshot submission requires the 'requests' package. "
            "Install it with 'pip install requests'."
        ),
    )

    def raise_missing_dependency(*_: object, **__: object) -> None:
        raise error

    monkeypatch.setattr(snapshot, "_submit_with_headers", raise_missing_dependency)

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "Dependency snapshot submission skipped из-за сетевой ошибки." in captured.err
    assert "Dependency snapshot submission requires the 'requests' package" in captured.err
    assert "Install it with 'pip install requests'." in captured.err


def test_submit_dependency_snapshot_uses_repository_dispatch_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "client_payload": {
                    "sha": "cafebabe",
                    "ref": "refs/heads/auto",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "repository_dispatch")
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "456")
    monkeypatch.setenv("GITHUB_WORKFLOW", "dependency-graph")
    monkeypatch.setenv("GITHUB_JOB", "submit")

    manifest: snapshot.Manifest = {
        "name": "requirements.txt",
        "file": {"source_location": "requirements.txt"},
        "resolved": {
            "httpx": {
                "package_url": HTTPX_PURL,
                "relationship": "direct",
                "scope": "runtime",
                "dependencies": [],
            }
        },
    }
    monkeypatch.setattr(snapshot, "_build_manifests", lambda _: {"requirements.txt": manifest})

    submitted: dict[str, object] = {}

    def capture_submission(url: str, body: bytes, headers: dict[str, str]) -> None:
        del url, headers
        submitted.update(json.loads(body))

    monkeypatch.setattr(snapshot, "_submit_with_headers", capture_submission)

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "Missing required environment variable" not in captured.err
    assert "Using repository_dispatch payload" in captured.out

    assert submitted["sha"] == "cafebabe"
    assert submitted["ref"] == "refs/heads/auto"


def test_submit_dependency_snapshot_uses_string_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")

    manifest: snapshot.Manifest = {
        "name": "requirements.txt",
        "file": {"source_location": "requirements.txt"},
        "resolved": {},
    }

    monkeypatch.setattr(
        snapshot,
        "_build_manifests",
        lambda _: {"requirements.txt": manifest},
    )

    captured_payload: dict[str, object] = {}

    def capture_submission(url: str, body: bytes, headers: dict[str, str]) -> None:
        del url, headers
        captured_payload.update(json.loads(body))

    monkeypatch.setattr(snapshot, "_submit_with_headers", capture_submission)

    snapshot.submit_dependency_snapshot()

    metadata = captured_payload.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata == {
        "run_attempt": "3",
        "job": "submit",
        "workflow": "dependency-graph",
    }


def test_build_manifests_skips_out_files_when_txt_present(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(f"{HTTPX_REQUIREMENT}\n")
    (tmp_path / "requirements.out").write_text(f"{HTTPX_REQUIREMENT}\n")

    manifests = snapshot._build_manifests(tmp_path)

    assert set(manifests.keys()) == {"requirements.txt"}


def test_build_manifests_skips_in_files_when_txt_present(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(f"{HTTPX_REQUIREMENT}\n")
    (tmp_path / "requirements.in").write_text(f"httpx>={HTTPX_VERSION}\n")

    manifests = snapshot._build_manifests(tmp_path)

    assert set(manifests.keys()) == {"requirements.txt"}
