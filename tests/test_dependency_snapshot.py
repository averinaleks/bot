from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path

import pytest

from scripts import github_paths
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


def test_extract_payload_value_supports_after_oid() -> None:
    payload = {
        "afterOid": "abc",
        "afterSha": "def",
        "after_sha": "ghi",
    }

    result = snapshot._extract_payload_value(payload, *snapshot._PAYLOAD_SHA_KEYS)

    assert result == "abc"


def test_extract_payload_value_supports_after_commit_oid() -> None:
    payload = {"afterCommitOid": "cafebabe"}

    result = snapshot._extract_payload_value(payload, *snapshot._PAYLOAD_SHA_KEYS)

    assert result == "cafebabe"


def test_extract_payload_value_supports_camel_case_refs() -> None:
    payload = {
        "refName": "refs/heads/main",
        "branchName": "main",
        "ref": "",
    }

    result = snapshot._extract_payload_value(payload, *snapshot._PAYLOAD_REF_KEYS)

    assert result == "refs/heads/main"


def test_extract_workflow_run_ref_supports_branch_variants() -> None:
    payload = {"headBranchName": "release-candidate"}

    result = snapshot._extract_workflow_run_ref(payload)

    assert result == "refs/heads/release-candidate"


def test_extract_workflow_run_ref_supports_head_ref_name() -> None:
    payload = {"headRefName": "refs/heads/hotfix"}

    result = snapshot._extract_workflow_run_ref(payload)

    assert result == "refs/heads/hotfix"


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
    monkeypatch.setenv("GITHUB_WORKSPACE", str(Path.cwd()))
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


def test_submit_dependency_snapshot_handles_conflict_notice(
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

    class DummyResponse:
        status_code = 409
        reason = "Conflict"
        headers: dict[str, str] = {}
        text = "snapshot already exists"

        def close(self) -> None:
            return None

    post_calls: list[dict[str, object]] = []

    class DummySession:
        def __init__(self) -> None:
            self.trust_env = True
            self.proxies = {"http": "http://proxy"}
            self.verify = True

        def __enter__(self) -> "DummySession":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, *_args: object, **kwargs: object) -> DummyResponse:
            post_calls.append(dict(kwargs))
            return DummyResponse()

    monkeypatch.setattr(snapshot.requests, "Session", lambda: DummySession())

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert post_calls, "HTTP POST should be attempted"
    assert captured.err == ""
    assert "Dependency snapshot submission skipped" in captured.out
    assert "HTTP 409" in captured.out


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

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
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
    assert "Using event payload" in captured.out

    assert submitted["sha"] == "cafebabe"
    assert submitted["ref"] == "refs/heads/auto"


def test_submit_dependency_snapshot_uses_nested_dependency_graph_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "client_payload": {
                    "dependency_graph": {
                        "commit_oid": "feedfacecafebabe",
                        "ref": "feature/nested",
                        "token": "ghs_nested_token",
                        "repository": "averinaleks/bot",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "repository_dispatch")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
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

    captured_submission: dict[str, object] = {}

    def capture_submission(url: str, body: bytes, headers: dict[str, str]) -> None:
        captured_submission["url"] = url
        captured_submission["payload"] = json.loads(body)
        captured_submission["authorization"] = headers.get("Authorization")

    monkeypatch.setattr(snapshot, "_submit_with_headers", capture_submission)

    snapshot.submit_dependency_snapshot()

    captured_stream = capsys.readouterr()
    assert "Missing required environment variable" not in captured_stream.err
    assert "Using event payload" in captured_stream.out

    authorization = captured_submission.get("authorization")
    assert authorization == "Bearer ghs_nested_token"

    payload = captured_submission.get("payload")
    assert isinstance(payload, dict)
    assert payload["sha"] == "feedfacecafebabe"
    assert payload["ref"] == "refs/heads/feature/nested"

    url = captured_submission.get("url")
    assert isinstance(url, str)
    assert url.endswith("/repos/averinaleks/bot/dependency-graph/snapshots")


def test_submit_dependency_snapshot_uses_dependency_graph_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "action": "auto-submission",
                "commit_oid": "0123456789abcdef",
                "ref": "feature/auto",
                "base_ref": "refs/heads/main",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "dependency_graph")
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "654")
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
    assert "Using event payload" in captured.out

    assert submitted["sha"] == "0123456789abcdef"
    assert submitted["ref"] == "refs/heads/feature/auto"


def test_submit_dependency_snapshot_falls_back_to_dependency_graph_base_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "action": "auto-submission",
                "commit_oid": "feedfacecafebabe",
                "base_ref": "refs/heads/stable",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "dependency_graph")
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)

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
    assert submitted["sha"] == "feedfacecafebabe"
    assert submitted["ref"] == "refs/heads/stable"


def test_submit_dependency_snapshot_recovers_repository_from_display_login(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "dependency_graph": {
                    "repository": {
                        "name": "bot",
                        "owner": {"display_login": "averinaleks"},
                    },
                    "commit_oid": "cafebabefeedface",
                    "branch": "main",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "dependency_graph")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_WORKFLOW", "dependency-graph")
    monkeypatch.setenv("GITHUB_JOB", "submit")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)

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
    assert submitted["sha"] == "cafebabefeedface"
    assert submitted["ref"] == "refs/heads/main"
    assert submitted["job"]["html_url"] == "https://github.com/averinaleks/bot/actions/runs/42"


def test_submit_dependency_snapshot_prefers_after_commit_oid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "client_payload": {
                    "afterCommitOid": "feedfacecafebabe",
                    "ref": "refs/heads/payload-main",
                    "repository": "averinaleks/bot",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "repository_dispatch")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
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
    assert "Using event payload" in captured.out
    assert submitted["sha"] == "feedfacecafebabe"
    assert submitted["ref"] == "refs/heads/payload-main"


def test_submit_dependency_snapshot_recovers_repository_from_workflow_run_parts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "workflow_run": {
                    "head_sha": "feedface",
                    "head_branch": "auto-main",
                    "head_repository": {
                        "name": "bot",
                        "owner": {"login": "averinaleks"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "dynamic")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "321")
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
    submitted_url = ""

    def capture_submission(url: str, body: bytes, headers: dict[str, str]) -> None:
        del headers
        nonlocal submitted_url
        submitted_url = url
        submitted.update(json.loads(body))

    monkeypatch.setattr(snapshot, "_submit_with_headers", capture_submission)

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "Missing required environment variable" not in captured.err
    assert "Using event payload" in captured.out

    assert submitted_url.endswith(
        "/repos/averinaleks/bot/dependency-graph/snapshots"
    )
    assert submitted["sha"] == "feedface"
    assert submitted["ref"] == "refs/heads/auto-main"


def test_submit_dependency_snapshot_recovers_repository_from_combined_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "dependency_graph": {
                    "repository": {"nameWithOwner": "averinaleks/bot"},
                    "commit_oid": "facefeedcafebabe",
                    "ref_name": "main",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "repository_dispatch")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_WORKFLOW", "dependency-graph")
    monkeypatch.setenv("GITHUB_JOB", "submit")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)

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
    monkeypatch.setattr(
        snapshot, "_build_manifests", lambda _: {"requirements.txt": manifest}
    )

    submitted: dict[str, object] = {}

    def capture_submission(url: str, body: bytes, headers: dict[str, str]) -> None:
        del url, headers
        submitted.update(json.loads(body))

    monkeypatch.setattr(snapshot, "_submit_with_headers", capture_submission)

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "Missing required environment variable" not in captured.err
    assert "Using event payload" in captured.out

    assert submitted["sha"] == "facefeedcafebabe"
    assert submitted["ref"] == "refs/heads/main"
    assert submitted["job"]["html_url"] == (
        "https://github.com/averinaleks/bot/actions/runs/123"
    )


def test_submit_dependency_snapshot_recovers_owner_name_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "workflow_run": {
                    "head_sha": "abcdef123456",
                    "head_branch": "owner-name-branch",
                    "head_repository": {
                        "name": "bot",
                        "owner": {"name": "averinaleks"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "777")
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
    assert "Using event payload" in captured.out

    assert submitted["sha"] == "abcdef123456"
    assert submitted["ref"] == "refs/heads/owner-name-branch"
    assert submitted["job"]["html_url"].endswith("/actions/runs/777")
    assert submitted["job"]["correlator"].startswith("dependency-graph:submit")


def test_submit_dependency_snapshot_supports_commit_oid_and_ref_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "client_payload": {
                    "commit_oid": "deadbeef",
                    "ref_name": "feature",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "repository_dispatch")
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "654")
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
    assert "Using event payload" in captured.out

    assert submitted["sha"] == "deadbeef"
    assert submitted["ref"] == "refs/heads/feature"


def test_submit_dependency_snapshot_uses_client_workflow_run_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "client_payload": {
                    "workflow_run": {
                        "head_sha": "feedface",
                        "head_branch": "auto-main",
                        "head_repository": {"full_name": "averinaleks/bot"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "repository_dispatch")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "789")
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
    assert "Using event payload" in captured.out

    assert submitted["sha"] == "feedface"
    assert submitted["ref"] == "refs/heads/auto-main"
    assert submitted["job"]["html_url"].endswith("/actions/runs/789")


def test_submit_dependency_snapshot_prefers_payload_token(
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
                    "token": "ghs_payload_token",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "repository_dispatch")
    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "789")
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

    captured: dict[str, object] = {}

    def capture_submission(url: str, body: bytes, headers: dict[str, str]) -> None:
        del url
        captured["authorization"] = headers.get("Authorization")
        captured["payload"] = json.loads(body)

    monkeypatch.setattr(snapshot, "_submit_with_headers", capture_submission)

    snapshot.submit_dependency_snapshot()

    captured_stream = capsys.readouterr()
    assert "Missing required environment variable" not in captured_stream.err
    assert captured.get("authorization") == "Bearer ghs_payload_token"
    payload = captured.get("payload")
    assert isinstance(payload, dict)
    assert payload["sha"] == "cafebabe"
    assert payload["ref"] == "refs/heads/auto"


def test_submit_dependency_snapshot_uses_workflow_run_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "workflow_run": {
                    "head_sha": "feedface",
                    "head_branch": "auto",
                    "head_repository": {"full_name": "averinaleks/bot"},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "dynamic")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "987")
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

    captured: dict[str, object] = {}

    def capture_submission(url: str, body: bytes, headers: dict[str, str]) -> None:
        captured["url"] = url
        captured["payload"] = json.loads(body)
        captured["authorization"] = headers.get("Authorization")

    monkeypatch.setattr(snapshot, "_submit_with_headers", capture_submission)

    snapshot.submit_dependency_snapshot()

    captured_stream = capsys.readouterr()
    assert "Missing required environment variable" not in captured_stream.err
    assert "Using event payload" in captured_stream.out

    payload = captured.get("payload")
    assert isinstance(payload, dict)
    assert payload["sha"] == "feedface"
    assert payload["ref"] == "refs/heads/auto"

    url = captured.get("url")
    assert isinstance(url, str)
    assert url.endswith("/repos/averinaleks/bot/dependency-graph/snapshots")


def test_extract_payload_value_skips_null_like_strings() -> None:
    payload: dict[str, object] = {"sha": "null", "after": " cafebabe ", "ref": "None"}

    result = snapshot._extract_payload_value(payload, "sha", "after", "ref")

    assert result == "cafebabe"


def test_env_treats_null_strings_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", " null ")

    with pytest.raises(snapshot.MissingEnvironmentVariableError):
        snapshot._env("GITHUB_TOKEN")


def test_submit_dependency_snapshot_uses_root_payload_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps({"after": "cafebabe", "ref": "refs/heads/root"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "repository_dispatch")
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)

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

    assert captured_payload["sha"] == "cafebabe"
    assert captured_payload["ref"] == "refs/heads/root"


def test_load_event_payload_rejects_outside_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    unsafe_path = tmp_path / "outside.json"
    unsafe_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(unsafe_path))
    monkeypatch.setattr(
        github_paths,
        "allowed_github_directories",
        lambda: [workspace.resolve()],
    )

    payload = snapshot._load_event_payload()
    captured = capsys.readouterr()

    assert payload is None
    assert "outside trusted directories" in captured.err


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


def test_submit_dependency_snapshot_falls_back_to_git_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest: snapshot.Manifest = {
        "name": "requirements.txt",
        "file": {"source_location": "requirements.txt"},
        "resolved": {},
    }

    monkeypatch.setenv("GITHUB_REPOSITORY", "averinaleks/bot")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "321")
    monkeypatch.setenv("GITHUB_WORKFLOW", "dependency-graph")
    monkeypatch.setenv("GITHUB_JOB", "submit")

    monkeypatch.setattr(snapshot, "_build_manifests", lambda _: {"requirements.txt": manifest})

    captured: dict[str, object] = {}

    def capture_submission(url: str, body: bytes, headers: dict[str, str]) -> None:
        del url, headers
        captured.update(json.loads(body))

    monkeypatch.setattr(snapshot, "_submit_with_headers", capture_submission)
    monkeypatch.setattr(snapshot, "_discover_git_sha", lambda: "cafebabe")
    monkeypatch.setattr(snapshot, "_discover_git_ref", lambda: "refs/heads/git-main")

    snapshot.submit_dependency_snapshot()

    assert captured["sha"] == "cafebabe"
    assert captured["ref"] == "refs/heads/git-main"


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
