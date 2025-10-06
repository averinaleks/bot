from __future__ import annotations

from pathlib import Path


def _extract_permissions(text: str) -> dict[str, str]:
    permissions: dict[str, str] = {}
    inside_block = False
    base_indent: int | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not inside_block:
            if stripped == "permissions:":
                inside_block = True
            continue

        if not stripped or stripped.startswith("#"):
            continue

        current_indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = current_indent

        if current_indent < base_indent:
            break
        if current_indent > base_indent:
            # Skip nested structures; none are expected today.
            continue

        key, _, value = stripped.partition(":")
        permissions[key.strip()] = value.strip()

    return permissions


def test_dependency_graph_permissions_are_valid() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")
    permissions = _extract_permissions(workflow)

    assert permissions, "permissions block must be present"
    assert workflow.count("permissions:") == 1, "workflow must not define nested permission blocks"
    assert "dependency-graph:" not in workflow, "GitHub rejects the dependency-graph permission scope"

    allowed_keys = {"contents", "security-events"}
    assert set(permissions) <= allowed_keys

    assert permissions.get("contents") == "write"
    assert permissions.get("security-events") == "write"


def test_dependency_graph_detect_step_handles_nested_manifests() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "from pathlib import Path" in workflow
    assert "Path(file).name" in workflow
    assert "file_lower = file.lower()" in workflow
    assert "fnmatch(filename_lower, pattern)" in workflow


def test_dependency_graph_detect_step_normalises_null_strings() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert '_NULL_STRINGS = {"null", "none", "undefined"}' in workflow
    assert "def _normalise_value" in workflow
    assert "candidate.lower() in _NULL_STRINGS" in workflow


def test_dependency_graph_detect_step_uses_dispatch_commit_fallbacks() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "github.event.client_payload.before" in workflow
    assert "github.event.client_payload.base_sha" in workflow
    assert "github.event.client_payload.before_sha" in workflow
    assert "github.event.client_payload.previous_sha" in workflow
    assert "github.event.client_payload.previous_oid" in workflow
    assert "github.event.client_payload.head_sha" in workflow
    assert "github.event.client_payload.after" in workflow
    assert "github.event.client_payload.after_sha" in workflow
    assert "github.event.client_payload.commit_oid" in workflow
    assert "github.event.client_payload.commit_sha" in workflow
    assert "github.event.client_payload.sha" in workflow
    assert "github.event.workflow_run.head_sha" in workflow
    assert "github.event.workflow_run.head_commit.id" in workflow
    assert "github.event.workflow_run.head_commit.sha" in workflow
    assert "github.event.workflow_run.before" in workflow
    assert "github.event.workflow_run.previous_sha" in workflow
    assert "github.event.workflow_run.head_commit.before" in workflow


def test_dependency_graph_installs_requests_before_submission() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "Install dependency snapshot dependencies" in workflow
    assert "python -m pip install --upgrade pip" in workflow
    assert "python -m pip install requests" in workflow


def test_dependency_graph_filters_ccxtpro_lines_before_snapshot() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "Prepare requirements" in workflow
    assert "os.walk(root)" in workflow
    assert "name_lower = filename.lower()" in workflow
    assert 'stripped_lower = stripped.lower()' in workflow
    assert 'if stripped_lower.startswith("ccxtpro")' in workflow
    assert 'if stripped_lower.startswith("#") and "ccxtpro" in stripped_lower' in workflow


def test_dependency_graph_prepare_step_handles_filesystem_errors() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "except OSError as exc" in workflow
    assert "due to filesystem error" in workflow
    assert "::warning::Unable to update" in workflow


def test_dependency_graph_supports_repository_dispatch_auto_submission() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "repository_dispatch" in workflow
    assert "dependency-graph-auto-submission" in workflow
    assert "github.event_name == 'repository_dispatch'" in workflow


def test_dependency_graph_checkout_resolves_dispatch_ref() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "github.event.client_payload.head_sha" in workflow
    assert "github.event.client_payload.commit_sha" in workflow
    assert "github.event.client_payload.ref_name" in workflow
    assert "github.event.client_payload.head_ref" in workflow
    assert "github.event.client_payload.branch" in workflow
    assert "github.event.workflow_run.head_sha" in workflow
    assert "github.event.workflow_run.head_commit.id" in workflow
    assert "github.event.workflow_run.head_commit.sha" in workflow
    assert "github.event.workflow_run.head_ref" in workflow
    assert "github.event.workflow_run.head_branch" in workflow
