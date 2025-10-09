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


def _extract_events(text: str) -> list[str]:
    events: list[str] = []
    inside_block = False
    base_indent: int | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not inside_block:
            if stripped == "on:":
                inside_block = True
            continue

        current_indent = len(line) - len(line.lstrip())
        if current_indent <= 0:
            break

        if not stripped or stripped.startswith("#"):
            continue

        if base_indent is None:
            base_indent = current_indent

        if current_indent < base_indent:
            break
        if current_indent > base_indent:
            continue

        events.append(stripped.rstrip(":"))

    return events


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


def test_dependency_graph_events_exclude_dependency_graph_trigger() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")
    events = _extract_events(workflow)

    assert {"push", "workflow_dispatch", "repository_dispatch"} <= set(events)
    assert "dependency_graph" not in events


def test_dependency_graph_detect_step_handles_nested_manifests() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "from pathlib import Path" in workflow
    assert "Path(file).name" in workflow
    assert "file_lower = file.lower()" in workflow
    assert "fnmatch(filename_lower, pattern)" in workflow


def test_dependency_graph_detect_step_normalises_null_strings() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert """_NULL_STRINGS = {"null", "none", "undefined", '""', "''"}""" in workflow
    assert "def _normalise_value" in workflow
    assert "candidate.lower() in _NULL_STRINGS" in workflow


def test_dependency_graph_detect_step_uses_dispatch_commit_fallbacks() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "github.event.base_sha" in workflow
    assert "github.event.base_ref" in workflow
    assert "github.event.dependency_graph.before_sha" in workflow
    assert "github.event.dependency_graph.beforeOid" in workflow
    assert "github.event.dependency_graph.beforeSha" in workflow
    assert "github.event.dependency_graph.base_ref" in workflow
    assert "github.event.dependency_graph.baseRef" in workflow
    assert "github.event.dependency_graph.baseRefName" in workflow
    assert "github.event.dependency_graph.base_sha" in workflow
    assert "github.event.client_payload.before" in workflow
    assert "github.event.client_payload.base_sha" in workflow
    assert "github.event.client_payload.before_sha" in workflow
    assert "github.event.client_payload.beforeSha" in workflow
    assert "github.event.client_payload.beforeOid" in workflow
    assert "github.event.client_payload.previous_sha" in workflow
    assert "github.event.client_payload.previous_oid" in workflow
    assert "github.event.client_payload.previousSha" in workflow
    assert "github.event.client_payload.previousOid" in workflow
    assert "github.event.client_payload.head_sha" in workflow
    assert "github.event.client_payload.headSha" in workflow
    assert "github.event.client_payload.after" in workflow
    assert "github.event.client_payload.after_sha" in workflow
    assert "github.event.client_payload.afterSha" in workflow
    assert "github.event.client_payload.afterOid" in workflow
    assert "github.event.client_payload.commit_oid" in workflow
    assert "github.event.client_payload.commitOid" in workflow
    assert "github.event.client_payload.commit_sha" in workflow
    assert "github.event.client_payload.sha" in workflow
    assert "github.event.commit_oid" in workflow
    assert "github.event.commitOid" in workflow
    assert "github.event.commit_sha" in workflow
    assert "github.event.sha" in workflow
    assert "github.event.dependency_graph.commit_oid" in workflow
    assert "github.event.dependency_graph.commitOid" in workflow
    assert "github.event.dependency_graph.after_sha" in workflow
    assert "github.event.dependency_graph.afterSha" in workflow
    assert "github.event.dependency_graph.afterOid" in workflow
    assert "github.event.dependency_graph.after" in workflow
    assert "github.event.dependency_graph.ref" in workflow
    assert "github.event.ref" in workflow
    assert "github.event.workflow_run.head_sha" in workflow
    assert "github.event.workflow_run.head_commit.id" in workflow
    assert "github.event.workflow_run.head_commit.sha" in workflow
    assert "github.event.workflow_run.head_commit.after" in workflow
    assert "github.event.workflow_run.head_commit.afterSha" in workflow
    assert "github.event.workflow_run.before" in workflow
    assert "github.event.workflow_run.previous_sha" in workflow
    assert "github.event.workflow_run.head_commit.before" in workflow
    assert "github.event.workflow_run.head_commit.beforeSha" in workflow


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


def test_dependency_graph_notes_dependency_graph_event_limitations() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "repository_dispatch" in workflow
    assert "dependency-graph-auto-submission" in workflow
    note = "GitHub does not yet accept the experimental ``dependency_graph`` event"
    assert note in workflow
    assert "Auto-submission payloads should" in workflow
    assert "therefore be forwarded via the repository dispatch hook declared above" in workflow


def test_dependency_graph_checkout_resolves_dispatch_ref() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "github.event.ref" in workflow
    assert "github.event.ref_name" in workflow
    assert "github.event.refName" in workflow
    assert "github.event.branch" in workflow
    assert "github.event.branch_name" in workflow
    assert "github.event.head_ref" in workflow
    assert "github.event.client_payload.headRef" in workflow
    assert "github.event.base_ref" in workflow
    assert "github.event.dependency_graph.sha" in workflow
    assert "github.event.dependency_graph.commit_oid" in workflow
    assert "github.event.dependency_graph.commitOid" in workflow
    assert "github.event.dependency_graph.ref" in workflow
    assert "github.event.dependency_graph.refName" in workflow
    assert "github.event.dependency_graph.branch" in workflow
    assert "github.event.dependency_graph.branchName" in workflow
    assert "github.event.client_payload.ref" in workflow
    assert "github.event.client_payload.ref_name" in workflow
    assert "github.event.client_payload.refName" in workflow
    assert "github.event.client_payload.branch" in workflow
    assert "github.event.client_payload.branch_name" in workflow
    assert "github.event.client_payload.branchName" in workflow
    assert "github.event.workflow_run.head_ref" in workflow
    assert "github.event.workflow_run.headRef" in workflow
    assert "github.event.workflow_run.head_branch" in workflow
    assert "github.event.workflow_run.head_branch_name" in workflow
    assert "github.event.workflow_run.headBranch" in workflow
    assert "github.event.dependency_graph.ref_name" in workflow
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
    assert "github.event.repository.default_branch" in workflow
    assert "format('refs/heads/{0}', github.event.repository.default_branch)" in workflow
