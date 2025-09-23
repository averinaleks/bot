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

    allowed_keys = {"contents", "security-events"}
    assert set(permissions) <= allowed_keys

    assert permissions.get("contents") == "read"
    assert permissions.get("security-events") == "write"


def test_dependency_graph_detect_step_handles_nested_manifests() -> None:
    workflow = Path(".github/workflows/dependency-graph.yml").read_text(encoding="utf-8")

    assert "from pathlib import Path" in workflow
    assert "Path(file).name" in workflow
    assert "fnmatch(filename, pattern)" in workflow
