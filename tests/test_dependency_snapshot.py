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
    assert any(
        entry["package_url"] == "pkg:pypi/requests@2.32.3"
        for entry in manifest_data["resolved"].values()
    )


def test_parse_requirements_skips_blocklisted_packages(tmp_path: Path) -> None:
    requirement_file = tmp_path / "requirements.txt"
    requirement_file.write_text("ccxtpro==1.0.1\nhttpx==0.27.2\n")

    resolved = snapshot._parse_requirements(requirement_file)

    assert "pkg:pypi/ccxtpro@1.0.1" not in resolved
    assert "pkg:pypi/httpx@0.27.2" in resolved
    assert (
        resolved["pkg:pypi/httpx@0.27.2"]["package_url"]
        == "pkg:pypi/httpx@0.27.2"
    )


def test_parse_requirements_encodes_versions_for_purl(tmp_path: Path) -> None:
    requirement_file = tmp_path / "requirements.txt"
    requirement_file.write_text("torch==2.8.0+cpu\n")

    resolved = snapshot._parse_requirements(requirement_file)

    assert (
        resolved["pkg:pypi/torch@2.8.0%2Bcpu"]["package_url"]
        == "pkg:pypi/torch@2.8.0%2Bcpu"
    )
