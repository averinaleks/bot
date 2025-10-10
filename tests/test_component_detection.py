from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace


def _load_component_detection():
    path = Path("component-detection")
    loader = importlib.machinery.SourceFileLoader("component_detection", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Unable to load component-detection script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_scan_uses_list_for_container_layer_ids(tmp_path: Path) -> None:
    module = _load_component_detection()
    manifest = tmp_path / "requirements.txt"
    manifest.write_text("requests==2.32.5\n", encoding="utf-8")

    output_path = tmp_path / "scan.json"
    args = SimpleNamespace(
        SourceDirectory=str(tmp_path),
        ManifestFile=str(output_path),
        PrintManifest=False,
    )

    return_code = module.run_scan(args)  # type: ignore[attr-defined]
    assert return_code == 0

    result = json.loads(output_path.read_text(encoding="utf-8"))
    components = result["componentsFound"]
    assert components, "Expected at least one component in scan results"
    for component in components:
        assert component["containerLayerIds"] == []


def test_run_scan_rejects_manifest_outside_source(tmp_path: Path, capsys) -> None:
    module = _load_component_detection()
    manifest = tmp_path / "requirements.txt"
    manifest.write_text("requests==2.32.5\n", encoding="utf-8")

    outside_path = tmp_path.parent / "outside.json"
    args = SimpleNamespace(
        SourceDirectory=str(tmp_path),
        ManifestFile=str(outside_path),
        PrintManifest=False,
    )

    return_code = module.run_scan(args)  # type: ignore[attr-defined]

    captured = capsys.readouterr()
    assert return_code == 1
    assert "Refusing to write manifest outside SourceDirectory" in captured.err
    assert not outside_path.exists()


def test_run_scan_rejects_manifest_symlink_parent(tmp_path: Path, capsys) -> None:
    module = _load_component_detection()
    manifest = tmp_path / "requirements.txt"
    manifest.write_text("requests==2.32.5\n", encoding="utf-8")

    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    symlink_path = safe_dir / "link"
    symlink_target = tmp_path.parent / "escaped"
    symlink_target.mkdir(exist_ok=True)
    os.symlink(symlink_target, symlink_path)

    manifest_path = symlink_path / "manifest.json"
    args = SimpleNamespace(
        SourceDirectory=str(tmp_path),
        ManifestFile=str(manifest_path.relative_to(tmp_path)),
        PrintManifest=False,
    )

    return_code = module.run_scan(args)  # type: ignore[attr-defined]

    captured = capsys.readouterr()
    assert return_code == 1
    assert "Refusing to write manifest outside SourceDirectory" in captured.err
    assert not manifest_path.exists()


def test_main_ignores_unknown_arguments(tmp_path: Path, capsys) -> None:
    module = _load_component_detection()
    manifest = tmp_path / "requirements.txt"
    manifest.write_text("requests==2.32.5\n", encoding="utf-8")

    return_code = module.main(
        [
            "scan",
            "--SourceDirectory",
            str(tmp_path),
            "--UnsupportedFlag",
            "value",
        ]
    )

    captured = capsys.readouterr()

    assert return_code == 0
    assert "Ignoring unsupported component-detection arguments" in captured.err
    assert "--UnsupportedFlag value" in captured.err
