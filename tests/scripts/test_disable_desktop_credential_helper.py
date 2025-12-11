import json

from scripts import disable_desktop_credential_helper as helper


def test_malformed_config_is_rebuilt(tmp_path, monkeypatch, capsys):
    config_dir = tmp_path / "docker-config"
    config_dir.mkdir()
    config_path = config_dir / helper.CONFIG_FILENAME
    config_path.write_text("{invalid json", encoding="utf-8")

    monkeypatch.setenv(helper.DOCKER_CONFIG_ENV, str(config_dir))

    helper.main()

    backups = list(config_dir.glob("config.json.backup-" + "*"))
    assert backups, "backup of invalid config should be created"

    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    assert loaded == {}

    output = capsys.readouterr().out
    assert "Invalid Docker config" in output
    assert "Rewrote Docker config" in output
