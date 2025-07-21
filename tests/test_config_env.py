import os
from config import load_config


def test_backup_ws_urls_env(monkeypatch, tmp_path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text("{}")
    monkeypatch.setenv("BACKUP_WS_URLS", "['wss://a','wss://b']")
    cfg = load_config(str(cfg_file))
    assert cfg.backup_ws_urls == ['wss://a', 'wss://b']


def test_backup_ws_urls_json_env(monkeypatch, tmp_path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text("{}")
    monkeypatch.setenv("BACKUP_WS_URLS", '["wss://a","wss://b"]')
    cfg = load_config(str(cfg_file))
    assert cfg.backup_ws_urls == ["wss://a", "wss://b"]


def test_ws_min_process_rate_default(tmp_path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text('{"timeframe": "2h"}')
    cfg = load_config(str(cfg_file))
    assert cfg.ws_min_process_rate == 1
