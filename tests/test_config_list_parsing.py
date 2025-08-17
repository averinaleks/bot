import config


def test_env_list_parsing_json(monkeypatch):
    monkeypatch.setenv("BACKUP_WS_URLS", '["ws://a", "ws://b"]')
    cfg = config.load_config()
    assert cfg.backup_ws_urls == ["ws://a", "ws://b"]


def test_env_list_parsing_csv(monkeypatch):
    monkeypatch.setenv("BACKUP_WS_URLS", 'ws://a,ws://b')
    cfg = config.load_config()
    assert cfg.backup_ws_urls == ["ws://a", "ws://b"]
