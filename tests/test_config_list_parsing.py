from typing import List

import config


def test_env_list_parsing_json(monkeypatch):
    monkeypatch.setenv("BACKUP_WS_URLS", '["ws://a", "ws://b"]')
    cfg = config.load_config()
    assert cfg.backup_ws_urls == ["ws://a", "ws://b"]


def test_env_list_parsing_csv(monkeypatch):
    monkeypatch.setenv("BACKUP_WS_URLS", 'ws://a,ws://b')
    cfg = config.load_config()
    assert cfg.backup_ws_urls == ["ws://a", "ws://b"]


def test_convert_list_int():
    assert config._convert("[1, 2, 3]", List[int]) == [1, 2, 3]
    assert config._convert("1,2,3", List[int]) == [1, 2, 3]


def test_convert_list_float():
    assert config._convert("[1.1, 2.2]", List[float]) == [1.1, 2.2]
    assert config._convert("1.1,2.2", List[float]) == [1.1, 2.2]


def test_convert_list_bool():
    assert config._convert("[true, false, true]", List[bool]) == [True, False, True]
    assert config._convert("true,false", List[bool]) == [True, False]
