import socket

import pytest

from docker.scripts import update_commons_lang3 as updater


@pytest.fixture
def mock_getaddrinfo(monkeypatch):
    def _apply(addresses):
        def _fake_getaddrinfo(_host, *_args, **_kwargs):
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    (address, 443),
                )
                for address in addresses
            ]

        monkeypatch.setattr(updater.socket, "getaddrinfo", _fake_getaddrinfo)

    return _apply


def test_validate_download_url_accepts_https_and_public_ip(mock_getaddrinfo):
    mock_getaddrinfo(["151.101.0.204"])
    parsed = updater._validate_download_url(updater.COMMONS_LANG3_URL)
    assert parsed.scheme == "https"
    assert parsed.hostname == updater.COMMONS_LANG3_ALLOWED_HOST


def test_validate_download_url_rejects_insecure_scheme(mock_getaddrinfo):
    mock_getaddrinfo(["151.101.0.204"])
    with pytest.raises(RuntimeError, match="Ожидалась схема https"):
        updater._validate_download_url(
            updater.COMMONS_LANG3_URL.replace("https://", "http://", 1)
        )


def test_validate_download_url_rejects_unexpected_host(mock_getaddrinfo):
    mock_getaddrinfo(["151.101.0.204"])
    with pytest.raises(RuntimeError, match="Получен неожидан"):
        updater._validate_download_url("https://example.com/file.jar")


def test_validate_download_url_rejects_private_ip(mock_getaddrinfo):
    mock_getaddrinfo(["127.0.0.1"])
    with pytest.raises(RuntimeError, match="небезопасные адреса"):
        updater._validate_download_url(updater.COMMONS_LANG3_URL)


def test_validate_download_url_dns_failure(monkeypatch):
    def _raise(_host, *_args, **_kwargs):
        raise updater.socket.gaierror("boom")

    monkeypatch.setattr(updater.socket, "getaddrinfo", _raise)

    with pytest.raises(RuntimeError, match="Не удалось разрешить хост"):
        updater._validate_download_url(updater.COMMONS_LANG3_URL)

