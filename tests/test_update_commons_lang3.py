"""Tests for the hardened commons-lang3 download helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable

import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "docker" / "scripts" / "update_commons_lang3.py"
    spec = importlib.util.spec_from_file_location("update_commons_lang3", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Не удалось загрузить модуль update_commons_lang3")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture()
def commons_module():
    """Return a fresh module instance for each test."""

    return _load_module()


def _mock_ip_resolver(monkeypatch: pytest.MonkeyPatch, module, values: Iterable[str]) -> None:
    monkeypatch.setattr(module, "_iter_resolved_ips", lambda _hostname: list(values))


def test_validate_download_url_accepts_public_ips(monkeypatch: pytest.MonkeyPatch, commons_module) -> None:
    module = commons_module
    _mock_ip_resolver(monkeypatch, module, ["8.8.8.8", "2001:4860:4860::8888"])

    parsed, ips = module._validate_download_url(module.COMMONS_LANG3_URL, return_ips=True)

    assert parsed.scheme == "https"
    assert parsed.hostname == module.COMMONS_LANG3_ALLOWED_HOST
    assert ips == {"8.8.8.8", "2001:4860:4860::8888"}


def test_validate_download_url_rejects_insecure_scheme(monkeypatch: pytest.MonkeyPatch, commons_module) -> None:
    module = commons_module
    insecure_url = module.COMMONS_LANG3_URL.replace("https://", "http://", 1)

    with pytest.raises(RuntimeError, match="Ожидалась схема https"):
        module._validate_download_url(insecure_url)


def test_validate_download_url_rejects_wrong_host(monkeypatch: pytest.MonkeyPatch, commons_module) -> None:
    module = commons_module
    other_host = module.COMMONS_LANG3_URL.replace(module.COMMONS_LANG3_ALLOWED_HOST, "example.com")

    with pytest.raises(RuntimeError, match="Получен неожиданный хост"):
        module._validate_download_url(other_host)


def test_validate_download_url_rejects_credentials(monkeypatch: pytest.MonkeyPatch, commons_module) -> None:
    module = commons_module
    url_with_credentials = module.COMMONS_LANG3_URL.replace("https://", "https://user:pass@", 1)

    with pytest.raises(RuntimeError, match="не должен содержать учетные данные"):
        module._validate_download_url(url_with_credentials)


def test_validate_download_url_rejects_custom_port(monkeypatch: pytest.MonkeyPatch, commons_module) -> None:
    module = commons_module
    parsed = module.urlsplit(module.COMMONS_LANG3_URL)
    custom_port = parsed._replace(netloc=f"{parsed.hostname}:8443")

    with pytest.raises(RuntimeError, match="Получен неожиданный порт"):
        module._validate_download_url(custom_port.geturl())


def test_validate_download_url_rejects_query(monkeypatch: pytest.MonkeyPatch, commons_module) -> None:
    module = commons_module
    parsed = module.urlsplit(module.COMMONS_LANG3_URL)
    with_query = parsed._replace(query="foo=bar")

    with pytest.raises(RuntimeError, match="не должен содержать параметры запроса"):
        module._validate_download_url(with_query.geturl())


def test_validate_download_url_rejects_fragment(monkeypatch: pytest.MonkeyPatch, commons_module) -> None:
    module = commons_module
    parsed = module.urlsplit(module.COMMONS_LANG3_URL)
    with_fragment = parsed._replace(fragment="section")

    with pytest.raises(RuntimeError, match="не должен содержать параметры запроса"):
        module._validate_download_url(with_fragment.geturl())


def test_validate_download_url_rejects_unexpected_path(monkeypatch: pytest.MonkeyPatch, commons_module) -> None:
    module = commons_module
    parsed = module.urlsplit(module.COMMONS_LANG3_URL)
    other_path = parsed._replace(path="/malicious.jar")

    with pytest.raises(RuntimeError, match="Получен неожиданный путь"):
        module._validate_download_url(other_path.geturl())


def test_validate_download_url_rejects_private_ips(monkeypatch: pytest.MonkeyPatch, commons_module) -> None:
    module = commons_module
    _mock_ip_resolver(monkeypatch, module, ["127.0.0.1", "10.0.0.5"])

    with pytest.raises(RuntimeError, match="небезопасные адреса"):
        module._validate_download_url(module.COMMONS_LANG3_URL)


def test_validate_download_url_rejects_empty_resolution(monkeypatch: pytest.MonkeyPatch, commons_module) -> None:
    module = commons_module
    _mock_ip_resolver(monkeypatch, module, [])

    with pytest.raises(RuntimeError, match="не разрешился"):
        module._validate_download_url(module.COMMONS_LANG3_URL)
