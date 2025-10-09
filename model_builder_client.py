from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, Iterable, List, Optional, Sequence, Tuple, cast
from urllib.parse import urlparse

try:
    import httpx
except ImportError:  # pragma: no cover - exercised in dedicated test
    from services.stubs import create_httpx_stub, is_offline_env

    httpx = create_httpx_stub()
    if not is_offline_env():
        logging.getLogger("TradingBot").warning(
            "Модуль httpx не найден: используется оффлайн-стаб"
        )

from bot import config as bot_config
from services.logging_utils import sanitize_log_value


logger = logging.getLogger("TradingBot")


@dataclass(frozen=True)
class ServiceEndpoint:
    """Validated endpoint information for model-related services."""

    scheme: str
    base_url: str
    hostname: str
    allowed_ips: frozenset[str]


_MODEL_VERSION = 0

_ALLOWED_HOSTS_ENV = "MODEL_BUILDER_ALLOWED_HOSTS"
_DEFAULT_ALLOWED_HOSTS = frozenset(
    {"127.0.0.1", "localhost", "::1", "model_builder", "data_handler"}
)


_MAX_TRAINING_PAYLOAD_BYTES = 1_000_000


AddrInfo = Tuple[Any, Any, Any, Any, Tuple[Any, ...]]


def _normalise_allowed_host(value: str) -> str | None:
    trimmed = value.strip()
    if not trimmed:
        return None
    if trimmed.startswith("[") and trimmed.endswith("]"):
        trimmed = trimmed[1:-1]
    return trimmed.lower()


def _load_allowed_hosts() -> set[str]:
    raw = os.getenv(_ALLOWED_HOSTS_ENV)
    hosts = set(_DEFAULT_ALLOWED_HOSTS)
    if not raw:
        return hosts
    for part in raw.split(","):
        normalised = _normalise_allowed_host(part)
        if not normalised:
            continue

        if normalised in hosts:
            continue

        try:
            infos = cast(
                Sequence[AddrInfo],
                socket.getaddrinfo(normalised, None, family=socket.AF_UNSPEC),
            )
        except socket.gaierror as exc:
            logger.warning(
                "Пропускаем %r из %s: не удалось разрешить хост (%s)",
                normalised,
                _ALLOWED_HOSTS_ENV,
                exc,
            )
            continue

        resolved_ips = _collect_ip_strings(infos)
        if not resolved_ips:
            logger.warning(
                "Пропускаем %r из %s: имя хоста не разрешилось ни в один IP",
                normalised,
                _ALLOWED_HOSTS_ENV,
            )
            continue

        unsafe_ips: list[str] = []
        for ip in resolved_ips:
            try:
                parsed = ip_address(ip)
            except ValueError:
                unsafe_ips.append(ip)
                continue
            if not (parsed.is_loopback or parsed.is_private):
                unsafe_ips.append(ip)
        if unsafe_ips:
            logger.warning(
                "Пропускаем %r из %s: resolve в непригодные IP %s",
                normalised,
                _ALLOWED_HOSTS_ENV,
                ", ".join(sorted(unsafe_ips)),
            )
            continue

        hosts.add(normalised)
    return hosts


def _host_is_allowlisted(hostname: str, allowed_hosts: set[str]) -> bool:
    host = hostname.lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if host in allowed_hosts:
        return True
    try:
        ip_obj = ip_address(host)
    except ValueError:
        return False
    if ip_obj.is_loopback or ip_obj.is_private or ip_obj.is_link_local:
        return True
    return host in allowed_hosts


def _collect_ip_strings(infos: Iterable[AddrInfo]) -> set[str]:
    """Extract string IP addresses from *infos* ignoring unexpected variants."""

    results: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        host = sockaddr[0]
        if isinstance(host, bytes):
            try:
                host = host.decode()
            except UnicodeDecodeError:
                continue
        if isinstance(host, str):
            results.add(host)
    return results


def _resolve_hostname(hostname: str) -> set[str]:
    """Return IP addresses resolving *hostname*, respecting ``TEST_MODE``."""

    try:
        infos = cast(Sequence[AddrInfo], socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC))
    except socket.gaierror as exc:
        if os.getenv("TEST_MODE") == "1":
            return {"127.0.0.1"}
        raise ValueError(
            f"Не удалось разрешить имя хоста {sanitize_log_value(hostname)!r}"
        ) from exc
    return _collect_ip_strings(infos)


def _is_private_ip(ip_text: str) -> bool:
    address = ip_address(ip_text)
    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
    )


def _prepare_endpoint(raw_url: str, *, purpose: str) -> ServiceEndpoint | None:
    """Validate and normalize *raw_url* for the given *purpose*."""

    safe_url = sanitize_log_value(raw_url)
    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.hostname:
        logger.error("%s URL %s отклонён: отсутствует схема или хост", purpose, safe_url)
        return None

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        logger.error(
            "%s URL %s использует неподдерживаемую схему %r", purpose, safe_url, scheme
        )
        return None

    if parsed.username or parsed.password:
        logger.error(
            "%s URL %s отклонён: учетные данные в URL запрещены",
            purpose,
            safe_url,
        )
        return None

    allowed_hosts = _load_allowed_hosts()
    try:
        resolved_ips = _resolve_hostname(parsed.hostname)
    except ValueError as exc:
        logger.error("%s URL %s отклонён: %s", purpose, safe_url, exc)
        return None

    if not resolved_ips:
        logger.error(
            "%s URL %s отклонён: имя хоста не разрешилось ни в один IP", purpose, safe_url
        )
        return None

    hostname = parsed.hostname
    if not _host_is_allowlisted(hostname, allowed_hosts):
        logger.error(
            "%s URL %s отклонён: хост не входит в список доверенных", purpose, safe_url
        )
        return None

    if scheme == "http" and not all(_is_private_ip(ip) for ip in resolved_ips):
        logger.error(
            "%s URL %s должен использовать HTTPS или частный адрес", purpose, safe_url
        )
        return None

    return ServiceEndpoint(
        scheme=scheme,
        base_url=raw_url.rstrip("/"),
        hostname=hostname,
        allowed_ips=frozenset(resolved_ips),
    )


async def _hostname_still_allowed(endpoint: ServiceEndpoint) -> bool:
    """Detect DNS rebinding by ensuring the host resolves within the allowed set."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        infos: Sequence[AddrInfo]
        if loop is None:
            infos = cast(
                Sequence[AddrInfo],
                socket.getaddrinfo(
                    endpoint.hostname, None, family=socket.AF_UNSPEC
                ),
            )
        else:
            infos = cast(
                Sequence[AddrInfo],
                await loop.getaddrinfo(
                    endpoint.hostname, None, family=socket.AF_UNSPEC
                ),
            )
    except socket.gaierror as exc:
        if os.getenv("TEST_MODE") == "1":
            return True
        logger.error(
            "Повторное разрешение %s не удалось: %s",
            sanitize_log_value(endpoint.hostname),
            exc,
        )
        return False

    current_ips = _collect_ip_strings(infos)
    if not current_ips:
        logger.error(
            "Повторное разрешение %s не вернуло IP-адресов",
            sanitize_log_value(endpoint.hostname),
        )
        return False

    if not current_ips & endpoint.allowed_ips:
        logger.error(
            "Обнаружена попытка DNS-rebinding для %s: %s не пересекаются с %s",
            sanitize_log_value(endpoint.hostname),
            {sanitize_log_value(ip) for ip in current_ips},
            {sanitize_log_value(ip) for ip in endpoint.allowed_ips},
        )
        return False
    return True


async def _fetch_training_data_from_endpoint(
    endpoint: ServiceEndpoint, symbol: str, limit: int = 50
) -> Tuple[List[List[float]], List[int]]:
    """Fetch recent OHLCV data and derive direction labels from *endpoint*."""

    features: List[List[float]] = []
    labels: List[int] = []

    if not await _hostname_still_allowed(endpoint):
        return features, labels

    timeout = 5.0
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
            resp = await client.get(
                f"{endpoint.base_url}/ohlcv/{symbol}",
                params={"limit": limit},
                timeout=timeout,
            )
        if resp.status_code != 200:
            logger.error("Failed to fetch OHLCV: HTTP %s", resp.status_code)
            return features, labels

        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type and content_type != "application/json":
            logger.error(
                "Rejected training data with unexpected Content-Type %s",
                sanitize_log_value(resp.headers.get("Content-Type", "")),
            )
            return features, labels

        payload_bytes = resp.content
        if len(payload_bytes) > _MAX_TRAINING_PAYLOAD_BYTES:
            logger.error(
                "Training data response exceeded %s bytes limit",
                _MAX_TRAINING_PAYLOAD_BYTES,
            )
            return features, labels

        try:
            parsed = resp.json()
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode training data JSON: %s", exc)
            return features, labels

        if not isinstance(parsed, dict):
            logger.error(
                "Training data payload must be a JSON object, got %s",
                type(parsed).__name__,
            )
            return features, labels

        ohlcv = parsed.get("ohlcv", [])
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        logger.error("Training data request error: %s", exc)
        return features, labels

    for i in range(1, len(ohlcv)):
        try:
            _ts, open_price, high_price, low_price, close_price, volume = ohlcv[i]
            prev_close = float(ohlcv[i - 1][4])
            direction = 1 if float(close_price) > prev_close else 0
            features.append(
                [
                    float(open_price),
                    float(high_price),
                    float(low_price),
                    float(close_price),
                    float(volume),
                ]
            )
            labels.append(direction)
        except (ValueError, TypeError, IndexError):
            continue
    return features, labels


async def _train_with_endpoint(
    endpoint: ServiceEndpoint, features: List[List[float]], labels: List[int]
) -> bool:
    """Send training data to *endpoint* ensuring DNS rebinding protection."""

    if not await _hostname_still_allowed(endpoint):
        return False

    payload = {"features": features, "labels": labels}
    timeout = 5.0
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
            response = await client.post(
                f"{endpoint.base_url}/train", json=payload, timeout=timeout
            )
        if response.status_code == 200:
            return True
        logger.error("Model training failed: HTTP %s", response.status_code)
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        logger.error("Model training request error: %s", exc)
    return False


async def _fetch_training_data(
    data_url: str, symbol: str, limit: int = 50
) -> Tuple[List[List[float]], List[int]]:
    """Backward compatible wrapper returning training data for *data_url*."""

    if bot_config.OFFLINE_MODE:
        logger.info("Offline mode: skipping training data fetch for %s", symbol)
        return [], []

    endpoint = _prepare_endpoint(data_url, purpose="URL сервиса данных")
    if endpoint is None:
        return [], []
    return await _fetch_training_data_from_endpoint(endpoint, symbol, limit)


async def train(url: str, features: List[List[float]], labels: List[int]) -> bool:
    """Send training data to the model_builder service."""

    if bot_config.OFFLINE_MODE:
        logger.info("Offline mode: skipping model training request")
        return True

    endpoint = _prepare_endpoint(url, purpose="URL сервиса обучения")
    if endpoint is None:
        return False
    return await _train_with_endpoint(endpoint, features, labels)


async def retrain(
    model_url: str, data_url: str, symbol: str = "BTCUSDT"
) -> Optional[int]:
    """Retrain the model using data from ``data_url`` with strict URL validation."""

    global _MODEL_VERSION
    if bot_config.OFFLINE_MODE:
        logger.info("Offline mode: retrain skipped")
        return _MODEL_VERSION

    data_endpoint = _prepare_endpoint(data_url, purpose="URL сервиса данных")
    model_endpoint = _prepare_endpoint(model_url, purpose="URL сервиса обучения")

    if data_endpoint is None or model_endpoint is None:
        logger.error("Ретрайн невозможен: некорректные URL сервисов")
        return None

    features, labels = await _fetch_training_data_from_endpoint(data_endpoint, symbol)
    if not features or not labels:
        logger.error("No training data available for retrain")
        return None

    if await _train_with_endpoint(model_endpoint, features, labels):
        _MODEL_VERSION += 1
        logger.info("Model retrained, new version %s", _MODEL_VERSION)
        return _MODEL_VERSION
    return None


async def _retrain_loop(
    model_url: str, data_url: str, interval: float, symbol: str
) -> None:
    if bot_config.OFFLINE_MODE:
        logger.info("Offline mode: retrain loop disabled")
        return
    while True:
        await retrain(model_url, data_url, symbol)
        await asyncio.sleep(interval)


def schedule_retrain(
    model_url: str, data_url: str, interval: float, symbol: str = "BTCUSDT"
) -> asyncio.Task:
    """Start periodic retraining task."""

    return asyncio.create_task(_retrain_loop(model_url, data_url, interval, symbol))

