"""Utilities for interacting with the GPT OSS API.

The API endpoint is configured via the ``GPT_OSS_API`` environment variable.
For security reasons, the URL must either use HTTPS or resolve to a local
or private address (HTTP is allowed only for such addresses).
"""

import asyncio
import importlib
import importlib.util
import json
import logging
import os
import socket
import threading
from enum import Enum
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any, Coroutine, Mapping
from urllib.parse import urljoin, urlparse

from bot import config as bot_config
from bot.pydantic_compat import BaseModel, Field, ValidationError
from bot.utils import retry
from services.logging_utils import sanitize_log_value
from services.stubs import create_httpx_stub, is_offline_env


_OFFLINE_ENV = bool(bot_config.OFFLINE_MODE) or is_offline_env()

try:  # pragma: no cover - exercised in offline/import-error scenarios
    if _OFFLINE_ENV:
        raise ImportError("offline mode uses httpx stub")
    import httpx as _httpx  # type: ignore
except Exception:  # noqa: BLE001 - ensure stubs are available without httpx
    httpx = create_httpx_stub()
    __offline_stub__ = True
else:
    httpx = _httpx
    __offline_stub__ = False

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from openai import OpenAI as OpenAIType  # noqa: F401
else:
    OpenAIType = Any

_openai_spec = importlib.util.find_spec("openai")
if _openai_spec is not None:
    _openai_module = importlib.import_module("openai")
    OpenAI: OpenAIType | None = getattr(_openai_module, "OpenAI", None)
else:
    OpenAI = None
if bot_config.OFFLINE_MODE:
    from services.offline import OfflineGPT

logger = logging.getLogger("TradingBot")

# Maximum allowed prompt size in bytes
MAX_PROMPT_BYTES = 10000
# Maximum allowed response size in bytes
MAX_RESPONSE_BYTES = 10000
# Maximum number of retries for network requests
MAX_RETRIES = 3
# Marker stored in allowed IP sets when DNS resolution failed in TEST_MODE.
_TEST_MODE_DNS_FALLBACK = "__test_mode_dns_fallback__"
# Default hosts that are considered safe targets for GPT OSS API requests.
#
# ``gptoss`` is the service name used inside docker-compose setups for the
# GPT-OSS container.  Allowing it here keeps local development and the
# integration tests (which rely on that hostname) working without requiring
# additional configuration.  Loopback addresses remain permitted out of the
# box for single-host deployments.
_DEFAULT_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "gptoss"})


def _normalise_ip(value: str) -> str:
    """Return a canonical representation for IPv4 and IPv6 addresses."""

    try:
        ip_obj = ip_address(value)
    except ValueError:
        return value

    # GitHub runners may resolve IPv4 hosts through IPv6-mapped addresses
    # (``::ffff:127.0.0.1``). Normalising keeps comparison stable.
    if hasattr(ip_obj, "ipv4_mapped") and ip_obj.ipv4_mapped is not None:
        return str(ip_obj.ipv4_mapped)
    return str(ip_obj)


def _allow_insecure_url() -> bool:
    """Return True if insecure GPT_OSS_API URLs are explicitly allowed."""
    return os.getenv("ALLOW_INSECURE_GPT_URL") == "1"

# Backward compatibility: expose module-level flag for tests/legacy code.
ALLOW_INSECURE_GPT_URL = _allow_insecure_url()


def _insecure_allowed() -> bool:
    """Return True if insecure GPT URLs are explicitly permitted."""

    return ALLOW_INSECURE_GPT_URL or os.getenv("ALLOW_INSECURE_GPT_URL") == "1"


def _parse_timeout(raw_timeout: str | None) -> float:
    """Return a positive timeout value parsed from ``raw_timeout``."""

    if not raw_timeout:
        return 5.0
    try:
        timeout = float(raw_timeout)
        if timeout <= 0:
            logger.warning(
                "Non-positive GPT_OSS_TIMEOUT value %r; defaulting to 5.0",
                raw_timeout,
            )
            return 5.0
        return timeout
    except ValueError:
        logger.warning(
            "Invalid GPT_OSS_TIMEOUT value %r; defaulting to 5.0",
            raw_timeout,
        )
        return 5.0


def _should_use_openai(api_url: str | None) -> bool:
    """Return ``True`` when the OpenAI client should service requests."""

    if OpenAI is None:
        return False
    if bot_config.OFFLINE_MODE:
        return False
    if not api_url:
        return bool(os.getenv("OPENAI_API_KEY"))
    trimmed = api_url.strip().lower()
    if trimmed in {"openai", "openai://"}:
        return True
    parsed = urlparse(api_url)
    scheme = parsed.scheme.lower()
    return scheme.startswith("openai")


def _resolve_openai_base_url(api_url: str | None) -> str | None:
    """Return the base URL that should be passed to the OpenAI client."""

    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_url:
        return base_url
    parsed = urlparse(api_url)
    scheme = parsed.scheme.lower()
    if scheme in {"openai", "openai+https", "openai+http"}:
        transport = "https" if scheme != "openai+http" else "http"
        netloc = parsed.netloc
        path = parsed.path.rstrip("/")
        if netloc:
            return f"{transport}://{netloc}{path}"
        return base_url
    return base_url


def _serialise_openai_response(response: Any) -> bytes:
    """Serialise *response* to JSON bytes compatible with :func:`_parse_gpt_response`."""

    if hasattr(response, "model_dump_json"):
        dumped = response.model_dump_json()
        if isinstance(dumped, str):
            return dumped.encode("utf-8")
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif hasattr(response, "dict"):
        try:
            data = response.dict()  # type: ignore[assignment]
        except TypeError:
            data = response.dict()  # type: ignore[assignment]
    elif hasattr(response, "to_dict"):
        data = response.to_dict()
    elif isinstance(response, Mapping):
        data = response
    else:
        data = getattr(response, "__dict__", None)
    if not isinstance(data, Mapping):
        raise GPTClientResponseError("Unexpected response structure from OpenAI API")
    try:
        return json.dumps(data).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GPTClientResponseError(
            "Unexpected response structure from OpenAI API",
        ) from exc


def _query_openai(prompt: str, api_url: str | None) -> str:
    """Send *prompt* to the OpenAI API and return the first completion text."""

    if OpenAI is None:
        raise GPTClientConfigurationError(
            "OpenAI client support is not available; install the 'openai' package"
        )
    timeout = _parse_timeout(os.getenv("GPT_OSS_TIMEOUT"))
    model_name = os.getenv("OPENAI_MODEL") or os.getenv("GPT_MODEL") or "gpt-4o-mini"
    client_kwargs: dict[str, Any] = {"timeout": timeout}
    base_url = _resolve_openai_base_url(api_url)
    if base_url:
        client_kwargs["base_url"] = base_url
    organization = os.getenv("OPENAI_ORGANIZATION") or os.getenv("OPENAI_ORG")
    if organization:
        client_kwargs["organization"] = organization

    logger.debug("Using OpenAI client with kwargs: %s", client_kwargs)
    client = OpenAI(**client_kwargs)

    completion_kwargs: dict[str, Any] = {}
    max_tokens_raw = os.getenv("OPENAI_MAX_TOKENS")
    if max_tokens_raw:
        try:
            max_tokens = int(max_tokens_raw)
            if max_tokens <= 0:
                raise ValueError
        except ValueError:
            logger.warning(
                "Invalid OPENAI_MAX_TOKENS value %r; ignoring", max_tokens_raw
            )
        else:
            completion_kwargs["max_tokens"] = max_tokens
    temperature_raw = os.getenv("OPENAI_TEMPERATURE")
    if temperature_raw:
        try:
            completion_kwargs["temperature"] = float(temperature_raw)
        except ValueError:
            logger.warning(
                "Invalid OPENAI_TEMPERATURE value %r; ignoring", temperature_raw
            )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
            **completion_kwargs,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime env
        logger.exception("Error querying OpenAI API: %s", exc)
        raise GPTClientNetworkError("Failed to query OpenAI API") from exc

    content = _serialise_openai_response(response)
    return _parse_gpt_response(content)

class GPTClientError(Exception):
    """Base exception for GPT client errors."""


class GPTClientConfigurationError(GPTClientError):
    """Raised when GPT client configuration is invalid."""


class GPTClientNetworkError(GPTClientError):
    """Raised when the GPT OSS API cannot be reached."""


class GPTClientJSONError(GPTClientError):
    """Raised when the GPT OSS API returns invalid JSON."""


class GPTClientResponseError(GPTClientError):
    """Raised when the GPT OSS API returns an unexpected structure."""


class SignalAction(str, Enum):
    """Possible trading actions."""

    buy = "buy"
    sell = "sell"
    hold = "hold"


class Signal(BaseModel):
    """Parsed trading signal from GPT output."""

    signal: SignalAction = SignalAction.hold
    tp_mult: float | None = Field(default=None, ge=0, le=10)
    sl_mult: float | None = Field(default=None, ge=0, le=10)


def _truncate_prompt(prompt: str) -> str:
    """Trim *prompt* to :data:`MAX_PROMPT_BYTES` bytes if necessary.

    A warning is logged when truncation happens.
    """

    prompt_bytes = prompt.encode("utf-8")
    if len(prompt_bytes) > MAX_PROMPT_BYTES:
        logger.warning(
            "Prompt exceeds maximum length of %s bytes; truncating",
            MAX_PROMPT_BYTES,
        )
        prompt = prompt_bytes[:MAX_PROMPT_BYTES].decode("utf-8", errors="ignore")
    return prompt


def _is_local_hostname(hostname: str) -> bool:
    """Return ``True`` when *hostname* clearly refers to a local address."""

    if hostname.lower() == "localhost":
        return True
    try:
        ip_obj = ip_address(hostname)
    except ValueError:
        return False
    return ip_obj.is_loopback or ip_obj.is_private


def _normalise_allowed_host(value: str) -> str | None:
    """Normalise *value* from ``GPT_OSS_ALLOWED_HOSTS`` to bare host syntax."""

    trimmed = value.strip()
    if not trimmed:
        return None
    if trimmed.startswith("[") and trimmed.endswith("]"):
        trimmed = trimmed[1:-1]
    return trimmed.lower()


def _load_allowed_hosts() -> set[str]:
    """Return a set of hosts explicitly permitted for GPT OSS requests."""

    raw = os.getenv("GPT_OSS_ALLOWED_HOSTS")
    hosts = set(_DEFAULT_ALLOWED_HOSTS)
    if not raw:
        return hosts

    for part in raw.split(","):
        normalised = _normalise_allowed_host(part)
        if not normalised:
            continue

        if _is_local_hostname(normalised):
            hosts.add(normalised)
            continue

        try:
            addr_info = socket.getaddrinfo(
                normalised,
                None,
                family=socket.AF_UNSPEC,
            )
        except socket.gaierror as exc:
            logger.warning(
                "Ignoring GPT_OSS_ALLOWED_HOSTS entry %r: resolution failed (%s)",
                normalised,
                exc,
            )
            continue

        resolved_ips: set[str] = set()
        for info in addr_info:
            sockaddr = info[4]
            if not sockaddr:
                continue
            candidate = sockaddr[0]
            if isinstance(candidate, bytes):
                try:
                    candidate = candidate.decode()
                except UnicodeDecodeError:
                    continue
            resolved_ips.add(_normalise_ip(str(candidate)))

        if not resolved_ips:
            logger.warning(
                "Ignoring GPT_OSS_ALLOWED_HOSTS entry %r: no IP addresses resolved",
                normalised,
            )
            continue

        unsafe_ips = []
        for ip_text in resolved_ips:
            try:
                parsed_ip = ip_address(ip_text)
            except ValueError:
                unsafe_ips.append(ip_text)
                continue
            if not (parsed_ip.is_loopback or parsed_ip.is_private):
                unsafe_ips.append(ip_text)

        if unsafe_ips:
            logger.warning(
                "Ignoring GPT_OSS_ALLOWED_HOSTS entry %r: resolves to non-private IPs %s",
                normalised,
                sorted(unsafe_ips),
            )
            continue

        hosts.add(normalised)

    return hosts


def _validate_api_url(api_url: str, allowed_hosts: set[str]) -> tuple[str, set[str]]:
    parsed = urlparse(api_url)
    if not parsed.scheme:
        raise GPTClientError(
            "GPT_OSS_API URL must include a scheme (http or https)"
        )
    if not parsed.hostname:
        raise GPTClientError("GPT_OSS_API URL must include a hostname")
    if parsed.username or parsed.password:
        raise GPTClientError(
            "GPT_OSS_API URL must not contain embedded credentials"
        )

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise GPTClientError(
            f"GPT_OSS_API URL scheme {scheme!r} is not supported; use http or https"
        )

    hostname = parsed.hostname.lower()
    if hostname.startswith("[") and hostname.endswith("]"):
        hostname = hostname[1:-1]

    safe_api_url = sanitize_log_value(api_url)
    safe_hostname_for_log = sanitize_log_value(parsed.hostname or hostname)

    try:
        host_ip = ip_address(hostname)
    except ValueError:
        host_ip = None

    if host_ip is None and hostname not in allowed_hosts:
        raise GPTClientError(
            "GPT_OSS_API host must be explicitly allowed via GPT_OSS_ALLOWED_HOSTS"
        )

    resolution_failed = False
    try:
        addr_info = socket.getaddrinfo(
            parsed.hostname, None, family=socket.AF_UNSPEC
        )
        resolved_ips = {_normalise_ip(info[4][0]) for info in addr_info}
    except socket.gaierror as exc:
        if os.getenv("TEST_MODE") == "1":
            resolution_failed = True
            resolved_ips = {
                _normalise_ip("127.0.0.1"),
                _TEST_MODE_DNS_FALLBACK,
            }
        else:
            logger.error(
                "Failed to resolve GPT_OSS_API host %s: %s",
                safe_hostname_for_log,
                exc,
            )
            raise GPTClientError(
                f"GPT_OSS_API host {parsed.hostname!r} cannot be resolved"
            ) from exc

    is_local_host = _is_local_hostname(parsed.hostname)
    if scheme == "http":
        if not is_local_host:
            all_private = (
                all(
                    ip_address(ip).is_private or ip_address(ip).is_loopback
                    for ip in resolved_ips
                )
                if not resolution_failed
                else False
            )
            if not all_private:
                if _insecure_allowed():
                    logger.warning(
                        "Using insecure GPT_OSS_API URL %s",
                        safe_api_url,
                    )
                else:
                    raise GPTClientError(
                        "GPT_OSS_API URL must use HTTPS or resolve to a private address"
                    )
            elif _insecure_allowed():
                logger.warning(
                    "Using insecure GPT_OSS_API URL %s",
                    safe_api_url,
                )
        elif _insecure_allowed():
            logger.warning(
                "Using insecure GPT_OSS_API URL %s",
                safe_api_url,
            )

    if host_ip is not None and not (host_ip.is_loopback or host_ip.is_private):
        raise GPTClientError(
            "GPT_OSS_API host IP must be loopback or private when not allowlisted"
        )

    return hostname, resolved_ips


async def _fetch_response(
    client: httpx.Client | httpx.AsyncClient,
    prompt: str,
    url: str,
    hostname: str,
    allowed_ips: set[str],
) -> bytes:
    """Resolve hostname, verify IP and return response bytes."""
    skip_ip_verification = False
    allowed_ips = {_normalise_ip(ip) for ip in allowed_ips}
    allowed_for_check = allowed_ips
    safe_hostname_for_log = sanitize_log_value(hostname)
    if (
        os.getenv("TEST_MODE") == "1"
        and _TEST_MODE_DNS_FALLBACK in allowed_ips
    ):
        skip_ip_verification = True
        allowed_for_check = {
            ip for ip in allowed_ips if ip != _TEST_MODE_DNS_FALLBACK
        }

    try:
        loop = asyncio.get_running_loop()
        current_ips = {
            _normalise_ip(info[4][0])
            for info in await loop.getaddrinfo(
                hostname, None, family=socket.AF_UNSPEC
            )
        }
    except socket.gaierror as exc:
        if os.getenv("TEST_MODE") == "1":
            current_ips = allowed_ips
        else:
            logger.error(
                "Failed to resolve GPT_OSS_API host %s before request: %s",
                safe_hostname_for_log,
                exc,
            )
            raise GPTClientNetworkError("Failed to resolve GPT_OSS_API host") from exc

    if not current_ips & allowed_for_check:
        if skip_ip_verification:
            return await _stream_response(client, prompt, url)
        safe_current_ips = sanitize_log_value(sorted(current_ips))
        safe_allowed_ips = sanitize_log_value(sorted(allowed_ips))
        logger.error(
            "GPT_OSS_API host IP mismatch: %s resolved to %s, expected %s",
            safe_hostname_for_log,
            safe_current_ips,
            safe_allowed_ips,
        )
        raise GPTClientNetworkError("GPT_OSS_API host resolution mismatch")

    return await _stream_response(client, prompt, url)


async def _stream_response(
    client: httpx.Client | httpx.AsyncClient,
    prompt: str,
    url: str,
) -> bytes:
    """Stream response content regardless of client type."""

    if isinstance(client, httpx.AsyncClient):
        async with client.stream("POST", url, json={"prompt": prompt}) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("application/json"):
                raise GPTClientResponseError("Unexpected Content-Type from GPT OSS API")
            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > MAX_RESPONSE_BYTES:
                    raise GPTClientError("Response exceeds maximum length")
            return bytes(content)

    with client.stream("POST", url, json={"prompt": prompt}) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("application/json"):
            raise GPTClientResponseError("Unexpected Content-Type from GPT OSS API")
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > MAX_RESPONSE_BYTES:
                raise GPTClientError("Response exceeds maximum length")
        return bytes(content)


def _get_api_url_timeout() -> tuple[str, float, str, set[str]]:
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        message = (
            "GPT_OSS_API environment variable is not set. "
            "Set GPT_OSS_API to the base URL of the GPT OSS service, e.g. http://localhost:8003."
        )
        logger.error(message)
        raise GPTClientNetworkError(message)

    allowed_hosts = _load_allowed_hosts()
    hostname, allowed_ips = _validate_api_url(api_url, allowed_hosts)

    timeout = _parse_timeout(os.getenv("GPT_OSS_TIMEOUT"))

    api_url = api_url.rstrip("/")
    url = urljoin(api_url + "/", "v1/completions")
    return url, timeout, hostname, allowed_ips


def _parse_gpt_response(content: bytes) -> str:
    """Parse GPT OSS API JSON *content* and return the first completion text."""

    def _extract_text(choice: Any) -> str | None:
        """Return textual content from a completion *choice* if available."""

        if not isinstance(choice, Mapping):
            return None

        text = choice.get("text")
        if isinstance(text, str):
            return text

        def _coalesce_content(value: Any) -> str | None:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                        continue
                    if isinstance(item, Mapping):
                        part = item.get("text")
                        if isinstance(part, str):
                            parts.append(part)
                            continue
                        if isinstance(part, Mapping):
                            nested = part.get("value")
                            if isinstance(nested, str):
                                parts.append(nested)
                                continue
                        value_field = item.get("value")
                        if isinstance(value_field, str):
                            parts.append(value_field)
                if parts:
                    return "".join(parts)
            if isinstance(value, Mapping):
                nested_value = value.get("value")
                if isinstance(nested_value, str):
                    return nested_value
            return None

        for key in ("message", "delta", "content"):
            container = choice.get(key)
            if isinstance(container, Mapping):
                content_value = container.get("content")
                extracted = _coalesce_content(content_value)
                if extracted is not None:
                    return extracted
            else:
                extracted = _coalesce_content(container)
                if extracted is not None:
                    return extracted
        return None

    try:
        data = json.loads(content)
    except ValueError as exc:
        logger.exception("Invalid JSON response from GPT OSS API: %s", exc)
        raise GPTClientJSONError("Invalid JSON response from GPT OSS API") from exc

    try:
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            raise KeyError("choices")
        choice = choices[0]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning(
            "Unexpected response structure from GPT OSS API: %s | data: %r",
            exc,
            data,
        )
        raise GPTClientResponseError(
            "Unexpected response structure from GPT OSS API",
        ) from exc

    extracted = _extract_text(choice)
    if extracted is not None:
        return extracted

    logger.warning(
        "Unexpected response structure from GPT OSS API: missing completion text | data: %r",
        data,
    )
    raise GPTClientResponseError(
        "Unexpected response structure from GPT OSS API",
    )

def query_gpt(prompt: str) -> str:
    """Send *prompt* to the GPT OSS API and return the first completion text.

    The API endpoint is read from the ``GPT_OSS_API`` environment variable. If
    it is not set but ``OPENAI_API_KEY`` is configured the OpenAI Chat
    Completions API is used instead. Request timeout is read from
    ``GPT_OSS_TIMEOUT`` (seconds, default ``5``). Network errors are retried up
    to MAX_RETRIES times with exponential backoff between one and ten seconds
    before giving up. Prompts longer than :data:`MAX_PROMPT_BYTES` are truncated
    with a warning.
    """
    if bot_config.OFFLINE_MODE:
        return OfflineGPT.query(prompt)

    prompt = _truncate_prompt(prompt)
    api_url = os.getenv("GPT_OSS_API")
    if _should_use_openai(api_url):
        return _query_openai(prompt, api_url)

    url, timeout, hostname, allowed_ips = _get_api_url_timeout()

    # Maximum time to wait for the asynchronous task considering retries and backoff
    backoff_total = sum(min(2**i, 10) for i in range(MAX_RETRIES - 1))
    max_wait_time = timeout * MAX_RETRIES + backoff_total

    def _run_coro_in_thread(coro: Coroutine[Any, Any, bytes]) -> bytes:
        result: dict[str, Any] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:  # pragma: no cover - unexpected
                result["error"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(max_wait_time)
        if thread.is_alive():
            raise GPTClientError(
                "Timed out waiting for async task after maximum retries"
            )
        if "error" in result:
            raise result["error"]
        return result["value"]

    @retry(
        MAX_RETRIES,
        lambda attempt: min(2 ** (attempt - 1), 10),
    )
    def _post() -> bytes:
        async def _async_post() -> bytes:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                return await _fetch_response(
                    client, prompt, url, hostname, allowed_ips
                )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_async_post())
        else:
            return _run_coro_in_thread(_async_post())

    try:
        content = _post()
    except httpx.TimeoutException as exc:  # pragma: no cover - request timeout
        logger.exception("Timed out querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError(
            "GPT OSS API request timed out after maximum retries"
        ) from exc
    except GPTClientError:
        raise
    except httpx.HTTPError as exc:  # pragma: no cover - other network errors
        logger.exception("Error querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError("Failed to query GPT OSS API") from exc
    return _parse_gpt_response(content)


async def query_gpt_async(prompt: str) -> str:
    """Asynchronously send *prompt* to the GPT OSS API and return the first completion text.

    The API endpoint is taken from the ``GPT_OSS_API`` environment variable. If it
    is not set but ``OPENAI_API_KEY`` is configured, the OpenAI Chat Completions
    API is used instead. Request timeout is read from ``GPT_OSS_TIMEOUT`` (seconds,
    default ``5``). Network errors are retried up to MAX_RETRIES times with
    exponential backoff between one and ten seconds before giving up.

    Uses :class:`httpx.AsyncClient` for the HTTP request but mirrors the behaviour of
    :func:`query_gpt` including error handling and environment configuration.
    Prompts longer than :data:`MAX_PROMPT_BYTES` are truncated with a warning.
    """
    if bot_config.OFFLINE_MODE:
        return await OfflineGPT.query_async(prompt)

    prompt = _truncate_prompt(prompt)
    api_url = os.getenv("GPT_OSS_API")
    if _should_use_openai(api_url):
        return await asyncio.to_thread(_query_openai, prompt, api_url)

    url, timeout, hostname, allowed_ips = await asyncio.to_thread(
        _get_api_url_timeout
    )

    @retry(
        MAX_RETRIES,
        lambda attempt: min(2 ** (attempt - 1), 10),
    )
    async def _post() -> bytes:
        async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
            return await _fetch_response(client, prompt, url, hostname, allowed_ips)

    try:
        content = await _post()
    except httpx.TimeoutException as exc:  # pragma: no cover - request timeout
        logger.exception("Timed out querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError(
            "GPT OSS API request timed out after maximum retries"
        ) from exc
    except GPTClientError:
        raise
    except httpx.HTTPError as exc:  # pragma: no cover - other network errors
        logger.exception("Error querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError("Failed to query GPT OSS API") from exc
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.exception("Unexpected error querying GPT OSS API: %s", exc)
        raise GPTClientError("Unexpected error querying GPT OSS API") from exc
    return _parse_gpt_response(content)


async def query_gpt_json_async(prompt: str) -> dict:
    """Return JSON parsed from :func:`query_gpt_async` text output."""

    if bot_config.OFFLINE_MODE:
        return await OfflineGPT.query_json_async(prompt)

    text = await query_gpt_async(prompt)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON from GPT OSS API: %s", exc)
        return {"signal": "hold"}
    if not isinstance(data, dict):
        raise GPTClientResponseError("Unexpected response structure from GPT OSS API")
    required = {"signal", "tp_mult", "sl_mult"}
    if not required.issubset(data):
        missing = sorted(required - set(data))
        logger.error("Missing fields in GPT response: %s", missing)
        return {"signal": "hold"}
    return data


def parse_signal(payload: str) -> Signal:
    """Parse *payload* JSON into a :class:`Signal` instance.

    On any parsing or validation error a default ``hold`` signal is returned and
    a warning is logged.
    """

    try:
        data = json.loads(payload)
        return Signal.model_validate(data)
    except (ValueError, ValidationError, TypeError) as exc:  # pragma: no cover - invalid input
        logger.warning("Failed to parse signal: %s", exc)
        return Signal()
