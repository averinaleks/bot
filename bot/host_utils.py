"""Utility helpers for validating bind hosts without heavy imports."""

from __future__ import annotations

import ipaddress
import logging
import os
import re

from services.logging_utils import sanitize_log_value


logger = logging.getLogger("TradingBot")


def validate_host() -> str:
    """Return a safe loopback host derived from the ``HOST`` environment."""

    host = os.getenv("HOST", "").strip()
    if not host:
        logger.info("HOST не установлен, используется 127.0.0.1")
        return "127.0.0.1"

    original_value = host
    port = ""
    has_port = False

    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            suffix = host[end + 1 :].strip()
            has_port = bool(suffix)
            port = suffix
            host = host[1:end].strip()
    else:
        if host.count(":") == 1:
            host_part, port_part = host.split(":", 1)
            host = host_part.strip()
            port = port_part
            has_port = True

    if port:
        port = port.lstrip(":").strip()

    if has_port:
        if not port:
            raise ValueError(f"Не указан порт в HOST: {original_value}")
        if not port.isdigit():
            raise ValueError(f"Некорректный порт: {port}")
        port_num = int(port)
        if not 0 < port_num <= 65535:
            raise ValueError(f"Недопустимый порт: {port_num}")

    if host.lower() == "localhost":
        logger.info("HOST 'localhost' интерпретирован как 127.0.0.1")
        return "127.0.0.1"

    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]

    try:
        ip = ipaddress.ip_address(host)
        if ip.is_unspecified:
            raise ValueError(f"HOST '{ip}' запрещен")
    except ValueError:
        if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
            raise ValueError(f"Некорректный IP: {host}")
        safe_host = sanitize_log_value(host)
        logger.warning("HOST '%s' не локальный хост", safe_host)
        raise ValueError(f"HOST '{host}' не локальный хост")
    else:
        if not ip.is_loopback:
            safe_host = sanitize_log_value(host)
            logger.warning("HOST '%s' не локальный хост", safe_host)
            raise ValueError(f"HOST '{host}' не локальный хост")

    return host


__all__ = ["validate_host"]

