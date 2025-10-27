"""Update the Ray commons-lang3 dependency to a vetted release."""

from __future__ import annotations

from pathlib import Path
import hashlib
import socket
from ipaddress import ip_address
from tempfile import TemporaryDirectory
from typing import Iterable, Tuple
from urllib.parse import SplitResult, urlsplit

import requests
from urllib3.util.retry import Retry


COMMONS_LANG3_URL = (
    "https://repo1.maven.org/maven2/org/apache/commons/commons-lang3/3.18.0/"
    "commons-lang3-3.18.0.jar"
)
COMMONS_LANG3_SHA256 = "4eeeae8d20c078abb64b015ec158add383ac581571cddc45c68f0c9ae0230720"
COMMONS_LANG3_ALLOWED_HOST = "repo1.maven.org"


def _iter_resolved_ips(hostname: str) -> Iterable[str]:
    """Yield textual representations of IPs for *hostname*.

    The helper normalises addresses returned by :func:`socket.getaddrinfo`
    so that IPv6 scope identifiers or byte strings are handled gracefully.
    """

    try:
        addr_info = socket.getaddrinfo(
            hostname,
            None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:  # pragma: no cover - network resolution varies
        raise RuntimeError(
            f"Не удалось разрешить хост скачивания commons-lang3: {hostname}"
        ) from exc

    for entry in addr_info:
        sockaddr = entry[4]
        if not sockaddr:
            continue
        candidate = sockaddr[0]
        if isinstance(candidate, bytes):
            try:
                candidate = candidate.decode("ascii")
            except UnicodeDecodeError:
                continue
        yield str(candidate)


def _validate_download_url(url: str) -> Tuple[SplitResult, set[str]]:
    """Validate *url* for commons-lang3 downloads and return the parse result."""

    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise RuntimeError(
            "Ожидалась схема https при скачивании commons-lang3, "
            f"получено: {parsed.scheme}"
        )
    hostname = parsed.hostname or parsed.netloc
    if hostname != COMMONS_LANG3_ALLOWED_HOST:
        raise RuntimeError(
            "Получен неожиданный хост при скачивании commons-lang3: "
            f"{parsed.netloc}"
        )

    unsafe_ips: set[str] = set()
    safe_ips: set[str] = set()
    for ip_text in _iter_resolved_ips(hostname):
        normalised = ip_text.split("%", 1)[0]
        try:
            ip_obj = ip_address(normalised)
        except ValueError:
            unsafe_ips.add(ip_text)
            continue
        if (
            ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            unsafe_ips.add(ip_text)
            continue

        safe_ips.add(ip_obj.compressed)

    if unsafe_ips:
        raise RuntimeError(
            "Хост commons-lang3 разрешился в небезопасные адреса: "
            + ", ".join(sorted(unsafe_ips))
        )

    if not safe_ips:
        raise RuntimeError(
            "Хост commons-lang3 не разрешился в допустимые публичные адреса"
        )

    return parsed, safe_ips


def update_commons_lang3() -> None:
    """Download and verify commons-lang3 for the Ray runtime jars directory."""

    try:
        import ray  # type: ignore
    except Exception:
        print("Ray отсутствует, пропускаем обновление commons-lang3")
        return

    jars_dir = Path(ray.__file__).resolve().parent / "jars"
    jars_dir.mkdir(parents=True, exist_ok=True)

    for jar in jars_dir.glob("commons-lang3-*.jar"):
        jar.unlink()

    destination = jars_dir / "commons-lang3-3.18.0.jar"
    hasher = hashlib.sha256()

    _, allowed_ips = _validate_download_url(COMMONS_LANG3_URL)

    with TemporaryDirectory(dir=jars_dir) as tmp_dir:
        temp_path = Path(tmp_dir) / destination.name

        with requests.Session() as session:
            retries = Retry(
                total=5,
                backoff_factor=1.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("HEAD", "GET"),
            )
            adapter = requests.adapters.HTTPAdapter(max_retries=retries)
            session.mount("https://", adapter)
            session.trust_env = False

            with session.get(
                COMMONS_LANG3_URL,
                stream=True,
                timeout=30,
                allow_redirects=False,
            ) as response:
                if 300 <= response.status_code < 400:
                    location = response.headers.get("Location", "")
                    raise RuntimeError(
                        "commons-lang3 download unexpectedly redirected"
                        + (f" to {location}" if location else "")
                    )
                response.raise_for_status()

                connection = getattr(response.raw, "_connection", None)
                peer_ip = None
                if connection is not None:
                    sock = getattr(connection, "sock", None)
                    if sock is not None and hasattr(sock, "getpeername"):
                        try:
                            peer_ip = sock.getpeername()[0]
                        except OSError:
                            peer_ip = None

                if peer_ip:
                    normalised_peer = peer_ip.split("%", 1)[0]
                    try:
                        peer_obj = ip_address(normalised_peer)
                    except ValueError:
                        raise RuntimeError(
                            "commons-lang3 download connected to malformed address "
                            f"{peer_ip}"
                        ) from None
                    if peer_obj.compressed not in allowed_ips:
                        raise RuntimeError(
                            "commons-lang3 download connected to unexpected address "
                            f"{peer_obj.compressed}"
                        )
                with temp_path.open("wb") as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        temp_file.write(chunk)
                        hasher.update(chunk)

        digest = hasher.hexdigest()
        if digest != COMMONS_LANG3_SHA256:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                "SHA256 mismatch while downloading commons-lang3: expected "
                f"{COMMONS_LANG3_SHA256}, got {digest}"
            )

        try:
            temp_path.replace(destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

def main() -> int:
    update_commons_lang3()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
