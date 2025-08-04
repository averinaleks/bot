import time
import random
import socket
import multiprocessing
from contextlib import contextmanager

import requests


def wait_for_service(
    url: str,
    timeout: float = 5.0,
    *,
    initial_delay: float = 0.1,
    max_delay: float = 1.0,
    backoff: float = 2.0,
) -> requests.Response:
    """Poll ``url`` until it responds with HTTP 200 or until ``timeout`` expires.

    The function retries requests with exponential backoff. ``initial_delay`` sets
    the first sleep interval, which is multiplied by ``backoff`` after each
    failed attempt up to ``max_delay``. Raises ``RuntimeError`` if the service
    does not become ready within ``timeout`` seconds.
    """
    deadline = time.time() + timeout
    delay = initial_delay
    last_exc: Exception | None = None
    while True:
        try:
            resp = requests.get(url, timeout=0.2)
            if resp.status_code == 200:
                return resp
        except Exception as exc:  # pragma: no cover
            last_exc = exc
        if time.time() >= deadline:
            break
        time.sleep(delay)
        delay = min(delay * backoff, max_delay)
    raise RuntimeError(
        f"Service at {url} did not become ready within {timeout} seconds"
    ) from last_exc


def get_free_port(retries: int = 10, low: int = 20000, high: int = 50000) -> int:
    """Return an available port on localhost.

    Ports are chosen randomly from the range ``low``-``high``. The function
    retries to mitigate bind conflicts when tests run in parallel.
    """
    for _ in range(retries):
        port = random.randint(low, high)
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("Could not allocate free port")


ctx = multiprocessing.get_context("spawn")


@contextmanager
def service_process(
    proc: multiprocessing.Process,
    *,
    url: str,
    start_timeout: float = 5.0,
    join_timeout: float = 0.2,
):
    """Start a service ``proc`` and ensure cleanup.

    ``wait_for_service`` verifies availability of ``url`` within ``start_timeout``.
    On exit, the process is terminated and joined with ``join_timeout`` to prevent
    hanging tests.
    """
    proc.start()
    try:
        resp = wait_for_service(url, timeout=start_timeout)
        yield resp
    finally:
        proc.terminate()
        proc.join(join_timeout)
