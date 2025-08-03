import time
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
    failed attempt up to ``max_delay``. Raises ``AssertionError`` if the service
    does not become ready within ``timeout`` seconds.
    """
    deadline = time.time() + timeout
    delay = initial_delay
    while True:
        try:
            resp = requests.get(url, timeout=0.2)
            if resp.status_code == 200:
                return resp
        except Exception:
            pass
        if time.time() >= deadline:
            break
        time.sleep(delay)
        delay = min(delay * backoff, max_delay)
    raise AssertionError(
        f"Service at {url} did not become ready within {timeout} seconds"
    )
