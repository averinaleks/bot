import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import httpx


logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised via docker compose integration
    from http_client import get_httpx_client as _get_httpx_client
except Exception as import_error:  # pragma: no cover - fallback for CI container

    @contextmanager
    def get_httpx_client(timeout: float = 10.0, **kwargs):
        """Provide a minimal ``httpx.Client`` when the shared helper is unavailable."""

        kwargs.setdefault("timeout", timeout)
        kwargs.setdefault("trust_env", False)
        client = httpx.Client(**kwargs)
        try:
            yield client
        finally:
            client.close()

    logger.debug("Using fallback httpx client for GPT-OSS check: %s", import_error)
else:  # pragma: no cover - import succeeds in fully configured environments
    get_httpx_client = _get_httpx_client

try:  # pragma: no cover - exercised via docker compose integration
    from gpt_client import GPTClientError as _GPTClientError, query_gpt as _query_gpt
except Exception as import_error_gpt:  # pragma: no cover - fallback for CI container

    class GPTClientError(RuntimeError):
        """Fallback GPT client error used when trading bot modules are unavailable."""

    def _extract_choice_text(choice: object) -> str | None:
        """Return textual content from a single GPT-OSS choice payload."""

        if not isinstance(choice, dict):
            return None

        text = choice.get("text")
        if isinstance(text, str) and text:
            return text

        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                pieces: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        fragment = item.get("text")
                    else:
                        fragment = item.get("content")
                    if isinstance(fragment, str) and fragment:
                        pieces.append(fragment)
                if pieces:
                    return "".join(pieces)

        content = choice.get("content")
        if isinstance(content, str) and content:
            return content

        return None

    def query_gpt(prompt: str) -> str:
        api_url = os.getenv("GPT_OSS_API")
        if not api_url:
            raise RuntimeError("Переменная окружения GPT_OSS_API не установлена")

        completions_url = _build_completions_url(api_url)
        payload = {"prompt": prompt}
        model = os.getenv("GPT_OSS_MODEL")
        if model:
            payload["model"] = model

        with get_httpx_client(timeout=30, trust_env=False) as client:
            response = client.post(completions_url, json=payload)
            try:
                response.raise_for_status()
                data = response.json()
            except ValueError as exc:  # pragma: no cover - unexpected API response
                raise RuntimeError("Некорректный JSON-ответ от GPT-OSS") from exc
            finally:
                response.close()

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Некорректный ответ GPT-OSS: {data!r}")

        text = _extract_choice_text(choices[0])
        if text is None:
            raise RuntimeError(f"Некорректный ответ GPT-OSS: {data!r}")
        return text

    logger.debug("Using fallback GPT client for GPT-OSS check: %s", import_error_gpt)
else:  # pragma: no cover - import succeeds in fully configured environments
    GPTClientError = _GPTClientError
    query_gpt = _query_gpt


def _build_completions_url(api_url: str) -> str:
    """Normalize the GPT-OSS URL so requests always target ``/v1/completions``."""

    parsed = urlparse(api_url)
    path = parsed.path
    v1_index = path.find("/v1")
    if v1_index != -1:
        path = path[:v1_index]
    base = urlunparse(parsed._replace(path=path.rstrip("/")))
    return urljoin(base.rstrip("/") + "/", "v1/completions")


def wait_for_api(api_url: str, timeout: int | None = None) -> None:
    """Ожидать готовности сервера GPT-OSS."""
    if timeout is None:
        try:
            timeout = int(os.getenv("GPT_OSS_WAIT_TIMEOUT", "300"))
        except ValueError:
            timeout = 300
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with get_httpx_client(timeout=5, trust_env=False) as client:
                health_url = _build_completions_url(api_url)
                response = client.post(health_url, json={})
                try:
                    response.raise_for_status()
                finally:
                    response.close()
            return
        except httpx.HTTPError:
            time.sleep(1)
    raise RuntimeError(f"Сервер GPT-OSS по адресу {api_url} не отвечает")



def query(prompt: str) -> str:
    """Отправить текст на сервер GPT-OSS и вернуть полученный ответ."""
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        raise RuntimeError("Переменная окружения GPT_OSS_API не установлена")

    max_retries = 3
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            return query_gpt(prompt)
        except GPTClientError as err:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Ошибка запроса к GPT-OSS API: {err}"
                ) from err
            time.sleep(backoff)
            backoff *= 2


def send_telegram(msg: str) -> None:
    """Отправить сообщение в Telegram, если заданы токен и chat_id."""
    if os.getenv("TEST_MODE") == "1":
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            with get_httpx_client(timeout=15, trust_env=False) as client:
                response = client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={"chat_id": chat_id, "text": msg[:4000]},
                )
                response.close()
        except httpx.HTTPError as err:
            logger.warning("⚠️ Failed to send Telegram message: %s", err)


def run() -> None:
    """Run GPT-OSS analysis for configured files."""
    paths_env = os.getenv("CHECK_CODE_PATH", "trading_bot.py")
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        warning = "Переменная окружения GPT_OSS_API не установлена, проверка пропущена"
        logger.warning(warning)
        send_telegram(warning)
        return
    try:
        wait_for_api(api_url)
    except RuntimeError as err:
        warning = f"{err}, проверка пропущена"
        logger.warning(warning)
        send_telegram(warning)
        return
    repo_root = Path(__file__).resolve().parent.parent
    for filename in (p.strip() for p in paths_env.split(",") if p.strip()):
        path = repo_root / filename
        if not path.exists():
            warning = f"⚠️ {filename} not found, skipping"
            logger.warning(warning)
            send_telegram(warning)
            continue

        with open(path, encoding="utf-8") as f:
            code = f.read()

        prompt = (
            "Проанализируй код Python. Выяви ошибки, уязвимости, улучшения. "
            "Объясни сигналы стратегии:\n" + code
        )
        try:
            result = query(prompt)
        except RuntimeError as err:
            logger.error("\n📄 %s\n%s\n", filename, err)
            send_telegram(f"📄 {filename}\n{err}")
            continue

        logger.info("\n📄 %s\n%s\n", filename, result)
        send_telegram(f"📄 {filename}\n{result}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    run()
