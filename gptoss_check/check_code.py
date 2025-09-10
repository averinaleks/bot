import os
import time
import logging
from pathlib import Path

import httpx

from http_client import get_httpx_client

from gpt_client import GPTClientError, query_gpt


logger = logging.getLogger(__name__)


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
                response = client.get(api_url.rstrip("/"))
                response.raise_for_status()
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
