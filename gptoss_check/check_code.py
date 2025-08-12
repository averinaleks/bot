import os
import time

import requests
from requests.exceptions import RequestException
from pathlib import Path


def wait_for_api(api_url: str, timeout: int | None = None) -> None:
    """Ожидать готовности сервера GPT-OSS."""
    if timeout is None:
        try:
            timeout = int(os.getenv("GPT_OSS_WAIT_TIMEOUT", "30"))
        except ValueError:
            timeout = 30
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(api_url.rstrip("/"), timeout=5)
            return
        except RequestException:
            time.sleep(1)
    raise RuntimeError(f"Сервер GPT-OSS по адресу {api_url} не отвечает")



def query(prompt: str) -> str:
    """Отправить текст на сервер GPT-OSS и вернуть полученный ответ."""
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        raise RuntimeError("Переменная окружения GPT_OSS_API не установлена")

    wait_for_api(api_url)

    max_retries = 3
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                api_url.rstrip("/") + "/v1/completions",
                json={"prompt": prompt, "max_tokens": 1024},
                timeout=30,
            )
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as err:
                raise RuntimeError(
                    f"Некорректный JSON от GPT-OSS API: {err}"
                ) from err

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError(
                    "Некорректный формат ответа GPT-OSS API: отсутствует список 'choices'"
                )

            first_choice = choices[0]
            if not isinstance(first_choice, dict) or "text" not in first_choice:
                raise RuntimeError(
                    "Некорректный формат ответа GPT-OSS API: отсутствует ключ 'text'"
                )

            return first_choice["text"]
        except RequestException as err:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Ошибка запроса к GPT-OSS API: {err}"
                ) from err
            time.sleep(backoff)
            backoff *= 2


def send_telegram(msg: str) -> None:
    """Отправить сообщение в Telegram, если заданы токен и chat_id."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": msg[:4000]},
                timeout=15,
            )
        except RequestException as err:
            print(f"⚠️ Failed to send Telegram message: {err}")


def run() -> None:
    """Run GPT-OSS analysis for configured files."""
    paths_env = os.getenv("CHECK_CODE_PATH", "trading_bot.py")
    repo_root = Path(__file__).resolve().parent.parent
    for filename in (p.strip() for p in paths_env.split(",") if p.strip()):
        path = repo_root / filename
        if not path.exists():
            warning = f"⚠️ {filename} not found, skipping"
            print(warning)
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
            print(f"\n📄 {filename}\n{err}\n")
            send_telegram(f"📄 {filename}\n{err}")
            continue

        print(f"\n📄 {filename}\n{result}\n")
        send_telegram(f"📄 {filename}\n{result}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    run()
