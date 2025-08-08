import os
import time
import requests
from requests.exceptions import RequestException


def query(prompt: str) -> str:
    """Отправить текст на сервер GPT-OSS и вернуть полученный ответ."""
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        raise RuntimeError("Переменная окружения GPT_OSS_API не установлена")

    max_retries = 3
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                api_url.rstrip("/") + "/completions",
                json={"prompt": prompt, "max_tokens": 1024},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except RequestException as err:
            if attempt == max_retries:
                raise RuntimeError(f"Ошибка запроса к GPT-OSS API: {err}") from err
            time.sleep(backoff)
            backoff *= 2


def send_telegram(msg: str) -> None:
    """Отправить сообщение в Telegram, если заданы токен и chat_id."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": msg[:4000]},
            timeout=15,
        )


# Список файлов, которые нужно анализировать
files = ("main.py", "strategy.py", "utils.py")

for filename in files:
    path = f"/repo/{filename}"
    if os.path.exists(path):
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
