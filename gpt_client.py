import os
import requests


def query_gpt(prompt: str) -> str:
    """Send *prompt* to the GPT OSS API and return the first completion text.

    The API endpoint is read from the ``GPT_OSS_API`` environment variable.
    If it is not set, ``http://localhost:8003`` is used.
    """
    url = os.getenv("GPT_OSS_API", "http://localhost:8003")
    response = requests.post(url, json={"prompt": prompt})
    response.raise_for_status()
    data = response.json()
    try:
        return data["completions"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""
