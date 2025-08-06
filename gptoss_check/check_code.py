import os, requests

def query_gpt(prompt: str) -> str:
    res = requests.post(
        os.getenv("GPT_OSS_API") + "/completions",
        json={"prompt": prompt, "max_tokens": 1024},
    )
    return res.json()["choices"][0]["text"]

with open("/repo/main.py") as f:
    code = f.read()

print("🧠 GPT-анализ:")
print(query_gpt(f"Анализируй код Python:\n{code}\nНайди ошибки, улучшения и уязвимости."))
