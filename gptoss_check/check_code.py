"""Run a quick GPT-based analysis on the trading bot code."""

from pathlib import Path

from gpt_client import query_gpt


code_path = Path(__file__).resolve().parents[1] / "trading_bot.py"
code = code_path.read_text(encoding="utf-8")

print("🧠 GPT-анализ:")
print(
    query_gpt(
        f"Анализируй код Python:\n{code}\nНайди ошибки, улучшения и уязвимости."
    )
)
