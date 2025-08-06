"""Run a simple GPT OSS check for a given source file."""

from pathlib import Path
import os

from bot.gpt_client import query_gpt


def main() -> None:
    """Read the target file and print GPT analysis output."""

    root = Path(__file__).resolve().parent.parent
    code_path = Path(os.getenv("CHECK_CODE_PATH", "trading_bot.py"))
    if not code_path.is_absolute():
        code_path = root / code_path

    code = code_path.read_text(encoding="utf-8")
    print("🧠 GPT-анализ:")
    print(
        query_gpt(
            f"Анализируй код Python:\n{code}\nНайди ошибки, улучшения и уязвимости."
        )
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
