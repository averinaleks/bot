"""Run a basic code review using the GPT OSS API."""
from gpt_client import query_gpt


def main() -> None:
    """Send a simple request to the GPT OSS service."""
    prompt = "Review the repository for style or logical issues."
    print(query_gpt(prompt))


if __name__ == "__main__":
    main()
