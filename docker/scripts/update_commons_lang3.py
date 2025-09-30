"""Update the Ray commons-lang3 dependency to a vetted release."""

from __future__ import annotations

from pathlib import Path
import hashlib
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import requests


COMMONS_LANG3_URL = (
    "https://repo1.maven.org/maven2/org/apache/commons/commons-lang3/3.18.0/"
    "commons-lang3-3.18.0.jar"
)
COMMONS_LANG3_SHA256 = "4eeeae8d20c078abb64b015ec158add383ac581571cddc45c68f0c9ae0230720"
COMMONS_LANG3_ALLOWED_HOST = "repo1.maven.org"


def update_commons_lang3() -> None:
    """Download and verify commons-lang3 for the Ray runtime jars directory."""

    try:
        import ray  # type: ignore
    except Exception:
        print("Ray отсутствует, пропускаем обновление commons-lang3")
        return

    jars_dir = Path(ray.__file__).resolve().parent / "jars"
    jars_dir.mkdir(parents=True, exist_ok=True)

    for jar in jars_dir.glob("commons-lang3-*.jar"):
        jar.unlink()

    destination = jars_dir / "commons-lang3-3.18.0.jar"
    hasher = hashlib.sha256()

    parsed_url = urlsplit(COMMONS_LANG3_URL)
    if parsed_url.scheme != "https":
        raise RuntimeError(
            "Ожидалась схема https при скачивании commons-lang3, "
            f"получено: {parsed_url.scheme}"
        )
    if parsed_url.netloc != COMMONS_LANG3_ALLOWED_HOST:
        raise RuntimeError(
            "Получен неожиданный хост при скачивании commons-lang3: "
            f"{parsed_url.netloc}"
        )

    temp_path: Path | None = None

    try:
        with requests.Session() as session:
            adapter = requests.adapters.HTTPAdapter()
            session.mount("https://", adapter)
            session.trust_env = False

            with session.get(
                COMMONS_LANG3_URL,
                stream=True,
                timeout=30,
                allow_redirects=False,
            ) as response:
                if 300 <= response.status_code < 400:
                    location = response.headers.get("Location", "")
                    raise RuntimeError(
                        "commons-lang3 download unexpectedly redirected"
                        + (f" to {location}" if location else "")
                    )
                response.raise_for_status()
                with NamedTemporaryFile(
                    "wb", delete=False, dir=jars_dir
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        temp_file.write(chunk)
                        hasher.update(chunk)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    if temp_path is None:
        raise RuntimeError("Failed to create temporary commons-lang3 download")

    digest = hasher.hexdigest()
    if digest != COMMONS_LANG3_SHA256:
        destination.unlink(missing_ok=True)
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "SHA256 mismatch while downloading commons-lang3: expected "
            f"{COMMONS_LANG3_SHA256}, got {digest}"
        )

    try:
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

def main() -> int:
    update_commons_lang3()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
