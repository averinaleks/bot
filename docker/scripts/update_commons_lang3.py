"""Update the Ray commons-lang3 dependency to a vetted release."""

from __future__ import annotations

from pathlib import Path
import hashlib
from urllib.request import urlopen


COMMONS_LANG3_URL = (
    "https://repo1.maven.org/maven2/org/apache/commons/commons-lang3/3.18.0/"
    "commons-lang3-3.18.0.jar"
)
COMMONS_LANG3_SHA256 = "4eeeae8d20c078abb64b015ec158add383ac581571cddc45c68f0c9ae0230720"


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

    with urlopen(COMMONS_LANG3_URL) as response, destination.open("wb") as fh:
        for chunk in iter(lambda: response.read(8192), b""):
            fh.write(chunk)
            hasher.update(chunk)

    digest = hasher.hexdigest()
    if digest != COMMONS_LANG3_SHA256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "SHA256 mismatch while downloading commons-lang3: expected "
            f"{COMMONS_LANG3_SHA256}, got {digest}"
        )


def main() -> int:
    update_commons_lang3()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
