from pathlib import Path
import os

if os.getenv("TEST_MODE") == "1":  # pragma: no cover - simple import guard
    import test_stubs as _test_stubs

    _test_stubs.apply()

# Allow loading submodules from project root
__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]
