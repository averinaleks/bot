from pathlib import Path
import os
import sys

# ``pytest`` imports this package before tests set ``TEST_MODE``.  Detect a
# pytest run by checking for the module and enable lightweight stubs so tests
# don't require heavy optional dependencies like ``httpx``.
if os.getenv("TEST_MODE") == "1" or "pytest" in sys.modules:
    os.environ.setdefault("TEST_MODE", "1")
    import test_stubs as _test_stubs

    _test_stubs.apply()

# Allow loading submodules from project root
__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]
