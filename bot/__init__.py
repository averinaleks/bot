import os
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

# ``pytest`` imports this package before tests set ``TEST_MODE``.  Detect a
# pytest run by checking for the module and enable lightweight stubs so tests
# don't require heavy optional dependencies like ``httpx``.
if os.getenv("TEST_MODE") == "1" or "pytest" in sys.modules:
    os.environ.setdefault("TEST_MODE", "1")
    import test_stubs as _test_stubs

    _test_stubs.apply()
    test_stubs = _test_stubs

# Allow loading submodules from project root
__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]

if TYPE_CHECKING:
    import config as config
else:
    _config_module: ModuleType | None
    try:
        _config_module = importlib.import_module("config")
    except ModuleNotFoundError:
        _config_module = None
    config = cast(ModuleType | None, _config_module)
