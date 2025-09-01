from pathlib import Path

# Allow loading submodules from project root
__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]
