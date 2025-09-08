"""Minimal iniconfig stub for pytest.

This module provides just enough structure for importing ``pytest`` in
subprocesses used by the integration tests. The real ``iniconfig`` package
exposes a :class:`IniConfig` class which ``pytest`` imports at module load
time. Our original stub only defined :class:`SectionWrapper`, leading to an
``ImportError`` when ``pytest`` attempted to import ``IniConfig`` inside child
processes spawned during the integration tests. Defining a lightweight
``IniConfig`` here satisfies that import without pulling in the third-party
dependency.
"""


class SectionWrapper(dict):
    """Simplified stand-in for ``iniconfig.SectionWrapper``."""


class IniConfig:  # pragma: no cover - trivial placeholder
    """Bare-bones ``IniConfig`` implementation used only for imports.

    The real class parses INI-style configuration files, but the integration
    tests merely require that the symbol exists so that ``pytest`` can be
    imported in spawned subprocesses. No parsing functionality is necessary
    for our tests, so this stub only initializes an empty ``sections``
    dictionary.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        self.sections: dict[str, SectionWrapper] = {}


__all__ = ["SectionWrapper", "IniConfig"]
