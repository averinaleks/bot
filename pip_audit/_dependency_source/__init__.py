"""Minimal dependency source module for pip_audit stub."""


class DependencySourceError(Exception):
    """Raised when dependency metadata cannot be collected."""


class PipSource:
    """Placeholder dependency source."""

    def __init__(self, state=None):
        self.state = state
