"""Minimal exceptiongroup stub for Python>=3.11."""

class BaseExceptionGroup(Exception):
    def __init__(self, message, exceptions):
        super().__init__(message)
        self.exceptions = exceptions

class ExceptionGroup(BaseExceptionGroup):
    pass

__all__ = ["BaseExceptionGroup", "ExceptionGroup"]
