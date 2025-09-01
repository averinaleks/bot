"""Lightweight stub of fastapi-csrf-protect for testing."""

from __future__ import annotations

import os
import pytest
import secrets


class CsrfProtectError(Exception):
    """Exception raised when CSRF validation fails."""


class CsrfProtect:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def load_config(func):
        return func

    def generate_csrf_tokens(self):
        """Generate a cryptographically secure CSRF token."""
        token = secrets.token_urlsafe()
        return token, token

    async def validate_csrf(self, request):
        token = request.headers.get("X-CSRF-Token")
        signed = request.cookies.get("fastapi-csrf-token")
        if token != signed:
            raise CsrfProtectError("CSRF token mismatch")


