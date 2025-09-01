class CsrfProtectError(Exception):
    pass


class CsrfProtect:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def load_config(func):
        return func

    def generate_csrf_tokens(self):
        token = "token"
        return token, token

    async def validate_csrf(self, request):
        token = request.headers.get("X-CSRF-Token")
        signed = request.cookies.get("fastapi-csrf-token")
        if token != signed:
            raise CsrfProtectError("CSRF token mismatch")


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------
try:  # pragma: no cover - pytest may not be installed
    import pytest
    import sys

    @pytest.fixture
    def csrf_secret(monkeypatch):
        """Provide a fixed CSRF secret for tests."""

        secret = "test-secret"
        monkeypatch.setenv("CSRF_SECRET", secret)
        return secret

    def pytest_configure(config):  # pragma: no cover - plugin registration
        config.pluginmanager.register(sys.modules[__name__])
except Exception:  # pragma: no cover - minimal fallback
    pass
