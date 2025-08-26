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
