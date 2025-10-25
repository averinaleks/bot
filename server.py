import os
import asyncio
import hmac
import json
import logging
import re
from contextlib import asynccontextmanager
from collections.abc import Mapping
from typing import Any, List

try:  # pragma: no cover - exercised in CI when dependency installed
    from fastapi import FastAPI, HTTPException, Request, Response
except ImportError:  # pragma: no cover - lightweight fallback for test environment
    class HTTPException(Exception):
        """Fallback ``HTTPException`` mirroring FastAPI's interface."""

        def __init__(self, status_code: int, detail: Any | None = None) -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class Response:  # type: ignore[override]
        """Minimal ``Response`` replacement used during tests."""

        def __init__(self, content: Any | None = None, status_code: int = 200) -> None:
            self.content = content
            self.status_code = status_code

    class Request:  # type: ignore[override]
        """Simplified ``Request`` object for unit tests."""

        def __init__(
            self,
            headers: Mapping[str, str] | None = None,
            method: str = "GET",
            url: str = "http://test",
            body: bytes | None = None,
        ) -> None:
            self.headers = dict(headers or {})
            self.method = method
            self._url = url
            self._body = [body] if body else []

        async def stream(self):
            for chunk in self._body:
                yield chunk

        @property
        def url(self) -> str:
            return self._url

    class FastAPI:  # type: ignore[override]
        """Very small subset of :class:`fastapi.FastAPI` used in tests."""

        def __init__(self, *, lifespan=None):
            self.lifespan = lifespan

        def middleware(self, _type: str):
            def decorator(func):
                return func

            return decorator

        def post(self, _path: str):
            def decorator(func):
                return func

            return decorator

from bot.dotenv_utils import DOTENV_AVAILABLE, DOTENV_ERROR, load_dotenv
from services.logging_utils import sanitize_log_value

logger = logging.getLogger(__name__)


class ServerConfigurationError(RuntimeError):
    """Raised when mandatory server configuration is invalid."""

    pass

_MASKED_HEADER_VALUE = "***"
_SENSITIVE_BOUNDARY_RE = re.compile(r"(^|[^a-z0-9])(auth|key|cookie)([^a-z0-9]|$)")
_SENSITIVE_KEYWORDS = (
    "token",
    "secret",
    "session",
    "sessionid",
    "password",
    "apikey",
    "authorization",
)


def _mask_header_value(name: str, value: Any) -> str:
    """Return ``value`` masked when ``name`` looks sensitive."""

    lowered = str(name).lower()
    if _SENSITIVE_BOUNDARY_RE.search(lowered) or any(keyword in lowered for keyword in _SENSITIVE_KEYWORDS):
        return _MASKED_HEADER_VALUE
    return sanitize_log_value(value)


def _format_headers_for_log(headers: Mapping[str, Any] | None) -> str:
    """Format request ``headers`` into a sanitized string for logging."""

    if not headers:
        return "{}"
    pairs: list[str] = []
    for raw_name, raw_value in headers.items():
        safe_name = sanitize_log_value(raw_name)
        masked_value = _mask_header_value(raw_name, raw_value)
        pairs.append(f"{safe_name}: {masked_value}")
    return "{" + ", ".join(pairs) + "}"

if not DOTENV_AVAILABLE:
    logger.warning(
        "python-dotenv is not installed; continuing without loading .env files (%s)",
        DOTENV_ERROR,
    )
try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError as exc:  # pragma: no cover - dependency required
    raise RuntimeError(
        "pydantic is required. Install it with 'pip install pydantic'."
    ) from exc

API_KEYS: set[str] = set()

try:  # pragma: no cover - handled in tests
    from fastapi_csrf_protect import CsrfProtect  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - dependency required
    raise RuntimeError(
        "fastapi-csrf-protect is required. Install it with 'pip install fastapi-csrf-protect'."
    ) from exc
try:  # pragma: no cover - handled in tests
    from fastapi_csrf_protect import CsrfProtectError  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - export varies by version
    try:
        from fastapi_csrf_protect.exceptions import CsrfProtectError  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - dependency required
        raise RuntimeError(
            "fastapi-csrf-protect is required. Install it with 'pip install fastapi-csrf-protect'."
        ) from exc

load_dotenv()


class ModelManager:
    """Manage loading and inference model state."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.device = None
        self.torch = None

    def load_model(self) -> str:
        """Load the tokenizer and model into instance attributes.

        Returns "primary" if the main model is loaded, "fallback" if the
        fallback model is loaded instead.
        """

        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError(
                "torch is required to load the model. Install it with 'pip install torch'."
            ) from exc

        device_local = "cuda" if torch_module.cuda.is_available() else "cpu"

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to load the model. Install it with 'pip install transformers'."
            ) from exc

        model_name = os.getenv("GPT_MODEL", "openai/gpt-oss-20b")
        fallback_model = os.getenv("GPT_MODEL_FALLBACK", "sshleifer/tiny-gpt2")
        model_revision = os.getenv(
            "GPT_MODEL_REVISION", "10e9d713f8e4a9281c59c40be6c58537480635ea"
        )
        fallback_revision = os.getenv(
            "GPT_MODEL_FALLBACK_REVISION", "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
        )
        if not re.fullmatch(r"[0-9a-f]{40}", model_revision, re.IGNORECASE):
            raise ValueError("GPT_MODEL_REVISION must be a 40-character SHA commit")
        if not re.fullmatch(r"[0-9a-f]{40}", fallback_revision, re.IGNORECASE):
            raise ValueError(
                "GPT_MODEL_FALLBACK_REVISION must be a 40-character SHA commit"
            )
        cache_dir = os.getenv("MODEL_CACHE_DIR")

        load_tokenizer = AutoTokenizer.from_pretrained
        load_model = AutoModelForCausalLM.from_pretrained

        def _load_pair(
            name: str,
            revision: str,
            *,
            local_only: bool,
        ) -> tuple[Any, Any]:
            tokenizer_local = load_tokenizer(
                name,
                revision=revision,
                trust_remote_code=False,
                cache_dir=cache_dir,
                local_files_only=local_only,
            )
            model_local = load_model(
                name,
                revision=revision,
                trust_remote_code=False,
                cache_dir=cache_dir,
                local_files_only=local_only,
            ).to(device_local)
            return tokenizer_local, model_local

        try:
            tokenizer_local, model_local = _load_pair(
                model_name,
                model_revision,
                local_only=False,
            )
        except (OSError, ValueError) as exc:
            logging.exception("Failed to load model '%s': %s", model_name, exc)
            try:
                tokenizer_local, model_local = _load_pair(
                    model_name,
                    model_revision,
                    local_only=True,
                )
            except (OSError, ValueError) as exc2:
                logging.exception(
                    "Failed to load model '%s' from local cache: %s", model_name, exc2
                )
            else:
                self.tokenizer = tokenizer_local
                self.model = model_local
                self.device = device_local
                self.torch = torch_module
                logging.info("Loaded model '%s' from local cache", model_name)
                return "primary"
        else:
            self.tokenizer = tokenizer_local
            self.model = model_local
            self.device = device_local
            self.torch = torch_module
            return "primary"

        try:
            tokenizer_local, model_local = _load_pair(
                fallback_model,
                fallback_revision,
                local_only=False,
            )
        except (OSError, ValueError) as exc:
            logging.exception(
                "Failed to load fallback model '%s': %s", fallback_model, exc
            )
            try:
                tokenizer_local, model_local = _load_pair(
                    fallback_model,
                    fallback_revision,
                    local_only=True,
                )
            except (OSError, ValueError) as exc2:
                logging.exception(
                    "Failed to load fallback model '%s' from local cache: %s",
                    fallback_model,
                    exc2,
                )
                raise RuntimeError(
                    "Failed to load both primary and fallback models"
                ) from exc2
            else:
                self.tokenizer = tokenizer_local
                self.model = model_local
                self.device = device_local
                self.torch = torch_module
                logging.info(
                    "Loaded fallback model '%s' from local cache", fallback_model
                )
                return "fallback"
        else:
            self.tokenizer = tokenizer_local
            self.model = model_local
            self.device = device_local
            self.torch = torch_module
            logging.info("Loaded fallback model '%s'", fallback_model)
            return "fallback"

    async def load_model_async(self) -> None:
        """Asynchronously load the model without blocking the event loop.

        Raises:
            RuntimeError: If both primary and fallback models fail to load.
        """
        await asyncio.to_thread(self.load_model)


model_manager = ModelManager()
model_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    API_KEYS.clear()
    API_KEYS.update(
        {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
    )
    if not API_KEYS:
        logging.error(
            "API_KEYS is empty; all requests will be rejected. Set the API_KEYS environment variable.",
        )
        raise RuntimeError("API_KEYS environment variable is required")
    await model_manager.load_model_async()
    yield


app = FastAPI(lifespan=lifespan)


class CsrfSettings(BaseModel):
    secret_key: str

def _resolve_csrf_secret() -> str:
    env_secret = os.getenv("CSRF_SECRET")
    if env_secret:
        return env_secret
    logging.error(
        "CSRF_SECRET is required; set a strong random value for production deployments.",
    )
    raise RuntimeError("CSRF_SECRET environment variable is required")


@CsrfProtect.load_config
def get_csrf_config() -> CsrfSettings:
    """Return CSRF settings for FastAPI CsrfProtect."""
    return CsrfSettings(secret_key=_resolve_csrf_secret())


# Validate configuration at import time so misconfigured deployments fail fast.
_resolve_csrf_secret()


csrf_protect = CsrfProtect()


ALLOWED_HOSTS = {"127.0.0.1", "::1"}
host = os.getenv(
    "HOST", "127.0.0.1"
)  # Bind to localhost to avoid exposing the service externally
port_str = os.getenv("PORT", "8000")
try:
    port = int(port_str)
except ValueError as exc:  # pragma: no cover - exercised in tests
    safe_port = sanitize_log_value(port_str)
    logger.error("Invalid PORT value '%s'; must be an integer", safe_port)
    raise ServerConfigurationError(
        f"Invalid PORT value '{safe_port}'; must be an integer"
    ) from exc

if not (1 <= port <= 65_535):
    safe_port = sanitize_log_value(port_str)
    logger.error(
        "Invalid PORT value '%s'; must be between 1 and 65535",
        safe_port,
    )
    raise ServerConfigurationError(
        f"Invalid PORT value '{safe_port}'; must be between 1 and 65535"
    )

if host not in ALLOWED_HOSTS:
    safe_host = sanitize_log_value(host)
    logger.error("Invalid HOST '%s'; allowed values are %s", safe_host, ALLOWED_HOSTS)
    raise ServerConfigurationError(
        f"Invalid HOST '{safe_host}'; allowed values are {sorted(ALLOWED_HOSTS)}"
    )

MAX_REQUEST_BYTES = 1 * 1024 * 1024  # 1 MB


@app.middleware("http")
async def enforce_csrf(request: Request, call_next):
    if request.method == "POST":
        safe_method = sanitize_log_value(request.method)
        safe_url = sanitize_log_value(str(request.url))
        safe_headers = _format_headers_for_log(getattr(request, "headers", None))
        forbidden_payload = json.dumps({"detail": "CSRF token missing or invalid"})

        def _csrf_error_response() -> Response:
            response = Response(content=forbidden_payload, status_code=403)
            try:
                response.media_type = "application/json"  # type: ignore[assignment]
            except Exception:  # pragma: no cover - fallback ``Response`` lacks attribute
                setattr(response, "media_type", "application/json")
            return response

        cookies = getattr(request, "cookies", {}) or {}
        header_name = getattr(csrf_protect, "_header_name", "X-CSRF-Token")
        cookie_name = getattr(csrf_protect, "_cookie_key", "fastapi-csrf-token")
        header_token = None
        if hasattr(request, "headers") and request.headers is not None:
            header_token = request.headers.get(header_name)
        cookie_token = cookies.get(cookie_name) if isinstance(cookies, Mapping) else None
        tokens_present = bool(header_token) and bool(cookie_token)
        try:
            await csrf_protect.validate_csrf(request)
        except CsrfProtectError as exc:
            logging.warning(
                "CSRF validation failed: method=%s url=%s headers=%s error=%s",
                safe_method,
                safe_url,
                safe_headers,
                sanitize_log_value(str(exc)),
            )
            return _csrf_error_response()
        except Exception as exc:  # pragma: no cover - unexpected library failure
            logging.exception(
                "Unexpected CSRF error: method=%s url=%s headers=%s error=%s",
                safe_method,
                safe_url,
                safe_headers,
                sanitize_log_value(str(exc)),
            )
            error_payload = json.dumps({"detail": "Internal Server Error"})
            response = Response(content=error_payload, status_code=500)
            try:
                response.media_type = "application/json"  # type: ignore[assignment]
            except Exception:  # pragma: no cover - fallback ``Response`` lacks attribute
                setattr(response, "media_type", "application/json")
            return response
        if not tokens_present:
            missing_parts: list[str] = []
            if not cookie_token:
                missing_parts.append("cookie")
            if not header_token:
                missing_parts.append("header")
            logging.warning(
                "CSRF validation failed: method=%s url=%s headers=%s error=%s",
                safe_method,
                safe_url,
                safe_headers,
                sanitize_log_value(
                    f"missing csrf {' and '.join(missing_parts)}"
                ),
            )
            return _csrf_error_response()
    return await call_next(request)


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    safe_method = sanitize_log_value(request.method)
    safe_url = sanitize_log_value(str(request.url))
    safe_headers = _format_headers_for_log(getattr(request, "headers", None))
    if not API_KEYS:
        logging.warning(
            "Rejecting request: authentication configuration is missing. method=%s url=%s headers=%s",
            safe_method,
            safe_url,
            safe_headers,
        )
        return Response(status_code=401)
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        logging.warning(
            "Unauthorized request: method=%s url=%s headers=%s",
            safe_method,
            safe_url,
            safe_headers,
        )
        return Response(status_code=401)
    token = auth[7:]
    for key in API_KEYS:
        if hmac.compare_digest(token, key):
            break
    else:
        logging.warning(
            "Unauthorized request: method=%s url=%s headers=%s",
            safe_method,
            safe_url,
            safe_headers,
        )
        return Response(status_code=401)
    return await call_next(request)


def generate_text(
    manager: ModelManager,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_new_tokens: int = 16,
) -> str:
    """Generate text from *prompt* using the loaded model."""
    if manager.tokenizer is None or manager.model is None:
        raise RuntimeError("Model is not loaded")
    inputs = manager.tokenizer(prompt, return_tensors="pt").to(manager.device)
    with manager.torch.no_grad():
        outputs = manager.model.generate(
            **inputs,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    text = manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return text


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(16, ge=1, le=512)


class CompletionRequest(BaseModel):
    prompt: str
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(16, ge=1, le=512)


def _parse_model_json(model_cls: type[BaseModel], data: bytes) -> BaseModel:
    """Parse *data* into *model_cls* supporting Pydantic v1 and v2.

    Pydantic v2 exposes :meth:`model_validate_json` while v1 uses
    :meth:`parse_raw`.  Tests run with the lightweight v1 dependency, so this
    helper allows the server to operate regardless of the installed version.
    """

    if hasattr(model_cls, "model_validate_json"):
        return getattr(model_cls, "model_validate_json")(data)  # type: ignore[call-arg]
    return model_cls.parse_raw(data)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")
    try:
        req = _parse_model_json(ChatRequest, bytes(body))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())
    if model_manager.tokenizer is None or model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    prompt = "\n".join(message.content for message in req.messages)
    if model_lock.locked():
        raise HTTPException(status_code=429, detail="Model is busy")
    async with model_lock:
        text = await asyncio.to_thread(
            generate_text,
            model_manager,
            prompt,
            temperature=req.temperature,
            max_new_tokens=req.max_tokens,
        )
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


@app.post("/v1/completions")
async def completions(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")
    try:
        req = _parse_model_json(CompletionRequest, bytes(body))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())
    if model_manager.tokenizer is None or model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if model_lock.locked():
        raise HTTPException(status_code=429, detail="Model is busy")
    async with model_lock:
        text = await asyncio.to_thread(
            generate_text,
            model_manager,
            req.prompt,
            temperature=req.temperature,
            max_new_tokens=req.max_tokens,
        )
    return {"choices": [{"text": text}]}


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    import uvicorn

    uvicorn.run("server:app", host=host, port=port)
