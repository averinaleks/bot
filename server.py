import os
import sys
import logging
import asyncio
import hmac
import re
import secrets
from typing import List, Mapping
from contextlib import asynccontextmanager

from bot.dotenv_utils import DOTENV_AVAILABLE, DOTENV_ERROR, load_dotenv
from services.logging_utils import sanitize_log_value

if not DOTENV_AVAILABLE:
    raise RuntimeError(
        "python-dotenv is required. Install it with 'pip install python-dotenv'."
    ) from DOTENV_ERROR

from fastapi import FastAPI, HTTPException, Request, Response
try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError as exc:  # pragma: no cover - dependency required
    raise RuntimeError(
        "pydantic is required. Install it with 'pip install pydantic'."
    ) from exc

API_KEYS: set[str] = set()

try:  # pragma: no cover - handled in tests
    from fastapi_csrf_protect import CsrfProtect, CsrfProtectError
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

        try:
            tokenizer_local = AutoTokenizer.from_pretrained(  # nosec  # revision validated above
                model_name,
                revision=model_revision,
                trust_remote_code=False,
                cache_dir=cache_dir,
            )
            model_local = AutoModelForCausalLM.from_pretrained(  # nosec  # revision validated above
                model_name,
                revision=model_revision,
                trust_remote_code=False,
                cache_dir=cache_dir,
            ).to(
                device_local
            )
        except (OSError, ValueError) as exc:
            logging.exception("Failed to load model '%s': %s", model_name, exc)
            try:
                tokenizer_local = AutoTokenizer.from_pretrained(  # nosec  # revision validated above
                    model_name,
                    revision=model_revision,
                    trust_remote_code=False,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                model_local = AutoModelForCausalLM.from_pretrained(  # nosec  # revision validated above
                    model_name,
                    revision=model_revision,
                    trust_remote_code=False,
                    cache_dir=cache_dir,
                    local_files_only=True,
                ).to(
                    device_local
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
            tokenizer_local = AutoTokenizer.from_pretrained(  # nosec  # revision validated above
                fallback_model,
                revision=fallback_revision,
                trust_remote_code=False,
                cache_dir=cache_dir,
            )
            model_local = AutoModelForCausalLM.from_pretrained(  # nosec  # revision validated above
                fallback_model,
                revision=fallback_revision,
                trust_remote_code=False,
                cache_dir=cache_dir,
                local_files_only=True,
            ).to(device_local)
        except (OSError, ValueError) as exc:
            logging.exception(
                "Failed to load fallback model '%s': %s", fallback_model, exc
            )
            try:
                tokenizer_local = AutoTokenizer.from_pretrained(  # nosec  # revision validated above
                    fallback_model,
                    revision=fallback_revision,
                    trust_remote_code=False,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                model_local = AutoModelForCausalLM.from_pretrained(  # nosec  # revision validated above
                    fallback_model,
                    revision=fallback_revision,
                    trust_remote_code=False,
                    cache_dir=cache_dir,
                    local_files_only=True,
                ).to(device_local)
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


_EPHEMERAL_CSRF_SECRET: str | None = None


def _resolve_csrf_secret() -> str:
    env_secret = os.getenv("CSRF_SECRET")
    if env_secret:
        return env_secret
    global _EPHEMERAL_CSRF_SECRET
    if _EPHEMERAL_CSRF_SECRET is None:
        _EPHEMERAL_CSRF_SECRET = secrets.token_urlsafe(32)
        logging.warning(
            "CSRF_SECRET is not set; generated ephemeral secret for this process",
        )
    return _EPHEMERAL_CSRF_SECRET


@CsrfProtect.load_config
def get_csrf_config() -> CsrfSettings:
    """Return CSRF settings for FastAPI CsrfProtect."""
    return CsrfSettings(secret_key=_resolve_csrf_secret())


csrf_protect = CsrfProtect()


ALLOWED_HOSTS = {"127.0.0.1", "::1"}
host = os.getenv(
    "HOST", "127.0.0.1"
)  # Bind to localhost to avoid exposing the service externally
port_str = os.getenv("PORT", "8000")
try:
    port = int(port_str)
except ValueError:
    logging.error(
        "Invalid PORT value '%s'; must be an integer",
        sanitize_log_value(port_str),
    )
    sys.exit(1)

MAX_REQUEST_BYTES = 1 * 1024 * 1024  # 1 MB


_TOKEN_RE = re.compile(r"(?i)(bearer\s+)[^\s,]+")
_SENSITIVE_RE = re.compile(r"(?i)(token|cookie|x-api-key)=([^;\s]+)")


def _sanitize_header_value(value: str) -> str:
    """Mask sensitive data within a header value."""
    value = _TOKEN_RE.sub(r"\1***", value)
    value = _SENSITIVE_RE.sub(lambda m: f"{m.group(1)}=***", value)
    return value


def _loggable_headers(headers: Mapping[str, str]) -> dict[str, str]:
    sensitive = {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "proxy-authorization",
        "x-auth-token",
        "x-csrf-token",
        "x-xsrf-token",
    }
    return {
        sanitize_log_value(k): (
            "***"
            if k.lower() in sensitive
            else sanitize_log_value(_sanitize_header_value(v))
        )
        for k, v in headers.items()
    }


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    redacted_headers = _loggable_headers(request.headers)
    safe_method = sanitize_log_value(request.method)
    safe_url = sanitize_log_value(str(request.url))
    if not API_KEYS:
        logging.warning(
            "API_KEYS is empty; rejecting request: method=%s url=%s headers=%s",
            safe_method,
            safe_url,
            redacted_headers,
        )
        return Response(status_code=401)
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        logging.warning(
            "Unauthorized request: method=%s url=%s headers=%s",
            safe_method,
            safe_url,
            redacted_headers,
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
            redacted_headers,
        )
        return Response(status_code=401)
    return await call_next(request)


@app.middleware("http")
async def enforce_csrf(request: Request, call_next):
    if request.method == "POST":
        auth = request.headers.get("Authorization")
        token = auth[7:] if auth and auth.startswith("Bearer ") else None
        if token and any(hmac.compare_digest(token, key) for key in API_KEYS):
            try:
                await csrf_protect.validate_csrf(request)
            except CsrfProtectError:
                logging.warning(
                    "CSRF validation failed: method=%s url=%s headers=%s",
                    sanitize_log_value(request.method),
                    sanitize_log_value(str(request.url)),
                    _loggable_headers(request.headers),
                )
                raise HTTPException(status_code=403, detail="CSRF token missing or invalid")
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

    if host in ALLOWED_HOSTS:
        uvicorn.run("server:app", host=host, port=port)
    else:
        logging.error(
            "Invalid HOST '%s'; allowed values are %s",
            sanitize_log_value(host),
            ALLOWED_HOSTS,
        )
        sys.exit(1)
