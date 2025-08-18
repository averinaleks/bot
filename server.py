import os
import logging
import asyncio
from typing import List
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

_TRANSFORMERS_IMPORT_ERROR = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.exceptions import TransformersError
except Exception as exc:  # pragma: no cover - used for optional dependency
    AutoModelForCausalLM = AutoTokenizer = None
    TransformersError = Exception
    _TRANSFORMERS_IMPORT_ERROR = exc


# Global variables for model and tokenizer. They will be loaded on app startup.
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
# Task responsible for loading the model asynchronously.
_load_model_task: asyncio.Task | None = None


def load_model() -> str:
    """Load the tokenizer and model into global variables.

    Returns "primary" if the main model is loaded, "fallback" if the
    fallback model is loaded instead.
    """

    global tokenizer, model
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(
            "transformers is required to load the model. Install it with 'pip install transformers'."
        ) from _TRANSFORMERS_IMPORT_ERROR
    model_name = os.getenv("GPT_MODEL", "openai/gpt-oss-20b")
    fallback_model = os.getenv("GPT_MODEL_FALLBACK", "sshleifer/tiny-gpt2")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision="10e9d713f8e4a9281c59c40be6c58537480635ea",
            trust_remote_code=False,
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                revision="10e9d713f8e4a9281c59c40be6c58537480635ea",
                trust_remote_code=False,
            )
            .to(device)
        )
        return "primary"
    except (OSError, ValueError, TransformersError):
        logging.exception("Failed to load model '%s'", model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            fallback_model,
            revision="5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be",
            trust_remote_code=False,
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                fallback_model,
                revision="5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be",
                trust_remote_code=False,
            )
            .to(device)
        )
        logging.info("Loaded fallback model '%s'", fallback_model)
        return "fallback"
    except (OSError, ValueError, TransformersError):
        logging.exception("Failed to load fallback model '%s'", fallback_model)


async def load_model_async() -> None:
    """Asynchronously load the model without blocking the event loop.

    Raises:
        RuntimeError: If both primary and fallback models fail to load.
    """
    await asyncio.to_thread(load_model)




def generate_text(prompt: str, *, temperature: float = 0.7, max_new_tokens: int = 16) -> str:
    """Generate text from *prompt* using the loaded model."""
    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    prompt = "\n".join(message.content for message in req.messages)
    text = generate_text(prompt, temperature=req.temperature, max_new_tokens=req.max_tokens)
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    text = generate_text(req.prompt, temperature=req.temperature, max_new_tokens=req.max_tokens)
    return {"choices": [{"text": text}]}
