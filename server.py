import os
import re
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


app = FastAPI()


# Global variables for model and tokenizer. They will be loaded on app startup.
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
PINNED_MODEL_REVISION = "10e9d713f8e4a9281c59c40be6c58537480635ea"


@app.on_event("startup")
def load_model() -> None:
    """Load the tokenizer and model into global variables."""
    global tokenizer, model
    model_name = os.getenv("GPT_MODEL", "openai/gpt-oss-20b")
    revision = os.getenv("GPT_MODEL_REVISION", PINNED_MODEL_REVISION)
    if os.getenv("GPT_MODEL_REVISION") is None or revision == "<commit-or-tag>":
        raise ValueError(
            "GPT_MODEL_REVISION environment variable must be set to a valid commit hash or tag"
        )
    if not re.fullmatch(r"[0-9a-f]{40}|[A-Za-z0-9._-]+", revision):
        raise ValueError(
            "GPT_MODEL_REVISION environment variable must be a valid commit hash or tag"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        .to(device)
    )


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
    temperature: float = 0.7
    max_tokens: int = 16


class CompletionRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 16


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
