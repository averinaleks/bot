import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = os.getenv("GPT_MODEL", "openai/gpt-oss-20b")
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

app = FastAPI()


def generate_text(prompt: str, *, temperature: float = 0.7, max_new_tokens: int = 16) -> str:
    """Generate text from *prompt* using the loaded model."""
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
    prompt = "\n".join(message.content for message in req.messages)
    text = generate_text(
        prompt, temperature=req.temperature, max_new_tokens=req.max_tokens
    )
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    text = generate_text(
        req.prompt, temperature=req.temperature, max_new_tokens=req.max_tokens
    )
    return {"choices": [{"text": text}]}
