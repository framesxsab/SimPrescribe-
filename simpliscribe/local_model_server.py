from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .inference import build_structuring_prompt, normalize_llm_json


class ExtractRequest(BaseModel):
    input: str = Field(default="", description="Raw OCR text to structure.")
    prompt: str | None = Field(default=None, description="Optional fully formed prompt.")


class ExtractResponse(BaseModel):
    medications: list[dict[str, Any]]
    output: str
    model: str


app = FastAPI(title="SimpliScribe Local Model Server")


def resolve_device() -> str:
    if settings.local_model_device != "auto":
        return settings.local_model_device

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_text_generator() -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Local model server dependencies are missing. Install requirements-local-model.txt first."
        ) from exc

    device = resolve_device()
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        settings.local_model_id,
        trust_remote_code=settings.local_model_trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.local_model_id,
        trust_remote_code=settings.local_model_trust_remote_code,
        dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def build_chat_prompt(raw_text: str, prompt_override: str | None = None) -> str:
    prompt = prompt_override or build_structuring_prompt(raw_text)
    generator = load_text_generator()
    tokenizer = generator.tokenizer
    messages = [
        {
            "role": "system",
            "content": (
                "You extract medication fields from prescription OCR text. "
                "Return only one JSON object with a medications array. "
                "Do not add explanations, markdown, or extra keys. "
                "Use conservative values when uncertain."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return (
        "You extract medication fields from prescription OCR text. "
        "Return only one JSON object with a medications array. "
        "Do not add explanations, markdown, or extra keys.\n\n"
        + prompt
    )


def generate_output(raw_text: str, prompt_override: str | None = None) -> str:
    generator = load_text_generator()
    prompt = build_chat_prompt(raw_text, prompt_override)
    outputs = generator(
        prompt,
        max_new_tokens=settings.local_model_max_new_tokens,
        do_sample=settings.local_model_temperature > 0,
        temperature=settings.local_model_temperature,
        repetition_penalty=1.08,
        return_full_text=False,
    )
    if not outputs:
        raise ValueError("Local model did not return any output.")

    generated = outputs[0]
    if isinstance(generated, dict):
        return str(generated.get("generated_text") or generated.get("text") or "").strip()
    return str(generated).strip()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.local_model_id}


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    raw_text = request.input.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="The input field must contain OCR text.")

    try:
        output = generate_output(raw_text, request.prompt)
        parsed = normalize_llm_json(output)
        medications = parsed.get("medications", [])
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local model extraction failed: {exc}") from exc

    if not isinstance(medications, list):
        raise HTTPException(status_code=500, detail="Local model returned an invalid medications payload.")

    return ExtractResponse(medications=medications, output=output, model=settings.local_model_id)