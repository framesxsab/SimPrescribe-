import json
import re
from typing import Any

import httpx
from huggingface_hub import InferenceClient

from .config import settings

DEFAULT_INSIGHT = "Use this medication exactly as prescribed and confirm unclear instructions with your clinician."


def build_structuring_prompt(raw_text: str) -> str:
    return f"""
You are extracting structured medication information from OCR text.
Return only valid JSON using the schema below.

OCR Text:
---
{raw_text}
---

Schema:
{{
  "medications": [
    {{
      "name": "medicine name",
      "category": "category",
      "type": "type",
      "dosage": "dosage",
      "frequency": "frequency",
      "duration": "duration",
      "insight": "one short safety note"
    }}
  ]
}}
""".strip()


def normalize_llm_json(raw_response: str) -> dict[str, Any]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model response did not contain valid JSON.")

    return json.loads(cleaned[start : end + 1])


def enrich_medications(medications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for medication in medications:
        normalized.append(
            {
                "name": str(medication.get("name") or "Unknown medication").strip(),
                "category": str(medication.get("category") or "General").strip(),
                "type": str(medication.get("type") or "Medication").strip(),
                "dosage": str(medication.get("dosage") or "N/A").strip(),
                "frequency": str(medication.get("frequency") or "N/A").strip(),
                "duration": str(medication.get("duration") or "N/A").strip(),
                "insight": str(medication.get("insight") or DEFAULT_INSIGHT).strip(),
            }
        )
    return normalized


def call_huggingface(raw_text: str) -> list[dict[str, Any]]:
    client = InferenceClient(token=settings.hf_token or None)
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": build_structuring_prompt(raw_text)},
        ],
        model=settings.hf_model,
        max_tokens=600,
        temperature=0.1,
    )
    payload = response.choices[0].message.content or "{}"
    parsed = normalize_llm_json(payload)
    medications = parsed.get("medications", [])
    if not isinstance(medications, list):
        raise ValueError("The model returned an invalid medications payload.")
    return enrich_medications(medications)


def call_http_endpoint(raw_text: str) -> list[dict[str, Any]]:
    if not settings.model_api_url:
        raise ValueError("MODEL_API_URL is required when INFERENCE_PROVIDER=endpoint.")

    headers = {"Content-Type": "application/json"}
    if settings.model_api_key:
        headers["Authorization"] = f"Bearer {settings.model_api_key}"

    payload = {"input": raw_text, "prompt": build_structuring_prompt(raw_text)}
    with httpx.Client(timeout=settings.request_timeout_seconds) as client:
        response = client.post(settings.model_api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    if isinstance(data, dict) and "medications" in data:
        medications = data["medications"]
    elif isinstance(data, dict) and "output" in data:
        parsed = normalize_llm_json(str(data["output"]))
        medications = parsed.get("medications", [])
    else:
        raise ValueError("Endpoint response format is not supported.")

    if not isinstance(medications, list):
        raise ValueError("Endpoint returned an invalid medications payload.")
    return enrich_medications(medications)


def fallback_extract(raw_text: str) -> list[dict[str, Any]]:
    candidates = [segment.strip() for segment in re.split(r"[,;\n]", raw_text) if segment.strip()]
    medications: list[dict[str, Any]] = []
    for segment in candidates[:6]:
        dosage_match = re.search(r"\b\d+\s?(?:mg|ml|mcg)\b", segment, flags=re.IGNORECASE)
        duration_match = re.search(r"\b\d+\s?(?:day|days|week|weeks)\b", segment, flags=re.IGNORECASE)
        words = segment.split()
        name = " ".join(words[:2]) if words else "Unknown medication"
        medications.append(
            {
                "name": name.title(),
                "category": "Unclassified",
                "type": "Medication",
                "dosage": dosage_match.group(0) if dosage_match else "N/A",
                "frequency": "Refer to prescription",
                "duration": duration_match.group(0) if duration_match else "N/A",
                "insight": DEFAULT_INSIGHT,
            }
        )

    if not medications:
        raise ValueError("No readable medication details were found in the OCR text.")
    return medications


def structure_medications(raw_text: str) -> list[dict[str, Any]]:
    if not raw_text.strip():
        raise ValueError("No readable text was extracted from the uploaded document.")

    provider = settings.inference_provider.strip().lower()
    if provider == "huggingface":
        if not settings.hf_token:
            return fallback_extract(raw_text)
        return call_huggingface(raw_text)
    if provider == "endpoint":
        return call_http_endpoint(raw_text)
    if provider == "fallback":
        return fallback_extract(raw_text)
    raise ValueError(f"Unsupported inference provider: {settings.inference_provider}")