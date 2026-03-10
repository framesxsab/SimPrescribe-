import csv
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import InferenceClient

from .config import settings

DEFAULT_INSIGHT = "Use this medication exactly as prescribed and confirm unclear instructions with your clinician."

FORM_MAP = {
    "tab": "tablet",
    "tabs": "tablet",
    "tablet": "tablet",
    "cap": "capsule",
    "caps": "capsule",
    "capsule": "capsule",
    "syp": "syrup",
    "syr": "syrup",
    "syrup": "syrup",
    "susp": "suspension",
    "suspension": "suspension",
    "inj": "injection",
    "injection": "injection",
    "cream": "cream",
    "ointment": "ointment",
    "drops": "drops",
}

FREQUENCY_MAP = {
    "od": "once daily",
    "bd": "twice daily",
    "tds": "three times daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "hs": "at bedtime",
    "sos": "as needed",
    "stat": "immediately",
}

MEAL_MAP = {
    "ac": "before food",
    "pc": "after food",
}

STOP_TOKENS = set(FREQUENCY_MAP) | set(MEAL_MAP) | {
    "x",
    "for",
    "days",
    "day",
    "week",
    "weeks",
    "morning",
    "night",
}


@dataclass(frozen=True)
class MedicineEntry:
    name: str
    composition: str
    category: str
    dosage_form: str
    manufacturer: str
    pack_size: str
    therapeutic_class: str
    chemical_class: str
    action_class: str
    substitutes: tuple[str, ...]
    uses: tuple[str, ...]
    side_effects: tuple[str, ...]
    sources: tuple[str, ...]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def title_case(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().title()


def compact_composition(*parts: str) -> str:
    filtered = [part.strip() for part in parts if part and part.strip()]
    return " + ".join(filtered)


def clean_value(value: str | None) -> str:
    text = str(value or "").strip()
    if text.upper() == "NA":
        return ""
    return re.sub(r"\s+", " ", text)


def collect_series(row: dict[str, Any], prefix: str, limit: int) -> tuple[str, ...]:
    values: list[str] = []
    for index in range(limit):
        value = clean_value(row.get(f"{prefix}{index}"))
        if value and value not in values:
            values.append(value)
    return tuple(values)


def merge_entries(existing: MedicineEntry | None, incoming: MedicineEntry) -> MedicineEntry:
    if existing is None:
        return incoming

    def pick(current: str, new: str) -> str:
        return current or new

    def merge_tuple(current: tuple[str, ...], new: tuple[str, ...]) -> tuple[str, ...]:
        merged = list(current)
        for item in new:
            if item and item not in merged:
                merged.append(item)
        return tuple(merged)

    return MedicineEntry(
        name=pick(existing.name, incoming.name),
        composition=pick(existing.composition, incoming.composition),
        category=pick(existing.category, incoming.category),
        dosage_form=pick(existing.dosage_form, incoming.dosage_form),
        manufacturer=pick(existing.manufacturer, incoming.manufacturer),
        pack_size=pick(existing.pack_size, incoming.pack_size),
        therapeutic_class=pick(existing.therapeutic_class, incoming.therapeutic_class),
        chemical_class=pick(existing.chemical_class, incoming.chemical_class),
        action_class=pick(existing.action_class, incoming.action_class),
        substitutes=merge_tuple(existing.substitutes, incoming.substitutes),
        uses=merge_tuple(existing.uses, incoming.uses),
        side_effects=merge_tuple(existing.side_effects, incoming.side_effects),
        sources=merge_tuple(existing.sources, incoming.sources),
    )


@lru_cache(maxsize=1)
def load_medicine_lexicon() -> dict[str, MedicineEntry]:
    alias_to_key: dict[str, str] = {}
    entries: dict[str, MedicineEntry] = {}

    def add_alias(alias: str, key: str) -> None:
        normalized = normalize_text(alias)
        if normalized and normalized not in alias_to_key:
            alias_to_key[normalized] = key

    def upsert_entry(entry: MedicineEntry) -> str:
        key = normalize_text(entry.name)
        if not key:
            return ""
        entries[key] = merge_entries(entries.get(key), entry)
        return key

    def read_csv(path: Path):
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            return list(csv.DictReader(handle))

    for row in read_csv(settings.india_medicine_dataset):
        name = clean_value(row.get("name"))
        if not name:
            continue
        dosage_form = clean_value(row.get("pack_size_label")) or clean_value(row.get("type")) or "Medication"
        composition = compact_composition(
            clean_value(row.get("short_composition1")),
            clean_value(row.get("short_composition2")),
        )
        entry = MedicineEntry(
            name=name,
            composition=composition,
            category=clean_value(row.get("type")) or "General",
            dosage_form=dosage_form,
            manufacturer=clean_value(row.get("manufacturer_name")),
            pack_size=clean_value(row.get("pack_size_label")),
            therapeutic_class="",
            chemical_class="",
            action_class="",
            substitutes=(),
            uses=(),
            side_effects=(),
            sources=("India Medicines Dataset",),
        )
        key = upsert_entry(entry)
        if not key:
            continue
        add_alias(name, key)

        short_alias = re.sub(r"\b(tablet|capsule|syrup|cream|suspension|injection|oral suspension)\b", "", name, flags=re.IGNORECASE).strip()
        add_alias(short_alias, key)

    for row in read_csv(settings.medicine_database_dataset):
        name = clean_value(row.get("name"))
        if not name:
            continue
        substitutes = collect_series(row, "substitute", 5)
        entry = MedicineEntry(
            name=title_case(name),
            composition="",
            category=clean_value(row.get("Therapeutic Class")) or "General",
            dosage_form="Medication",
            manufacturer="",
            pack_size="",
            therapeutic_class=clean_value(row.get("Therapeutic Class")),
            chemical_class=clean_value(row.get("Chemical Class")),
            action_class=clean_value(row.get("Action Class")),
            substitutes=substitutes,
            uses=collect_series(row, "use", 5),
            side_effects=collect_series(row, "sideEffect", 42),
            sources=("Medicine Database",),
        )
        key = upsert_entry(entry)
        if not key:
            continue
        add_alias(name, key)
        for substitute_key in ("substitute0", "substitute1", "substitute2", "substitute3", "substitute4"):
            add_alias(clean_value(row.get(substitute_key)), key)

    return {alias: entries[key] for alias, key in alias_to_key.items() if key in entries}


def find_best_medicine_match(segment: str) -> MedicineEntry | None:
    lexicon = load_medicine_lexicon()
    if not lexicon:
        return None

    normalized_segment = normalize_text(segment)
    if not normalized_segment:
        return None

    tokens = normalized_segment.split()
    for size in range(min(5, len(tokens)), 0, -1):
        for start in range(0, len(tokens) - size + 1):
            candidate = " ".join(tokens[start : start + size])
            if candidate in lexicon:
                return lexicon[candidate]

    best_entry: MedicineEntry | None = None
    best_score = 0.0
    candidates = tokens[:4]
    for size in range(len(candidates), 0, -1):
        candidate = " ".join(candidates[:size])
        for alias, entry in lexicon.items():
            score = SequenceMatcher(None, candidate, alias).ratio()
            if score > best_score:
                best_score = score
                best_entry = entry

    return best_entry if best_score >= 0.86 else None


def split_segments(raw_text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", raw_text.replace("\r", " ")).strip()
    chunks = re.split(r"(?:(?<=\b(?:tab|cap|syp|syrup|inj|cream)\b)|[\n;]+)", cleaned, flags=re.IGNORECASE)
    return [chunk.strip(" ,.-") for chunk in chunks if chunk.strip(" ,.-")]


def extract_form(segment: str) -> str:
    match = re.search(r"\b(tab(?:s)?|tablet|cap(?:s)?|capsule|syp|syr|syrup|susp|suspension|inj|injection|cream|ointment|drops)\b", segment, flags=re.IGNORECASE)
    if not match:
        return "Medication"
    return FORM_MAP.get(match.group(1).lower(), "Medication").title()


def extract_dosage(segment: str, form: str) -> str:
    match = re.search(r"\b\d+(?:\.\d+)?\s?(?:mg|ml|mcg|g)\b", segment, flags=re.IGNORECASE)
    if match:
        return match.group(0).replace("  ", " ")

    bare_strength = re.search(r"\b(\d{2,4})\b", segment)
    if bare_strength and form.lower() in {"tablet", "capsule"}:
        return f"{bare_strength.group(1)} mg"
    return "N/A"


def extract_duration(segment: str) -> str:
    match = re.search(r"(?:x|for)?\s*(\d+)\s*(d|day|days|w|week|weeks)\b", segment, flags=re.IGNORECASE)
    if not match:
        return "N/A"
    amount = match.group(1)
    unit = match.group(2).lower()
    if unit in {"d", "day", "days"}:
        return f"{amount} days"
    return f"{amount} weeks"


def extract_frequency(segment: str) -> str:
    lower_segment = segment.lower()
    for shorthand, expanded in FREQUENCY_MAP.items():
        if re.search(rf"\b{re.escape(shorthand)}\b", lower_segment):
            meal_suffix = ""
            for meal_key, meal_text in MEAL_MAP.items():
                if re.search(rf"\b{re.escape(meal_key)}\b", lower_segment):
                    meal_suffix = f" {meal_text}"
                    break
            return f"{expanded}{meal_suffix}".strip()

    timing_match = re.search(r"\b([01]-[01]-[01])\b", lower_segment)
    if timing_match:
        pattern = timing_match.group(1)
        timing_map = {
            "1-0-0": "once daily in the morning",
            "0-1-0": "once daily in the afternoon",
            "0-0-1": "once daily at night",
            "1-0-1": "twice daily",
            "1-1-0": "twice daily",
            "0-1-1": "twice daily",
            "1-1-1": "three times daily",
        }
        return timing_map.get(pattern, "Refer to prescription")

    return "Refer to prescription"


def derive_name(segment: str, entry: MedicineEntry | None) -> str:
    if entry is not None:
        return entry.name

    tokens = []
    for token in re.split(r"\s+", segment):
        normalized = normalize_text(token)
        if not normalized:
            continue
        if normalized in FORM_MAP or normalized in STOP_TOKENS:
            continue
        if re.fullmatch(r"\d+(?:mg|ml|mcg|g)?", normalized):
            break
        if re.fullmatch(r"[01]-[01]-[01]", normalized):
            break
        tokens.append(token)
        if len(tokens) == 3:
            break
    return title_case(" ".join(tokens) or "Unknown medication")


def build_insight(entry: MedicineEntry | None, frequency: str, duration: str, form: str) -> str:
    parts: list[str] = []
    if entry and entry.composition:
        parts.append(f"Contains {entry.composition}.")
    if frequency != "Refer to prescription":
        message = f"Take {frequency}"
        if duration != "N/A":
            message += f" for {duration}"
        parts.append(f"{message}.")
    elif form != "Medication":
        parts.append(f"Prescription form identified as {form.lower()}.")
    if not parts:
        parts.append(DEFAULT_INSIGHT)
    return " ".join(parts)


def dataset_payload(entry: MedicineEntry | None) -> dict[str, Any]:
    if entry is None:
        return {
            "source": "OCR only",
            "source_datasets": [],
            "composition": "",
            "manufacturer": "",
            "pack_size": "",
            "therapeutic_class": "",
            "chemical_class": "",
            "action_class": "",
            "substitutes": [],
            "uses": [],
            "side_effects": [],
        }

    return {
        "source": ", ".join(entry.sources),
        "source_datasets": list(entry.sources),
        "composition": entry.composition,
        "manufacturer": entry.manufacturer,
        "pack_size": entry.pack_size,
        "therapeutic_class": entry.therapeutic_class,
        "chemical_class": entry.chemical_class,
        "action_class": entry.action_class,
        "substitutes": list(entry.substitutes),
        "uses": list(entry.uses),
        "side_effects": list(entry.side_effects),
    }


def build_medication_record(
    *,
    name: str,
    category: str,
    medication_type: str,
    dosage: str,
    frequency: str,
    duration: str,
    insight: str,
    entry: MedicineEntry | None,
) -> dict[str, Any]:
    resolved_category = category.strip() or "General"
    if resolved_category.lower() == "general" and entry and entry.category:
        resolved_category = entry.category.title()

    payload = {
        "name": name.strip() or "Unknown medication",
        "category": resolved_category,
        "type": medication_type.strip() or "Medication",
        "dosage": dosage.strip() or "N/A",
        "frequency": frequency.strip() or "N/A",
        "duration": duration.strip() or "N/A",
        "insight": insight.strip() or DEFAULT_INSIGHT,
    }
    payload.update(dataset_payload(entry))
    return payload


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
        name = str(medication.get("name") or "Unknown medication").strip()
        entry = find_best_medicine_match(name)
        normalized.append(
            build_medication_record(
                name=name,
                category=str(medication.get("category") or "General"),
                medication_type=str(medication.get("type") or "Medication"),
                dosage=str(medication.get("dosage") or "N/A"),
                frequency=str(medication.get("frequency") or "N/A"),
                duration=str(medication.get("duration") or "N/A"),
                insight=str(medication.get("insight") or DEFAULT_INSIGHT),
                entry=entry,
            )
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
    candidates = split_segments(raw_text)
    medications: list[dict[str, Any]] = []
    for segment in candidates[:6]:
        entry = find_best_medicine_match(segment)
        dosage_form = extract_form(segment)
        duration = extract_duration(segment)
        frequency = extract_frequency(segment)
        name = derive_name(segment, entry)
        category = entry.category.title() if entry and entry.category else "General"
        medication_type = dosage_form if dosage_form != "Medication" else "Medication"
        if entry and entry.dosage_form and dosage_form == "Medication":
            medication_type = title_case(entry.dosage_form)
        medications.append(
            build_medication_record(
                name=name,
                category=category,
                medication_type=medication_type,
                dosage=extract_dosage(segment, dosage_form),
                frequency=frequency,
                duration=duration,
                insight=build_insight(entry, frequency, duration, dosage_form),
                entry=entry,
            )
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