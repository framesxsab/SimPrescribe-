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

FORM_PATTERN = r"tab(?:s)?|tablet|cap(?:s)?|capsule|syp|syr|syrup|susp|suspension|inj|injection|cream|ointment|drops"

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

GENERIC_NAME_TOKENS = set(FORM_MAP) | {
    "medicine",
    "medication",
    "oral",
    "strip",
    "vial",
    "bottle",
    "pack",
    "of",
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


def canonicalize_medicine_name(value: str) -> str:
    tokens = [token for token in re.split(r"\s+", str(value or "").strip()) if token]
    while tokens:
        normalized = normalize_text(tokens[-1])
        if not normalized:
            tokens.pop()
            continue
        if normalized in GENERIC_NAME_TOKENS or re.fullmatch(r"\d+(?:\.\d+)?(?:mg|ml|mcg|g)?", normalized):
            tokens.pop()
            continue
        break
    cleaned = " ".join(tokens)
    return title_case(cleaned or str(value or "").strip())


def extract_candidate_name(segment: str) -> str:
    tokens: list[str] = []
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
    return canonicalize_medicine_name(" ".join(tokens)) if tokens else "Unknown medication"


def build_match_candidates(value: str) -> list[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []

    tokens = [
        token
        for token in normalized.split()
        if token not in FORM_MAP
        and token not in STOP_TOKENS
        and not re.fullmatch(r"\d+(?:mg|ml|mcg|g)?", token)
        and not re.fullmatch(r"[01]-[01]-[01]", token)
    ]
    if not tokens:
        return []

    candidates: list[str] = []
    max_size = min(4, len(tokens))
    for size in range(max_size, 0, -1):
        candidate = " ".join(tokens[:size])
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


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

    candidates = build_match_candidates(segment)
    if not candidates:
        return None

    for candidate in candidates:
        if candidate in lexicon:
            return lexicon[candidate]

    best_entry: MedicineEntry | None = None
    best_score = 0.0
    for candidate in candidates:
        for alias, entry in lexicon.items():
            score = SequenceMatcher(None, candidate, alias).ratio()
            if score > best_score:
                best_score = score
                best_entry = entry

    return best_entry if best_score >= 0.92 else None


def split_segments(raw_text: str) -> list[str]:
    normalized = raw_text.replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()
    if not normalized:
        return []

    chunks = [chunk.strip(" ,.-") for chunk in re.split(r"(?:\n+|;|•)", normalized) if chunk.strip(" ,.-")]
    if len(chunks) > 1:
        return chunks

    cleaned = re.sub(r"\s+", " ", normalized)
    chunks = re.split(
        rf"\s+(?=(?:[A-Z][A-Za-z0-9.+/-]*(?:\s+[A-Z0-9][A-Za-z0-9.+/-]*){{0,3}}\s+(?:{FORM_PATTERN})\b))",
        cleaned,
    )
    return [chunk.strip(" ,.-") for chunk in chunks if chunk.strip(" ,.-")]


def extract_form(segment: str) -> str:
    match = re.search(rf"\b({FORM_PATTERN})\b", segment, flags=re.IGNORECASE)
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
    derived_name = extract_candidate_name(segment)
    if derived_name != "Unknown medication":
        return derived_name
    if entry is not None:
        return canonicalize_medicine_name(entry.name)
    return "Unknown medication"


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


def normalize_medication_type(value: str, entry: MedicineEntry | None = None) -> str:
    normalized = normalize_text(value)
    type_map = {
        "tab": "Tablet",
        "tabs": "Tablet",
        "tablet": "Tablet",
        "tablets": "Tablet",
        "tabular": "Tablet",
        "cap": "Capsule",
        "caps": "Capsule",
        "capsule": "Capsule",
        "capsules": "Capsule",
        "syr": "Syrup",
        "syp": "Syrup",
        "syrup": "Syrup",
        "susp": "Suspension",
        "suspension": "Suspension",
        "inj": "Injection",
        "injection": "Injection",
        "cream": "Cream",
        "ointment": "Ointment",
        "drops": "Drops",
        "drop": "Drops",
        "medication": "Medication",
        "medicine": "Medication",
    }
    if normalized in type_map:
        return type_map[normalized]

    if entry and entry.dosage_form:
        entry_type = normalize_medication_type(entry.dosage_form)
        if entry_type != "Medication":
            return entry_type

    return "Medication"


def normalize_frequency_value(value: str) -> str:
    text = clean_value(value)
    normalized = normalize_text(text)
    if not normalized:
        return "Refer to prescription"

    for shorthand, expanded in FREQUENCY_MAP.items():
        if normalized == shorthand:
            return expanded

    canonical_patterns = {
        r"once (a|per)? day|daily|every day|1x day": "once daily",
        r"twice (a|per)? day|2x day|two times daily": "twice daily",
        r"three times daily|thrice daily|3x day": "three times daily",
        r"four times daily|4x day": "four times daily",
        r"at bedtime|bedtime|hs": "at bedtime",
        r"as needed|when needed|prn|sos": "as needed",
        r"immediately|stat": "immediately",
    }
    for pattern, replacement in canonical_patterns.items():
        if re.fullmatch(pattern, normalized):
            return replacement

    timing_match = re.fullmatch(r"([01]-[01]-[01])", normalized)
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

    return text or "Refer to prescription"


def normalize_duration_value(value: str) -> str:
    text = clean_value(value)
    normalized = normalize_text(text)
    if not normalized or normalized in {"na", "n a", "none", "unknown"}:
        return "N/A"

    match = re.search(r"(\d+)\s*(day|days|d|week|weeks|w)", normalized)
    if not match:
        return text or "N/A"

    amount = match.group(1)
    unit = match.group(2)
    if unit in {"day", "days", "d"}:
        return f"{amount} day" if amount == "1" else f"{amount} days"
    return f"{amount} week" if amount == "1" else f"{amount} weeks"


def normalize_dosage_value(value: str, medication_type: str) -> str:
    text = clean_value(value)
    if not text:
        return "N/A"

    strength_match = re.search(r"\b\d+(?:\.\d+)?\s?(?:mg|ml|mcg|g)\b", text, flags=re.IGNORECASE)
    if strength_match:
        return strength_match.group(0).replace("  ", " ")

    count_match = re.search(r"\b\d+(?:\.\d+)?\s?(?:tablet|tablets|capsule|capsules|drop|drops|puff|puffs)\b", text, flags=re.IGNORECASE)
    if count_match:
        return count_match.group(0).replace("  ", " ")

    bare_number = re.fullmatch(r"\d{2,4}", normalize_text(text))
    if bare_number and medication_type in {"Tablet", "Capsule"}:
        return f"{bare_number.group(0)} mg"

    return text


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

    normalized_type = normalize_medication_type(medication_type, entry)
    normalized_frequency = normalize_frequency_value(frequency)
    normalized_duration = normalize_duration_value(duration)
    normalized_dosage = normalize_dosage_value(dosage, normalized_type)

    payload = {
        "name": name.strip() or "Unknown medication",
        "category": resolved_category,
        "type": normalized_type,
        "dosage": normalized_dosage,
        "frequency": normalized_frequency,
        "duration": normalized_duration,
        "insight": insight.strip() or DEFAULT_INSIGHT,
    }
    payload.update(dataset_payload(entry))
    return payload


def build_structuring_prompt(raw_text: str) -> str:
    return f"""
You are extracting structured medication information from OCR text.
Return only valid JSON using the schema below.

Rules:
- Return a JSON object, not an array.
- Use the key `medications` exactly.
- Do not include markdown, comments, or explanation text.
- Do not invent medicines that are not present in the OCR text.
- Split multiple medicines into separate objects.
- Keep `dosage` to strength or amount only, for example `650 mg`, `5 ml`, `1 tablet`.
- Do not put schedule or duration inside `dosage`.
- `type` must be one of: `Tablet`, `Capsule`, `Syrup`, `Suspension`, `Injection`, `Cream`, `Ointment`, `Drops`, `Medication`.
- `frequency` should be short, for example `once daily`, `twice daily`, `three times daily`, `at bedtime`, `as needed`, or `Refer to prescription`.
- `duration` should be short, for example `5 days`, `1 week`, or `N/A`.
- `category` should be brief. If unknown, use `General`.
- `insight` must be one short safety note. If unclear, use `Follow the prescription exactly as written.`
- If a field is unclear, prefer `N/A`, `General`, `Medication`, or `Refer to prescription` instead of guessing.

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

Example:
OCR Text: `Paracetamol 650 tab od 5 days`

Output:
{{
    "medications": [
        {{
            "name": "Paracetamol",
            "category": "General",
            "type": "Tablet",
            "dosage": "650 mg",
            "frequency": "once daily",
            "duration": "5 days",
            "insight": "Follow the prescription exactly as written."
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


def refine_model_medications(raw_text: str, medications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        heuristic_medications = fallback_extract(raw_text)
    except Exception:
        return medications

    refined: list[dict[str, Any]] = []
    for index, medication in enumerate(medications):
        merged = dict(medication)
        heuristic = heuristic_medications[index] if index < len(heuristic_medications) else None
        if heuristic is None:
            refined.append(merged)
            continue

        heuristic_name = str(heuristic.get("name") or "").strip()
        current_name = str(merged.get("name") or "").strip()
        if heuristic_name and not current_name:
            merged["name"] = heuristic_name

        heuristic_type = str(heuristic.get("type") or "").strip()
        current_type = normalize_medication_type(str(merged.get("type") or ""), None)
        if heuristic_type and heuristic_type != "Medication" and heuristic_type != current_type:
            merged["type"] = heuristic_type

        heuristic_frequency = str(heuristic.get("frequency") or "").strip()
        current_frequency = normalize_frequency_value(str(merged.get("frequency") or ""))
        if heuristic_frequency and heuristic_frequency != "Refer to prescription" and heuristic_frequency != current_frequency:
            merged["frequency"] = heuristic_frequency

        heuristic_duration = str(heuristic.get("duration") or "").strip()
        current_duration = normalize_duration_value(str(merged.get("duration") or ""))
        if heuristic_duration and heuristic_duration != "N/A" and current_duration == "N/A":
            merged["duration"] = heuristic_duration

        heuristic_dosage = str(heuristic.get("dosage") or "").strip()
        current_dosage = normalize_dosage_value(
            str(merged.get("dosage") or ""),
            normalize_medication_type(str(merged.get("type") or ""), None),
        )
        if heuristic_dosage and heuristic_dosage != "N/A" and current_dosage == "N/A":
            merged["dosage"] = heuristic_dosage

        refined.append(merged)

    return refined


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
    medications = refine_model_medications(raw_text, medications)
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
    medications = refine_model_medications(raw_text, medications)
    return enrich_medications(medications)


def fallback_extract(raw_text: str) -> list[dict[str, Any]]:
    candidates = split_segments(raw_text)
    medications: list[dict[str, Any]] = []
    for segment in candidates[:6]:
        derived_name = extract_candidate_name(segment)
        entry = find_best_medicine_match(derived_name if derived_name != "Unknown medication" else segment)
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