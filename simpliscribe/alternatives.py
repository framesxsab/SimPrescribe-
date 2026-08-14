"""Alternative medicine reference candidates from local datasets, then optional model/web search.

Local trained data always runs: CSV substitute columns plus other brands that share
the same composition. Model/web search only runs when that local list is empty and
ALTERNATIVES_ENABLED is on. Every off-box candidate is validated against the local
medicine lexicon so a hallucinated drug name can never be surfaced.

Safety:
- All results are reference candidates, never recommendations, and force review.
- Web/model lookup is disabled by default; enabling sends only the canonical medicine name off-box.
- Errors and timeouts fail open to an empty list so the pipeline never breaks.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

import httpx
from huggingface_hub import InferenceClient

from .config import settings
from .inference import (
    find_medicine_match,
    is_junk_medication,
    load_medicine_lexicon,
    normalize_llm_json,
    normalize_text,
    title_case,
)

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS

    DDGS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    DDGS_AVAILABLE = False

UNKNOWN_KEYS = {"", "unknown medication", "unknown", "na", "n a", "none"}

_CACHE_MAX_ENTRIES = 512
_cache: dict[str, tuple[float, list[dict[str, str]]]] = {}

_SCAN_STOP = {
    "the", "and", "for", "with", "from", "tablet", "tablets", "capsule", "capsules",
    "syrup", "suspension", "injection", "cream", "ointment", "drops", "medicine",
    "medicines", "medication", "medications", "drug", "drugs", "uses", "use", "used",
    "side", "effects", "effect", "substitute", "substitutes", "alternative",
    "alternatives", "price", "prices", "review", "reviews", "buy", "online",
    "free", "otc", "cod", "pharmacy", "shop", "store", "offer", "delivery", "mg",
}


def _cache_get(key: str) -> list[dict[str, str]] | None:
    cached = _cache.get(key)
    if cached is None:
        return None
    stored_at, value = cached
    if time.time() - stored_at > settings.alternatives_cache_ttl_seconds:
        _cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: list[dict[str, str]]) -> None:
    if key not in _cache and len(_cache) >= _CACHE_MAX_ENTRIES:
        _cache.pop(next(iter(_cache)))
    _cache[key] = (time.time(), value)


def build_alternatives_prompt(name: str) -> str:
    return (
        "You provide alternative medicine reference candidates for a prescription review aid.\n"
        'Return ONLY valid JSON with no markdown or explanation: {"alternatives": ["Name A", "Name B"]}\n'
        "Rules:\n"
        "- List 3 to 5 real alternative medicines commonly used as substitutes for the same condition.\n"
        "- Prefer generic names or widely available brands.\n"
        "- Do not invent medicines.\n"
        f"Medicine: {name}"
    )


@lru_cache(maxsize=1)
def _composition_to_names() -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for entry in load_medicine_lexicon().values():
        composition = normalize_text(entry.composition)
        if len(composition) < 8:
            continue
        display = str(entry.name or "").strip()
        key = normalize_text(display)
        if not display or key in UNKNOWN_KEYS:
            continue
        names = grouped.setdefault(composition, [])
        already = seen.setdefault(composition, set())
        if key in already:
            continue
        already.add(key)
        names.append(display)
    return {composition: tuple(names) for composition, names in grouped.items()}


def dataset_reference_candidates(medicine_name: str, existing: list[str] | None = None, limit: int | None = None) -> list[str]:
    """CSV substitutes plus other local brands that share the same composition."""
    cap = limit if limit is not None else settings.alternatives_max_candidates
    query = normalize_text(medicine_name)
    names: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        key = normalize_text(raw)
        if not key or key in UNKNOWN_KEYS or key == query or key in seen:
            return
        display = title_case(raw) if raw == key else str(raw).strip()
        if is_junk_medication(display.replace("/", " ")):
            return
        seen.add(key)
        names.append(display)

    for item in existing or []:
        add(str(item))
    if len(names) >= cap:
        return names[:cap]

    match = find_medicine_match(medicine_name)
    entry = match.entry if match else load_medicine_lexicon().get(query)
    if entry is None:
        return names[:cap]
    for item in entry.substitutes:
        add(item)
        if len(names) >= cap:
            return names[:cap]
    composition = normalize_text(entry.composition)
    if len(composition) >= 8:
        for peer in _composition_to_names().get(composition, ()):
            add(peer)
            if len(names) >= cap:
                break
    return names[:cap]


@lru_cache(maxsize=1)
def _lexicon_first_token_index() -> dict[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    index: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for alias in load_medicine_lexicon():
        tokens = tuple(alias.split())
        if tokens and len(tokens[0]) > 2:
            index.setdefault(tokens[0], []).append((alias, tokens))
    return {token: tuple(entries) for token, entries in index.items()}


def candidate_names_in_text(text: str) -> list[str]:
    """Aliases (normalized) of known medicines that appear in the given text."""
    lexicon = load_medicine_lexicon()
    if not lexicon:
        return []
    tokens = normalize_text(text).split()
    if not tokens:
        return []
    index = _lexicon_first_token_index()
    found: list[str] = []
    seen: set[str] = set()
    for start, token in enumerate(tokens):
        if token in _SCAN_STOP:
            continue
        for alias, alias_tokens in index.get(token, ()):
            end = start + len(alias_tokens)
            if end > len(tokens):
                continue
            if tuple(tokens[start:end]) != alias_tokens or alias in seen:
                continue
            if is_junk_medication(title_case(alias)):
                continue
            seen.add(alias)
            found.append(alias)
    return found


def _display_name(key: str, fallback: str) -> str | None:
    lexicon = load_medicine_lexicon()
    if key not in lexicon:
        if len(key) < 3:
            return None
        prefix_hits = [alias for alias, _ in _lexicon_first_token_index().get(key.split(" ", 1)[0], ()) if alias.startswith(key)]
        if not prefix_hits:
            return None
    display = title_case(fallback)
    if is_junk_medication(display.replace("/", " ")):
        return None
    return display


def _validate_names(raw_names: list[str], *, source: str, provider: str, url: str = "") -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_names:
        key = normalize_text(raw)
        if not key or key in UNKNOWN_KEYS:
            continue
        display = _display_name(key, raw)
        if display is None or display in seen:
            continue
        seen.add(display)
        items.append({"name": display, "source": source, "provider": provider, "url": url})
    return items


def _extract_names(parsed: dict[str, Any]) -> list[str]:
    for field in ("alternatives", "medications", "names"):
        value = parsed.get(field)
        if isinstance(value, list):
            names: list[str] = []
            for item in value:
                name = str(item.get("name") if isinstance(item, dict) else item or "").strip()
                if name:
                    names.append(name)
            return names
    return []


def _parse_model_payload(raw_response: str, *, provider: str) -> list[dict[str, str]]:
    try:
        parsed = normalize_llm_json(raw_response)
    except Exception:
        logger.warning("Model alternatives response was not valid JSON.")
        return []
    return _validate_names(_extract_names(parsed), source="model", provider=provider)


def call_model_alternatives(name: str) -> list[dict[str, str]]:
    provider = settings.inference_provider.strip().lower()
    if provider == "huggingface":
        if not settings.hf_token:
            return []
        client = InferenceClient(token=settings.hf_token, timeout=settings.alternatives_timeout_seconds)
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a pharmacology reference assistant. Return only valid JSON."},
                {"role": "user", "content": build_alternatives_prompt(name)},
            ],
            model=settings.hf_model,
            max_tokens=200,
            temperature=0.1,
        )
        return _parse_model_payload(response.choices[0].message.content or "{}", provider="huggingface")
    if provider == "endpoint":
        if not settings.model_api_url:
            return []
        headers = {"Content-Type": "application/json"}
        if settings.model_api_key:
            headers["Authorization"] = f"Bearer {settings.model_api_key}"
        payload = {"input": name, "prompt": build_alternatives_prompt(name)}
        with httpx.Client(timeout=settings.alternatives_timeout_seconds) as client:
            response = client.post(settings.model_api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        if isinstance(data, dict) and "output" in data:
            return _parse_model_payload(str(data["output"]), provider="endpoint")
        if isinstance(data, dict):
            return _validate_names(_extract_names(data), source="model", provider="endpoint")
    return []


def call_web_alternatives(name: str) -> list[dict[str, str]]:
    if not DDGS_AVAILABLE:
        logger.info("duckduckgo-search is not installed; skipping web alternatives for %s", name)
        return []
    query = f"{name} substitute alternative medicine"
    results: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return []
    key = normalize_text(name)
    found: dict[str, str] = {}
    for result in results:
        url = str(result.get("href") or "")
        text = " ".join([str(result.get("title") or ""), str(result.get("body") or "")])
        for alias in candidate_names_in_text(text):
            if alias == key:
                continue
            found.setdefault(alias, url)
    items: list[dict[str, str]] = []
    for alias, url in found.items():
        display = _display_name(alias, alias)
        if display:
            items.append({"name": display, "source": "web", "provider": "duckduckgo", "url": url})
    return items


def fetch_alternatives(medicine_name: str) -> list[dict[str, str]]:
    if not settings.alternatives_enabled:
        return []
    key = normalize_text(medicine_name)
    if not key or key in UNKNOWN_KEYS:
        return []
    cached = _cache_get(key)
    if cached is not None:
        return cached
    name = str(medicine_name).strip()
    merged: dict[str, dict[str, str]] = {}
    for tier in settings.alternatives_provider_chain:
        try:
            provider = call_model_alternatives if tier == "model" else call_web_alternatives
            for item in provider(name):
                merged.setdefault(item["name"], item)
        except Exception:
            logger.exception("Alternatives %s tier failed for %s.", tier, name)
    items = list(merged.values())[: settings.alternatives_max_candidates]
    _cache_set(key, items)
    return items


def attach_alternative_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach local dataset peers always, then optional model/web candidates when those are absent."""
    existing = payload.get("substitutes") if isinstance(payload.get("substitutes"), list) else []
    local = dataset_reference_candidates(str(payload.get("name") or ""), existing)
    if local:
        payload["substitutes"] = local
        payload["requires_review"] = True
        reason = "Dataset reference candidates are not a dispensing decision; confirm availability and equivalence with a pharmacist or prescriber."
        reasons = payload.setdefault("review_reasons", [])
        if reason not in reasons:
            reasons.append(reason)
    if not settings.alternatives_enabled or payload.get("substitutes"):
        return payload
    candidates = fetch_alternatives(str(payload.get("name") or ""))
    if not candidates:
        return payload
    payload["web_alternatives"] = candidates
    payload["requires_review"] = True
    web_reason = "Alternative reference candidates were sourced from a model/web search and must be verified by a prescriber."
    reasons = payload.setdefault("review_reasons", [])
    if web_reason not in reasons:
        reasons.append(web_reason)
    return payload
