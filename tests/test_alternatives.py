"""Tests for web/model-sourced alternative medicine reference candidates."""

import json
from types import SimpleNamespace

import pytest

from simpliscribe import alternatives
from simpliscribe.alternatives import (
    _cache,
    attach_alternative_candidates,
    call_model_alternatives,
    call_web_alternatives,
    candidate_names_in_text,
    fetch_alternatives,
)

ENABLED_SETTINGS = SimpleNamespace(
    alternatives_enabled=True,
    alternatives_provider_chain=["model", "web"],
    alternatives_max_candidates=5,
    alternatives_cache_ttl_seconds=86400,
    alternatives_timeout_seconds=5,
    inference_provider="fallback",
    hf_token="",
    hf_model="Qwen/Qwen2.5-7B-Instruct",
    model_api_url="",
    model_api_key="",
)

DISABLED_SETTINGS = SimpleNamespace(
    alternatives_enabled=False,
    alternatives_provider_chain=["model", "web"],
    alternatives_max_candidates=5,
    alternatives_cache_ttl_seconds=86400,
    alternatives_timeout_seconds=5,
    inference_provider="fallback",
    hf_token="",
    hf_model="Qwen/Qwen2.5-7B-Instruct",
    model_api_url="",
    model_api_key="",
)


@pytest.fixture(autouse=True)
def clear_alternatives_cache(monkeypatch):
    _cache.clear()
    yield
    _cache.clear()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr("simpliscribe.alternatives.settings", ENABLED_SETTINGS)


def test_fetch_alternatives_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr("simpliscribe.alternatives.settings", DISABLED_SETTINGS)
    assert fetch_alternatives("Augmentin 625 Duo Tablet") == []


def test_fetch_alternatives_skips_unknown_names(enabled):
    assert fetch_alternatives("") == []
    assert fetch_alternatives("Unknown medication") == []
    assert fetch_alternatives("N/A") == []


def test_fetch_alternatives_fail_open_when_all_providers_error(enabled, monkeypatch):
    def boom(_name):
        raise RuntimeError("provider down")

    monkeypatch.setattr(alternatives, "call_model_alternatives", boom)
    monkeypatch.setattr(alternatives, "call_web_alternatives", boom)
    assert fetch_alternatives("Zonflorab") == []


def test_fetch_alternatives_chains_model_then_web(enabled, monkeypatch):
    monkeypatch.setattr(
        alternatives,
        "call_model_alternatives",
        lambda name: [{"name": "Paracetamol", "source": "model", "provider": "huggingface", "url": ""}],
    )
    monkeypatch.setattr(
        alternatives,
        "call_web_alternatives",
        lambda name: [{"name": "Ibuprofen", "source": "web", "provider": "duckduckgo", "url": "https://example.com"}],
    )
    result = fetch_alternatives("SomeDrug")
    names = [item["name"] for item in result]
    assert names == ["Paracetamol", "Ibuprofen"]


def test_fetch_alternatives_deduplicates_and_caps(enabled, monkeypatch):
    monkeypatch.setattr(
        alternatives,
        "call_model_alternatives",
        lambda name: [{"name": "Paracetamol", "source": "model", "provider": "huggingface", "url": ""}] * 3,
    )
    monkeypatch.setattr(
        alternatives,
        "call_web_alternatives",
        lambda name: [{"name": "Paracetamol", "source": "web", "provider": "duckduckgo", "url": "https://example.com"}],
    )
    result = fetch_alternatives("SomeDrug")
    assert len(result) == 1


def test_fetch_alternatives_respects_max_candidates(enabled, monkeypatch):
    monkeypatch.setattr(alternatives, "settings", SimpleNamespace(**{**ENABLED_SETTINGS.__dict__, "alternatives_max_candidates": 2}))
    model_items = [{"name": f"Drug {i}", "source": "model", "provider": "huggingface", "url": ""} for i in range(1, 6)]
    monkeypatch.setattr(alternatives, "call_model_alternatives", lambda name: model_items)
    monkeypatch.setattr(alternatives, "call_web_alternatives", lambda name: [])
    assert len(fetch_alternatives("SomeDrug")) == 2


def test_fetch_alternatives_caches_results(enabled, monkeypatch):
    calls = {"count": 0}

    def counting_model(name):
        calls["count"] += 1
        return [{"name": "Paracetamol", "source": "model", "provider": "huggingface", "url": ""}]

    monkeypatch.setattr(alternatives, "call_model_alternatives", counting_model)
    monkeypatch.setattr(alternatives, "call_web_alternatives", lambda name: [])
    assert fetch_alternatives("SomeDrug") != []
    assert fetch_alternatives("SomeDrug") != []
    assert calls["count"] == 1


def test_validate_names_keeps_only_lexicon_medicines(enabled):
    from simpliscribe.alternatives import _validate_names

    items = _validate_names(
        ["Moxikind-CV 625 Tablet", "TotallyFakeDrugXYZ", "N/A"],
        source="model",
        provider="huggingface",
    )
    names = [item["name"] for item in items]
    assert any("Moxikind" in name for name in names)
    assert "TotallyFakeDrugXYZ" not in names


def test_parse_model_payload_rejects_invalid_json(enabled):
    from simpliscribe.alternatives import _parse_model_payload

    assert _parse_model_payload("not json at all", provider="huggingface") == []
    items = _parse_model_payload(
        json.dumps({"alternatives": ["Paracetamol", "ZzzFakeDrug"]}),
        provider="huggingface",
    )
    assert [item["name"] for item in items] == ["Paracetamol"]


def test_call_model_alternatives_returns_empty_without_token(enabled):
    assert call_model_alternatives("Paracetamol") == []


def test_call_model_alternatives_huggingface_parses_response(enabled, monkeypatch):
    class FakeResponse:
        def __init__(self, content):
            self.message = SimpleNamespace(content=content)

    class FakeChoices:
        def __init__(self, content):
            self.choices = [FakeResponse(content)]

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat_completion(self, **kwargs):
            return FakeChoices(json.dumps({"alternatives": ["Paracetamol", "FakeDrugXYZ"]}))

    hf_settings = SimpleNamespace(**{**ENABLED_SETTINGS.__dict__, "inference_provider": "huggingface", "hf_token": "token", "hf_model": "model"})
    monkeypatch.setattr(alternatives, "settings", hf_settings)
    monkeypatch.setattr(alternatives, "InferenceClient", FakeClient)
    items = call_model_alternatives("SomeDrug")
    assert [item["name"] for item in items] == ["Paracetamol"]


def test_call_model_alternatives_endpoint_parses_output(enabled, monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output": json.dumps({"alternatives": ["Brufen", "FakeDrugXYZ"]})}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    endpoint_settings = SimpleNamespace(**{**ENABLED_SETTINGS.__dict__, "inference_provider": "endpoint", "model_api_url": "http://localhost:8001/extract", "model_api_key": ""})
    monkeypatch.setattr(alternatives, "settings", endpoint_settings)
    monkeypatch.setattr(alternatives.httpx, "Client", FakeClient)
    items = call_model_alternatives("SomeDrug")
    assert [item["name"] for item in items] == ["Brufen"]


class FakeDDGS:
    def __init__(self):
        self.results = [
            {
                "title": "Moxikind CV 625 Tablet: Uses, Side Effects, Substitutes",
                "href": "https://example.com/moxikind",
                "body": "Penciclav 500 mg/125 mg Tablet is a substitute for Augmentin 625 Duo Tablet.",
            }
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, query, max_results=5):
        return self.results


def test_call_web_alternatives_extracts_lexicon_names(enabled, monkeypatch):
    monkeypatch.setattr(alternatives, "DDGS", FakeDDGS, raising=False)
    monkeypatch.setattr(alternatives, "DDGS_AVAILABLE", True)
    items = call_web_alternatives("Augmentin 625 Duo Tablet")
    names = [item["name"] for item in items]
    assert any("Moxikind" in name for name in names)
    assert any("Penciclav" in name for name in names)
    assert not any("Augmentin 625 Duo Tablet" in name for name in names)
    assert items[0]["url"] == "https://example.com/moxikind"


def test_call_web_alternatives_empty_when_unavailable(enabled, monkeypatch):
    monkeypatch.setattr(alternatives, "DDGS_AVAILABLE", False)
    assert call_web_alternatives("Augmentin") == []


def test_candidate_names_in_text_finds_lexicon_aliases():
    hits = candidate_names_in_text("Moxikind CV 625 Tablet: Uses, Side Effects, Substitutes")
    assert any("moxikind cv 625 tablet" in hit for hit in hits)


def test_attach_does_not_run_web_when_local_substitutes_exist(enabled, monkeypatch):
    called: list[str] = []

    def should_not_run(name: str):
        called.append(name)
        return [{"name": "X", "source": "web", "provider": "duckduckgo", "url": ""}]

    monkeypatch.setattr(alternatives, "fetch_alternatives", should_not_run)
    payload = {"name": "Augmentin 625 Duo Tablet", "substitutes": ["Penciclav 500 mg/125 mg Tablet"]}
    result = attach_alternative_candidates(dict(payload))
    assert called == []
    assert "Penciclav 500 mg/125 mg Tablet" in result["substitutes"]
    assert "web_alternatives" not in result
    assert result["requires_review"] is True
    assert result["alternatives_lookup"]["skipped_reason"] == "local_candidates_present"
    assert result["alternatives_lookup"]["web_ran"] is False


def test_attach_sets_review_flags_without_substitutes(enabled, monkeypatch):
    monkeypatch.setattr(
        alternatives,
        "fetch_alternatives",
        lambda name: [{"name": "Paracetamol", "source": "model", "provider": "huggingface", "url": ""}],
    )
    payload = {"name": "SomeDrug", "requires_review": False, "review_reasons": []}
    result = attach_alternative_candidates(payload)
    assert result["web_alternatives"]
    assert result["requires_review"] is True
    assert any("must be verified" in reason for reason in result["review_reasons"])
    assert result["alternatives_lookup"]["skipped_reason"] == ""
    assert result["alternatives_lookup"]["web_ran"] is True


def test_attach_records_disabled_web_lookup(monkeypatch):
    monkeypatch.setattr("simpliscribe.alternatives.settings", DISABLED_SETTINGS)
    monkeypatch.setattr(alternatives, "dataset_reference_candidates", lambda *_args, **_kwargs: [])
    result = attach_alternative_candidates({"name": "UnknownBrandXYZ", "substitutes": []})
    assert "web_alternatives" not in result
    assert result["alternatives_lookup"]["skipped_reason"] == "lookup_disabled"
    assert result["alternatives_lookup"]["web_enabled"] is False
    assert result["alternatives_lookup"]["web_ran"] is False
    assert result["alternatives_lookup"]["web_count"] == 0


def test_only_medicine_name_is_sent_off_box(enabled, monkeypatch):
    sent = {}

    def recording_web(name):
        sent["query"] = name
        return [{"name": "Paracetamol", "source": "web", "provider": "duckduckgo", "url": ""}]

    monkeypatch.setattr(alternatives, "call_model_alternatives", lambda name: [])
    monkeypatch.setattr(alternatives, "call_web_alternatives", recording_web)
    fetch_alternatives("Augmentin 625 Duo Tablet")
    assert sent["query"] == "Augmentin 625 Duo Tablet"
    assert "patient" not in sent["query"].lower()
    assert "Dr" not in sent["query"]


def _entry(name: str, composition: str, substitutes: tuple[str, ...] = ()) -> object:
    from simpliscribe.inference import MedicineEntry

    return MedicineEntry(
        name=name,
        composition=composition,
        category="General",
        dosage_form="Tablet",
        manufacturer="",
        pack_size="",
        therapeutic_class="",
        chemical_class="",
        action_class="",
        substitutes=substitutes,
        uses=(),
        side_effects=(),
        sources=("Test Dataset",),
    )


def test_dataset_reference_candidates_use_same_composition(monkeypatch):
    from simpliscribe.inference import MedicineMatch

    alpha = _entry("Alpha 650 Tablet", "Paracetamol 650 mg")
    beta = _entry("Beta 650 Tablet", "Paracetamol 650 mg")
    lexicon = {"alpha 650 tablet": alpha, "beta 650 tablet": beta}

    monkeypatch.setattr(alternatives, "load_medicine_lexicon", lambda: lexicon)
    monkeypatch.setattr(
        alternatives,
        "find_medicine_match",
        lambda name: MedicineMatch(alpha, 1.0, "exact", "alpha 650 tablet") if "alpha" in name.lower() else None,
    )
    alternatives._composition_to_names.cache_clear()
    peers = alternatives.dataset_reference_candidates("Alpha 650 Tablet")
    assert "Beta 650 Tablet" in peers
    assert "Alpha 650 Tablet" not in peers


def test_attach_fills_local_peers_even_when_web_is_disabled(monkeypatch):
    from simpliscribe.inference import MedicineMatch

    listed = _entry("Listed Brand", "Ibuprofen 400 mg", substitutes=("Other Brand Tablet",))
    monkeypatch.setattr("simpliscribe.alternatives.settings", DISABLED_SETTINGS)
    monkeypatch.setattr(alternatives, "load_medicine_lexicon", lambda: {"listed brand": listed})
    monkeypatch.setattr(
        alternatives,
        "find_medicine_match",
        lambda name: MedicineMatch(listed, 1.0, "exact", "listed brand"),
    )
    monkeypatch.setattr(
        alternatives,
        "fetch_alternatives",
        lambda name: (_ for _ in ()).throw(AssertionError("web lookup must not run")),
    )
    alternatives._composition_to_names.cache_clear()
    result = attach_alternative_candidates({"name": "Listed Brand", "substitutes": [], "review_reasons": []})
    assert "Other Brand Tablet" in result["substitutes"]
    assert "web_alternatives" not in result
    assert result["alternatives_lookup"]["skipped_reason"] == "local_candidates_present"


def test_attach_records_disabled_web_lookup_when_local_list_empty(monkeypatch):
    monkeypatch.setattr("simpliscribe.alternatives.settings", DISABLED_SETTINGS)
    monkeypatch.setattr(alternatives, "dataset_reference_candidates", lambda *args, **kwargs: [])
    result = attach_alternative_candidates({"name": "UnknownBrand", "substitutes": [], "review_reasons": []})
    assert "web_alternatives" not in result
    assert result["alternatives_lookup"]["web_enabled"] is False
    assert result["alternatives_lookup"]["skipped_reason"] == "lookup_disabled"
