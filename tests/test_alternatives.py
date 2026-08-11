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


def test_attach_does_not_run_when_local_substitutes_exist(enabled, monkeypatch):
    payload = {"name": "Augmentin 625 Duo Tablet", "substitutes": ["Penciclav 500 mg/125 mg Tablet"]}
    monkeypatch.setattr(alternatives, "fetch_alternatives", lambda name: [{"name": "X", "source": "web", "provider": "duckduckgo", "url": ""}])
    result = attach_alternative_candidates(dict(payload))
    assert result == payload


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
