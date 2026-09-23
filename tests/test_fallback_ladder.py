from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from simpliscribe import inference
from simpliscribe.config import settings
from simpliscribe.inference import structure_medications
from simpliscribe.main import app
from simpliscribe.ocr import OCRLine, OCRResult
from simpliscribe.storage import append_history, get_analysis_record, get_prescription_file, save_history
from tests.test_app import csrf_for


client = TestClient(app)


def _png_bytes() -> bytes:
    image_buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(image_buffer, format="PNG")
    return image_buffer.getvalue()


def test_huggingface_timeout_falls_back_to_heuristic(monkeypatch):
    object.__setattr__(settings, "inference_provider", "huggingface")
    object.__setattr__(settings, "hf_token", "test-token")

    def boom(_raw_text: str):
        raise TimeoutError("model timeout")

    monkeypatch.setattr("simpliscribe.inference.call_huggingface", boom)
    try:
        result = structure_medications("Paracetamol 650 tab od 5 days")
    finally:
        object.__setattr__(settings, "inference_provider", "fallback")
        object.__setattr__(settings, "hf_token", "")
    assert result["pipeline"]["used_provider"] == "fallback"
    assert result["pipeline"]["degraded"] is True
    assert result["medications"]
    assert result["patient_name"] == "N/A"
    assert any("model was unavailable" in warning for warning in result["pipeline"]["warnings"])


def test_invalid_model_json_falls_back_to_heuristic(monkeypatch):
    object.__setattr__(settings, "inference_provider", "huggingface")
    object.__setattr__(settings, "hf_token", "test-token")

    def bad_payload(_raw_text: str):
        return {"patient_name": "A", "medications": "not-a-list"}

    monkeypatch.setattr("simpliscribe.inference.call_huggingface", bad_payload)
    try:
        result = structure_medications("Paracetamol 650 tab od 5 days")
    finally:
        object.__setattr__(settings, "inference_provider", "fallback")
        object.__setattr__(settings, "hf_token", "")
    assert result["pipeline"]["used_provider"] == "fallback"
    assert isinstance(result["medications"], list)
    assert result["medications"]


def test_heuristic_failure_returns_empty_complete_payload(monkeypatch):
    def boom(_raw_text: str):
        raise RuntimeError("parser crashed")

    monkeypatch.setattr("simpliscribe.inference.fallback_extract", boom)
    result = structure_medications("Paracetamol 650 tab od 5 days")
    assert result["medications"] == []
    assert result["patient_name"] == "N/A"
    assert result["doctor_name"] == "N/A"
    assert result["date"] == "N/A"
    assert result["pipeline"]["human_review_required"] is True
    assert result["pipeline"]["degraded"] is True
    assert result["pipeline"]["error_code"] == "HEURISTIC_FAILED"


def test_unsupported_provider_falls_back_to_heuristic():
    object.__setattr__(settings, "inference_provider", "unknown-provider")
    try:
        result = structure_medications("Paracetamol 650 tab od 5 days")
    finally:
        object.__setattr__(settings, "inference_provider", "fallback")
    assert result["pipeline"]["used_provider"] == "fallback"
    assert result["pipeline"]["error_code"] == "UNSUPPORTED_PROVIDER"
    assert result["medications"]


def test_ocr_engine_failure_without_text_stays_unusable(monkeypatch):
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    def boom(_path):
        raise RuntimeError("paddle crashed")

    monkeypatch.setattr("simpliscribe.web.asyncio.to_thread", run_inline)
    monkeypatch.setattr("simpliscribe.web.extract_ocr_result", boom)
    response = client.post(
        "/api/analyze",
        data={"consent": "true", "csrf": csrf_for(client)},
        files={"file": ("rx.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "UNUSABLE_PRESCRIPTION"


def test_storage_failure_returns_complete_unsaved_payload(monkeypatch):
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("simpliscribe.web.asyncio.to_thread", run_inline)
    monkeypatch.setattr(
        "simpliscribe.web.extract_ocr_result",
        lambda _: OCRResult("Paracetamol 650 tab od 5 days", 0.9, (OCRLine("Paracetamol 650 tab od 5 days", 0.9),), ()),
    )
    monkeypatch.setattr("simpliscribe.web.try_append_history", lambda *args, **kwargs: False)
    response = client.post(
        "/api/analyze",
        data={"consent": "true", "csrf": csrf_for(client)},
        files={"file": ("rx.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 503
    payload = response.json()
    assert payload["review_status"] == "needs_review"
    assert payload["pipeline"]["error_code"] == "STORAGE_FAILED"
    assert payload["medications"] or payload["pipeline"]["human_review_required"] is True
    assert "analysis_id" in payload


def test_structuring_failure_keeps_ocr_and_retries_without_reupload(monkeypatch):
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("simpliscribe.web.asyncio.to_thread", run_inline)
    monkeypatch.setattr("simpliscribe.web.extract_ocr_result", lambda _: OCRResult(
        "Paracetamol 650 tab od 5 days", 0.9,
        (OCRLine("Paracetamol 650 tab od 5 days", 0.9),), (),
    ))
    monkeypatch.setattr("simpliscribe.web.structure_medications", lambda _: (_ for _ in ()).throw(RuntimeError("parser down")))
    response = client.post("/api/analyze", data={"consent": "true", "csrf": csrf_for(client)},
                           files={"file": ("rx.png", _png_bytes(), "image/png")})
    assert response.status_code == 503
    analysis_id = response.json()["analysis_id"]
    saved = get_analysis_record(analysis_id)
    source = get_prescription_file(analysis_id)
    try:
        assert saved["prescription_state"] == "processing_failed"
        assert saved["raw_text"] == "Paracetamol 650 tab od 5 days"
        assert saved["ocr_lines"][0]["confidence"] == 0.9
        assert source is not None
        assert client.get(f"/api/report/{analysis_id}").json()["error_code"] == "REPORT_UNAVAILABLE"
        assert client.get(f"/details/{analysis_id}").status_code == 200

        monkeypatch.setattr("simpliscribe.web.structure_medications", lambda _: {
            "patient_name": "N/A", "doctor_name": "N/A", "date": "N/A",
            "medications": [{"name": "Paracetamol", "dosage": "650 mg", "type": "Tablet", "frequency": "once daily", "duration": "5 days"}],
            "pipeline": {"used_provider": "fallback"},
        })
        retry = client.post(f"/api/analyses/{analysis_id}/retry", headers={"X-CSRF-Token": csrf_for(client)})
        assert retry.status_code == 200
        assert get_analysis_record(analysis_id)["prescription_state"] == "review_required"
        assert get_analysis_record(analysis_id)["raw_text"] == saved["raw_text"]
    finally:
        if source:
            (settings.prescription_storage_dir / source["storage_name"]).unlink(missing_ok=True)


def test_upload_handoff_shows_saved_stage_and_processes_once(monkeypatch):
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    analysis_id = ""
    def fake_ocr(_path):
        saved = get_analysis_record(analysis_id)
        assert saved["prescription_state"] == "processing"
        assert saved["processing_stage"] == "ocr_initializing"
        return OCRResult("Paracetamol 650 tab od 5 days", 0.9,
                         (OCRLine("Paracetamol 650 tab od 5 days", 0.9),), ())

    monkeypatch.setattr("simpliscribe.web.asyncio.to_thread", run_inline)
    monkeypatch.setattr("simpliscribe.web.get_ocr_state", lambda: {"state": "initializing", "ready": False, "error_code": None})
    monkeypatch.setattr("simpliscribe.web.extract_ocr_result", fake_ocr)
    monkeypatch.setattr("simpliscribe.web.structure_medications", lambda _: {
        "patient_name": "N/A", "doctor_name": "N/A", "date": "N/A",
        "medications": [{"name": "Paracetamol", "dosage": "650 mg", "type": "Tablet", "frequency": "once daily", "duration": "5 days"}],
        "pipeline": {"used_provider": "fallback"},
    })
    token = csrf_for(client)
    start = client.post("/api/analyses/start", data={"consent": "true", "csrf": token},
                        files={"file": ("rx.png", _png_bytes(), "image/png")})
    assert start.status_code == 201
    analysis_id = start.json()["analysis_id"]
    source = get_prescription_file(analysis_id)
    try:
        assert source is not None
        assert client.get(f"/api/analyses/{analysis_id}/status").json()["prescription_state"] == "uploaded"
        processing = client.get(f"/details/{analysis_id}")
        assert processing.status_code == 200
        assert "preparing a medicine list" in processing.text
        result = client.post(f"/api/analyses/{analysis_id}/process", headers={"X-CSRF-Token": token})
        assert result.status_code == 200
        assert client.get(f"/api/analyses/{analysis_id}/status").json()["prescription_state"] == "review_required"
        assert client.post(f"/api/analyses/{analysis_id}/process", headers={"X-CSRF-Token": token}).status_code == 409
        assert client.get(f"/details/{analysis_id}").status_code == 200
    finally:
        if source:
            (settings.prescription_storage_dir / source["storage_name"]).unlink(missing_ok=True)


def test_fallback_core_does_not_run_composition_enrichment(monkeypatch):
    monkeypatch.setattr("simpliscribe.inference._attach_alternatives", lambda _: (_ for _ in ()).throw(AssertionError("core fallback should not enrich alternatives")))
    monkeypatch.setattr("simpliscribe.inference.load_optional_reference_fields", lambda: (_ for _ in ()).throw(AssertionError("core fallback should not load optional reference fields")))
    result = structure_medications("Paracetamol 650 tab od 5 days")
    assert result["medications"]


def test_optional_reference_fields_hydrate_on_request(monkeypatch):
    from simpliscribe.inference import MedicineEntry, hydrate_medicine_entry

    entry = MedicineEntry(
        name="Paracetamol", composition="", category="General", dosage_form="Tablet", manufacturer="", pack_size="",
        therapeutic_class="", chemical_class="", action_class="", substitutes=(), uses=(), side_effects=(), sources=(),
    )
    monkeypatch.setattr("simpliscribe.inference.load_optional_reference_fields", lambda: {
        "paracetamol": {"substitutes": ("Dolo 650",), "uses": ("Pain relief",), "side_effects": ("Nausea",)},
    })
    hydrated = hydrate_medicine_entry(entry)
    assert hydrated.substitutes == ("Dolo 650",)
    assert hydrated.uses == ("Pain relief",)
    assert hydrated.side_effects == ("Nausea",)


def test_lexicon_initialization_lock_allows_one_concurrent_build(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from functools import lru_cache

    calls = {"count": 0}

    @lru_cache(maxsize=1)
    def fake_loader():
        calls["count"] += 1
        return {"paracetamol": object()}

    monkeypatch.setattr("simpliscribe.inference._load_medicine_lexicon", fake_loader)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: inference.load_medicine_lexicon(), range(4)))
    assert calls["count"] == 1
    assert all(result is results[0] for result in results)


def test_pdf_builder_failure_returns_unavailable_code(monkeypatch):
    append_history({
        "id": "report-fallback-record",
        "created_at": "2026-08-14T00:00:00+00:00",
        "filename": "rx.png",
        "medications": [],
        "review_status": "needs_review",
    })
    try:
        monkeypatch.setattr("simpliscribe.web.build_pdf_report", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pdf fail")))
        response = client.get("/api/report/report-fallback-record")
        assert response.status_code == 503
        assert response.json()["error_code"] == "REPORT_UNAVAILABLE"
    finally:
        save_history([])
