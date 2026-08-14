from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from simpliscribe.config import settings
from simpliscribe.inference import structure_medications
from simpliscribe.main import app
from simpliscribe.ocr import OCRLine, OCRResult
from simpliscribe.storage import append_history, save_history
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
