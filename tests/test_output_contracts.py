from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from simpliscribe.inference import fallback_extract, structure_medications
from simpliscribe.main import app
from simpliscribe.ocr import OCRLine, OCRResult
from simpliscribe.schemas import HEADER_FIELDS, PIPELINE_CORE_FIELDS
from simpliscribe.storage import append_history, save_history
from tests.test_app import csrf_for


client = TestClient(app)

PIPELINE_KEYS = set(PIPELINE_CORE_FIELDS)
HEADER_KEYS = set(HEADER_FIELDS)


def test_fallback_extract_returns_stable_headers_and_medication_list():
    result = fallback_extract("Paracetamol 650 tab od 5 days")
    assert HEADER_KEYS <= set(result)
    assert isinstance(result["medications"], list)
    assert result["medications"]
    medication = result["medications"][0]
    for field in ("name", "type", "dosage", "frequency", "duration", "requires_review"):
        assert field in medication


def test_structure_medications_pipeline_contract():
    result = structure_medications("Paracetamol 650 tab od 5 days")
    assert HEADER_KEYS <= set(result)
    assert PIPELINE_KEYS <= set(result["pipeline"])
    assert result["pipeline"]["human_review_required"] is True
    assert result["pipeline"]["used_provider"] == "fallback"
    assert isinstance(result["medications"], list)


def test_analyze_json_includes_review_and_ocr_pipeline(monkeypatch):
    original_history = []

    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("simpliscribe.web.asyncio.to_thread", run_inline)
    monkeypatch.setattr(
        "simpliscribe.web.extract_ocr_result",
        lambda _: OCRResult("Paracetamol 650 tab od 5 days", 0.91, (OCRLine("Paracetamol 650 tab od 5 days", 0.91),), ()),
    )
    image_buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(image_buffer, format="PNG")
    try:
        response = client.post(
            "/api/analyze",
            data={"consent": "true", "csrf": csrf_for(client)},
            files={"file": ("rx.png", image_buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["review_status"] == "needs_review"
        assert "analysis_id" in payload
        assert payload["pipeline"]["human_review_required"] is True
        assert "ocr_confidence" in payload["pipeline"]
        assert "ocr_warnings" in payload["pipeline"]
        original_history = payload
    finally:
        save_history([])
    assert original_history["id"]


def test_review_returns_status_and_version_contract():
    record = {
        "id": "contract-review-record",
        "created_at": "2026-08-14T00:00:00+00:00",
        "filename": "rx.png",
        "medications": [{"name": "Paracetamol", "type": "Tablet", "dosage": "650 mg", "frequency": "once daily", "duration": "5 days"}],
        "review_status": "needs_review",
    }
    append_history(record)
    try:
        response = client.patch(
            "/api/analyses/contract-review-record/review",
            json={"status": "confirmed"},
            headers={"X-CSRF-Token": csrf_for(client)},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["review_status"] == "confirmed"
        assert payload["review_version"] == 1
        assert payload["analysis_id"] == "contract-review-record"
        conflict = client.patch(
            "/api/analyses/contract-review-record/review",
            json={"status": "corrected", "medications": [{"name": "Paracetamol"}]},
            headers={"X-CSRF-Token": csrf_for(client)},
        )
        assert conflict.status_code in {200, 409}
    finally:
        save_history([])


def test_health_includes_database_ready():
    payload = client.get("/api/health").json()
    assert payload["clinical_use"] == "human_review_required"
    assert payload["database_ready"] is True
    assert "status" in payload


def test_root_module_exports_asgi_app():
    import app as root_app

    assert root_app.app.title
