from io import BytesIO
import re

import fitz
from fastapi.testclient import TestClient
from PIL import Image

from simpliscribe.main import app
from simpliscribe.inference import fallback_extract
from simpliscribe.inference import build_medication_record
from simpliscribe.inference import refine_model_medications
from simpliscribe.ocr import OCRLine, OCRResult
from simpliscribe.storage import append_history, load_history, save_history
from simpliscribe.storage import get_analysis_record
from simpliscribe.config import settings


client = TestClient(app)


def test_dashboard_route():
    response = client.get("/")
    assert response.status_code == 200


def test_history_api_route():
    response = client.get("/api/history")
    assert response.status_code == 200
    assert "analyses" in response.json()


def test_health_route_exposes_review_boundary():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["clinical_use"] == "human_review_required"


def test_authentication_redirects_and_creates_secure_session():
    original_required = settings.auth_required
    original_email = settings.admin_email
    original_password = settings.admin_password
    object.__setattr__(settings, "auth_required", True)
    object.__setattr__(settings, "admin_email", "reviewer@example.test")
    object.__setattr__(settings, "admin_password", "correct horse battery staple")
    auth_client = TestClient(app)
    try:
        response = auth_client.get("/", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"

        login_page = auth_client.get("/login")
        csrf = re.search(r'name="csrf" value="([^"]+)"', login_page.text).group(1)
        login_response = auth_client.post(
            "/login",
            data={"email": "reviewer@example.test", "password": "correct horse battery staple", "csrf": csrf},
            follow_redirects=False,
        )
        assert login_response.status_code == 303
        assert login_response.headers["location"] == "/"
        assert auth_client.get("/").status_code == 200
    finally:
        object.__setattr__(settings, "auth_required", original_required)
        object.__setattr__(settings, "admin_email", original_email)
        object.__setattr__(settings, "admin_password", original_password)


def test_analyze_preserves_parsed_contract_and_pipeline_metadata(monkeypatch):
    original_history = load_history()
    monkeypatch.setattr(
        "simpliscribe.web.extract_ocr_result",
        lambda _: OCRResult(
            "Paracetamol 650 mg tab od 5 days",
            0.94,
            (OCRLine("Paracetamol 650 mg tab od 5 days", 0.94),),
            (),
        ),
    )
    monkeypatch.setattr(
        "simpliscribe.web.structure_medications",
        lambda _: {
            "patient_name": "A Patient",
            "doctor_name": "Dr. B",
            "date": "2026-07-10",
            "medications": [{"name": "Paracetamol", "requires_review": True}],
            "pipeline": {"used_provider": "fallback", "warnings": []},
        },
    )

    image_buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(image_buffer, format="PNG")
    image = image_buffer.getvalue()
    try:
        response = client.post("/api/analyze", data={"consent": "true"}, files={"file": ("rx.png", image, "image/png")})
        assert response.status_code == 200
        payload = response.json()
        assert payload["medications"] == [{"name": "Paracetamol", "requires_review": True}]
        assert payload["patient_name"] == "A Patient"
        assert payload["pipeline"]["ocr_confidence"] == 0.94
        assert payload["review_status"] == "needs_review"
        assert response.headers["cache-control"] == "no-store"
    finally:
        save_history(original_history)


def test_analyze_rejects_spoofed_image_and_removes_upload():
    before = {path.name for path in settings.uploads_dir.iterdir()}

    response = client.post("/api/analyze", data={"consent": "true"}, files={"file": ("fake.png", b"not an image", "image/png")})

    assert response.status_code == 400
    assert "not a valid supported image" in response.json()["detail"]
    assert {path.name for path in settings.uploads_dir.iterdir()} == before


def test_analyze_requires_explicit_consent():
    response = client.post("/api/analyze", files={"file": ("rx.png", b"not an image", "image/png")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Explicit processing consent is required."


def test_storage_isolates_analysis_owners():
    owner_a = "test-owner-a"
    owner_b = "test-owner-b"
    record = {"id": "owner-isolation-record", "created_at": "2026-07-11T00:00:00+00:00", "medications": []}
    try:
        append_history(record, owner_id=owner_a)
        assert get_analysis_record(record["id"], owner_a) == record
        assert get_analysis_record(record["id"], owner_b) is None
    finally:
        save_history([], owner_a)


def test_review_endpoint_updates_owned_analysis():
    original_history = load_history()
    record = {
        "id": "review-test-id",
        "created_at": "2026-07-11T00:00:00+00:00",
        "medications": [{"name": "Paracetmol", "type": "Tablet", "dosage": "650 mg", "frequency": "once daily", "duration": "5 days"}],
    }
    try:
        append_history(record)
        response = client.patch(
            "/api/analyses/review-test-id/review",
            json={"status": "corrected", "medications": [{"name": "Paracetamol"}]},
        )
        assert response.status_code == 200
        updated = get_analysis_record("review-test-id")
        assert updated["review_status"] == "corrected"
        assert updated["medications"][0]["name"] == "Paracetamol"
        assert updated["reviewed_by"] == "local"
    finally:
        save_history(original_history)


def test_analyze_removes_upload_when_ocr_is_unusable(monkeypatch):
    before = {path.name for path in settings.uploads_dir.iterdir()}
    monkeypatch.setattr("simpliscribe.web.extract_ocr_result", lambda _: (_ for _ in ()).throw(ValueError("unusable")))
    image_buffer = BytesIO()
    Image.new("RGB", (2, 2), "white").save(image_buffer, format="PNG")

    response = client.post("/api/analyze", data={"consent": "true"}, files={"file": ("rx.png", image_buffer.getvalue(), "image/png")})

    assert response.status_code == 422
    assert response.json()["error_code"] == "UNUSABLE_PRESCRIPTION"
    assert {path.name for path in settings.uploads_dir.iterdir()} == before


def test_fallback_extract_handles_multiline_prescriptions():
    raw_text = "Paracetamol 650 tab od 5 days\nAmoxycillin 500 cap bd 5 days"
    medications = fallback_extract(raw_text)["medications"]

    assert len(medications) >= 2
    assert medications[0]["name"]
    assert medications[0]["frequency"] == "once daily"


def test_build_medication_record_normalizes_model_output_fields():
    record = build_medication_record(
        name="Paracetamol",
        category="General",
        medication_type="Tabular",
        dosage="650 mg daily for 5 days",
        frequency="od",
        duration="for 5 day",
        insight="Follow the prescription exactly as written.",
        entry=None,
    )

    assert record["type"] == "Tablet"
    assert record["dosage"] == "650 mg"
    assert record["frequency"] == "once daily"
    assert record["duration"] == "5 days"


def test_refine_model_medications_uses_ocr_heuristics_for_shorthand_fields():
    raw_text = "Amoxycillin 500 cap bd 5 days\nCetirizine 10 tab hs 3 days"
    model_medications = [
        {
            "name": "Amoxycillin",
            "category": "General",
            "type": "Tablet",
            "dosage": "500 mg",
            "frequency": "twice daily",
            "duration": "5 days",
            "insight": "Follow the prescription exactly as written.",
        },
        {
            "name": "Cetirizine",
            "category": "General",
            "type": "Tablet",
            "dosage": "10 mg",
            "frequency": "three times daily",
            "duration": "3 days",
            "insight": "Follow the prescription exactly as written.",
        },
    ]

    refined = refine_model_medications(raw_text, model_medications)

    assert refined[0]["type"] == "Capsule"
    assert refined[1]["frequency"] == "at bedtime"


def test_report_download_route_returns_pdf():
    original_history = load_history()
    record = {
        "id": "test-report-id",
        "filename": "sample-prescription.pdf",
        "created_at": "2026-03-11T12:00:00+00:00",
        "raw_text": "Paracetamol 650 tab od 5 days",
        "medications": [
            {
                "name": "Paracetamol",
                "category": "General",
                "type": "Tablet",
                "dosage": "650 mg",
                "frequency": "once daily",
                "duration": "5 days",
                "insight": "Use as directed.",
                "source": "OCR only",
                "source_datasets": [],
                "composition": "Paracetamol",
                "manufacturer": "",
                "pack_size": "",
                "therapeutic_class": "",
                "chemical_class": "",
                "action_class": "",
                "substitutes": [],
                "uses": [],
                "side_effects": [],
            }
        ],
    }

    try:
        append_history(record)
        response = client.get("/api/report/test-report-id")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")
    finally:
        save_history(original_history)


def test_report_download_pdf_contains_expected_content():
    original_history = load_history()
    record = {
        "id": "test-report-content-id",
        "filename": "sample prescription.pdf",
        "created_at": "2026-03-11T12:00:00+00:00",
        "patient_name": "Ananya Sharma",
        "doctor_name": "Dr. Meera Rao",
        "date": "2026-03-11",
        "review_status": "needs_review",
        "pipeline": {"ocr_confidence": 0.87},
        "raw_text": "Paracetamol 650 tab od 5 days",
        "medications": [
            {
                "name": "Paracetamol",
                "category": "Analgesic",
                "type": "Tablet",
                "dosage": "650 mg",
                "frequency": "once daily",
                "duration": "5 days",
                "insight": "Use as directed.",
                "source": "OCR + dataset match",
                "source_datasets": ["India Medicines Dataset"],
                "composition": "Paracetamol",
                "manufacturer": "ABC Pharma",
                "pack_size": "15 tablets",
                "therapeutic_class": "Pain relief",
                "chemical_class": "Anilide",
                "action_class": "Analgesic",
                "substitutes": ["Dolo 650"],
                "uses": ["Fever"],
                "side_effects": ["Nausea"],
                "requires_review": True,
                "review_reasons": ["Medicine name must be confirmed."],
            }
        ],
    }

    try:
        append_history(record)
        response = client.get("/api/report/test-report-content-id")

        assert response.status_code == 200
        assert 'filename="sample_prescription_report.pdf"' in response.headers["content-disposition"]

        document = fitz.open(stream=response.content, filetype="pdf")
        extracted_text = "\n".join(page.get_text() for page in document)

        assert "Prescription Analysis Report" in extracted_text
        assert "Ananya Sharma" in extracted_text
        assert "Dr. Meera Rao" in extracted_text
        assert "87%" in extracted_text
        assert "Paracetamol" in extracted_text
        assert "ABC Pharma" in extracted_text
        assert "Use as directed." in extracted_text
        assert "Paracetamol 650 tab od 5 days" in extracted_text
        assert "Needs review" in extracted_text
        assert "Medicine name must be confirmed." in extracted_text
    finally:
        save_history(original_history)
