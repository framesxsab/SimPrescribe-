import fitz
from fastapi.testclient import TestClient

from simpliscribe.main import app
from simpliscribe.inference import fallback_extract
from simpliscribe.inference import build_medication_record
from simpliscribe.inference import refine_model_medications
from simpliscribe.storage import append_history, load_history, save_history


client = TestClient(app)


def test_dashboard_route():
    response = client.get("/")
    assert response.status_code == 200


def test_history_api_route():
    response = client.get("/api/history")
    assert response.status_code == 200
    assert "analyses" in response.json()


def test_fallback_extract_handles_multiline_prescriptions():
    raw_text = "Paracetamol 650 tab od 5 days\nAmoxycillin 500 cap bd 5 days"
    medications = fallback_extract(raw_text)

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
        assert "Paracetamol" in extracted_text
        assert "ABC Pharma" in extracted_text
        assert "Use as directed." in extracted_text
        assert "Paracetamol 650 tab od 5 days" in extracted_text
    finally:
        save_history(original_history)