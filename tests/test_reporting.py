"""Smoke tests for the PDF report generator."""

from simpliscribe.reporting import build_pdf_report, paragraph, safe_text


def test_build_pdf_report_produces_valid_pdf_with_medication():
    analysis = {
        "id": "report-test",
        "patient_name": "N/A",
        "doctor_name": "N/A",
        "date": "2026-08-12",
        "ocr_confidence": 0.9,
        "provider": "fallback",
        "medications": [
            {
                "name": "Paracetamol",
                "type": "Tablet",
                "dosage": "650 mg",
                "frequency": "once daily",
                "duration": "5 days",
                "insight": "Take as prescribed.",
                "requires_review": False,
                "review_reasons": [],
                "source": "Medicine Database",
                "composition": "",
                "substitutes": [],
                "uses": [],
                "side_effects": [],
            }
        ],
    }
    pdf_bytes = build_pdf_report(analysis, "SimpliScribe")
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_build_pdf_report_renders_web_alternatives_row():
    import re

    import fitz

    analysis = {
        "id": "report-web",
        "patient_name": "N/A",
        "doctor_name": "N/A",
        "date": "N/A",
        "ocr_confidence": 0.9,
        "provider": "fallback",
        "medications": [
            {
                "name": "Oksar",
                "type": "Tablet",
                "dosage": "10 mg",
                "frequency": "once daily",
                "duration": "2 weeks",
                "insight": "Use exactly as prescribed.",
                "requires_review": True,
                "review_reasons": ["Alternative reference candidates were sourced from a model/web search and must be verified by a prescriber."],
                "source": "OCR only",
                "composition": "",
                "substitutes": [],
                "uses": [],
                "side_effects": [],
                "web_alternatives": [
                    {"name": "Montair 10 Tablet", "source": "web", "provider": "duckduckgo", "url": "https://example.com/montair"},
                ],
            }
        ],
    }
    pdf_bytes = build_pdf_report(analysis, "SimpliScribe")
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        text = " ".join(page.get_text() for page in document)
    finally:
        document.close()
    normalized = re.sub(r"\s+", " ", text).upper()
    assert "MONT AIR" not in normalized
    assert "MONTAIR 10 TABLET" in normalized
    assert "VERIFIED BY A PRESCRIBER" in normalized


def test_build_pdf_report_explains_disabled_web_lookup():
    import re
    import fitz

    analysis = {
        "id": "report-disabled",
        "patient_name": "N/A",
        "doctor_name": "N/A",
        "date": "N/A",
        "pipeline": {"requested_provider": "huggingface", "used_provider": "fallback", "degraded": True, "error_code": "PROVIDER_FAILED"},
        "medications": [
            {
                "name": "Imaginarin",
                "type": "Tablet",
                "dosage": "25 mg",
                "frequency": "twice daily",
                "duration": "7 days",
                "requires_review": True,
                "substitutes": [],
                "web_alternatives": [],
                "alternatives_lookup": {
                    "local_count": 0,
                    "web_enabled": False,
                    "web_ran": False,
                    "web_count": 0,
                    "skipped_reason": "lookup_disabled",
                },
            }
        ],
    }
    pdf_bytes = build_pdf_report(analysis, "SimpliScribe")
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        text = " ".join(page.get_text() for page in document)
    finally:
        document.close()
    normalized = re.sub(r"\s+", " ", text).upper()
    assert "WEB/MODEL LOOKUP WAS OFF" in normalized
    assert "USED PROVIDER" in normalized
    assert "FALLBACK" in normalized
    assert "PROVIDER_FAILED" in normalized


def test_paragraph_escapes_html():
    from io import BytesIO

    import fitz
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate

    style = getSampleStyleSheet()["BodyText"]
    buffer = BytesIO()
    SimpleDocTemplate(buffer).build([paragraph("A <script>alert(1)</script> & B", style)])
    document = fitz.open(stream=buffer.getvalue(), filetype="pdf")
    try:
        text = "".join(page.get_text() for page in document)
    finally:
        document.close()
    assert "alert(1)" in text
    assert "<script>" in text


def test_safe_text_falls_back():
    assert safe_text("") == "Not available"
    assert safe_text("x") == "x"
    assert safe_text(None) == "Not available"


def test_display_timestamp_compacts_iso_values():
    from simpliscribe.reporting import display_timestamp

    assert display_timestamp("2026-09-16T04:43:31.456537+00:00") == "16 Sep 2026, 04:43 UTC"


def test_empty_report_explains_missing_structured_medicines():
    import fitz

    pdf_bytes = build_pdf_report({"id": "empty-report", "created_at": "2026-09-16T04:43:31+00:00", "raw_text": "OCR captured text", "medications": []}, "SimpliScribe")
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        text = " ".join(page.get_text() for page in document)
        assert "NO MEDICATIONS" in text and "STRUCTURED" in text
        assert "Review the original prescription" in text
        assert document.page_count >= 2
    finally:
        document.close()


def test_multi_medication_cards_do_not_split_detail_rows():
    import fitz

    medications = []
    for name in ("Paracetamol", "Amoxycillin", "Cetirizine"):
        medications.append({
            "name": name, "type": "Tablet", "category": "General", "dosage": "500 mg",
            "frequency": "once daily", "duration": "3 days", "insight": "Use as prescribed.",
            "requires_review": True, "review_reasons": ["Confirm against original."],
            "composition": f"{name} composition", "source": "OCR only", "uses": ["General"],
        })
    pdf_bytes = build_pdf_report({"id": "multi-layout", "created_at": "2026-09-16T04:43:31+00:00", "raw_text": "line 1\nline 2", "medications": medications}, "SimpliScribe")
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        assert document.page_count == 3
        assert all("Medication summary" not in page.get_text() for page in document[1:])
        assert "Report trace" in document[-1].get_text()
    finally:
        document.close()
