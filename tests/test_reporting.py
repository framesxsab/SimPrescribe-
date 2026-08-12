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
