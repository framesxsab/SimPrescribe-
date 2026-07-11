from types import SimpleNamespace

import fitz
import pytest
from PIL import Image

from simpliscribe.ocr import _collect_paddle_lines, extract_ocr_result, validate_document


def test_collect_paddle_lines_supports_v2_results():
    results = [[[[[0, 0]], ("Paracetamol 650 mg", 0.93)]]]
    lines = _collect_paddle_lines(results)
    assert [(line.text, line.confidence) for line in lines] == [("Paracetamol 650 mg", 0.93)]


def test_collect_paddle_lines_supports_v3_results():
    lines = _collect_paddle_lines([{"rec_texts": ["Paracetamol", "OD"], "rec_scores": [0.98, 0.72]}])
    assert [line.text for line in lines] == ["Paracetamol", "OD"]
    assert [line.confidence for line in lines] == [0.98, 0.72]


def test_validate_document_rejects_spoofed_image(tmp_path):
    path = tmp_path / "spoofed.png"
    path.write_bytes(b"this is not an image")

    with pytest.raises(ValueError, match="not a valid supported image"):
        validate_document(path)


def test_validate_document_rejects_oversized_image(monkeypatch, tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (3, 3), "white").save(path)
    monkeypatch.setattr("simpliscribe.ocr.settings", SimpleNamespace(max_image_pixels=4, max_pdf_pages=10))

    with pytest.raises(ValueError, match="dimensions are too large"):
        validate_document(path)


def test_validate_document_enforces_pdf_page_limit(monkeypatch, tmp_path):
    path = tmp_path / "long.pdf"
    document = fitz.open()
    document.new_page()
    document.new_page()
    document.save(path)
    document.close()
    monkeypatch.setattr("simpliscribe.ocr.settings", SimpleNamespace(max_image_pixels=100, max_pdf_pages=1))

    with pytest.raises(ValueError, match="1-page limit"):
        validate_document(path)


def test_extract_ocr_result_marks_low_confidence(monkeypatch, tmp_path):
    path = tmp_path / "rx.png"
    Image.new("RGB", (2, 2), "white").save(path)

    class Reader:
        def ocr(self, *_args, **_kwargs):
            return [{"rec_texts": ["Paracetamol 650 mg"], "rec_scores": [0.42]}]

    monkeypatch.setattr("simpliscribe.ocr.get_ocr_reader", lambda: Reader())
    monkeypatch.setattr("simpliscribe.ocr.settings", SimpleNamespace(min_ocr_confidence=0.8))

    result = extract_ocr_result(path)

    assert result.confidence == 0.42
    assert "OCR confidence is low" in result.warnings[0]
