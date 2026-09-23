from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import sleep
from types import SimpleNamespace

import fitz
import pytest
from PIL import Image

from simpliscribe import ocr
from simpliscribe.ocr import _collect_paddle_lines, _paddle_runtime_environment, extract_ocr_result, extract_pdf_pages, validate_document


def test_collect_paddle_lines_supports_v2_results():
    results = [[[[[0, 0]], ("Paracetamol 650 mg", 0.93)]]]
    lines = _collect_paddle_lines(results)
    assert [(line.text, line.confidence) for line in lines] == [("Paracetamol 650 mg", 0.93)]


def test_ocr_reader_initializes_once_under_concurrent_first_access(monkeypatch):
    calls = 0

    class FakePaddleOCR:
        def __init__(self, **_kwargs):
            nonlocal calls
            calls += 1

    monkeypatch.setattr(ocr, "_ocr_reader", None)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOCR))

    with ThreadPoolExecutor(max_workers=4) as executor:
        readers = list(executor.map(lambda _item: ocr.get_ocr_reader(), range(4)))

    assert calls == 1
    assert all(reader is readers[0] for reader in readers)


def test_ocr_warmup_failure_can_recover_without_restart(monkeypatch):
    calls = 0

    class FlakyPaddleOCR:
        def __init__(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls < 5:
                raise ValueError("synthetic model failure")

    monkeypatch.setattr(ocr, "_ocr_reader", None)
    monkeypatch.setattr(ocr, "_ocr_state", "not_started")
    monkeypatch.setattr(ocr, "_ocr_error_code", None)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FlakyPaddleOCR))

    assert ocr.warm_ocr_reader() is False
    assert ocr.get_ocr_state()["state"] == "failed"
    assert ocr.warm_ocr_reader() is True
    assert ocr.get_ocr_state()["ready"] is True


def test_collect_paddle_lines_supports_v3_results():
    lines = _collect_paddle_lines([{"rec_texts": ["Paracetamol", "OD"], "rec_scores": [0.98, 0.72]}])
    assert [line.text for line in lines] == ["Paracetamol", "OD"]
    assert [line.confidence for line in lines] == [0.98, 0.72]


def test_paddle_runtime_environment_uses_configured_cache_and_restores_process(monkeypatch, tmp_path):
    import os

    from types import SimpleNamespace

    cache = tmp_path / "ocr-cache"
    monkeypatch.setattr("simpliscribe.ocr.settings", SimpleNamespace(
        ocr_cache_dir=cache, paddle_home=cache / "paddle", paddlex_cache_home=cache / "paddlex"
    ))
    monkeypatch.setenv("USERPROFILE", "original-profile")
    monkeypatch.delenv("PADDLE_HOME", raising=False)
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
    with _paddle_runtime_environment():
        assert os.environ["USERPROFILE"] == str(cache)
        assert os.environ["HOME"] == str(cache)
        assert os.environ["PADDLE_HOME"] == str(cache / "paddle")
        assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(cache / "paddlex")
        assert cache.is_dir()
    assert os.environ["USERPROFILE"] == "original-profile"
    assert "PADDLE_HOME" not in os.environ
    assert "PADDLE_PDX_CACHE_HOME" not in os.environ


def test_settings_resolves_relative_ocr_paths_from_project_root(tmp_path):
    from simpliscribe.config import BASE_DIR, Settings

    relative = Settings(
        ocr_cache_dir=Path("tmp/relative-ocr-cache"),
        paddle_home=Path("tmp/relative-ocr-cache/paddle"),
        paddlex_cache_home=Path("tmp/relative-ocr-cache/paddlex"),
    )
    assert relative.ocr_cache_dir == (BASE_DIR / "tmp/relative-ocr-cache").resolve()
    assert relative.paddle_home == (BASE_DIR / "tmp/relative-ocr-cache/paddle").resolve()
    assert relative.paddlex_cache_home == (BASE_DIR / "tmp/relative-ocr-cache/paddlex").resolve()

    absolute_cache = tmp_path / "absolute-ocr-cache"
    absolute = Settings(
        ocr_cache_dir=absolute_cache,
        paddle_home=absolute_cache / "paddle",
        paddlex_cache_home=absolute_cache / "paddlex",
    )
    assert absolute.ocr_cache_dir == absolute_cache
    assert absolute.paddle_home == absolute_cache / "paddle"
    assert absolute.paddlex_cache_home == absolute_cache / "paddlex"


def test_clean_subprocess_resolves_relative_ocr_cache_from_project_root(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update({
        "APP_ENV": "test",
        "OCR_CACHE_DIR": "tmp/subprocess-ocr-cache",
        "PADDLE_HOME": "tmp/subprocess-ocr-cache/paddle",
        "PADDLE_PDX_CACHE_HOME": "tmp/subprocess-ocr-cache/paddlex",
        "PYTHONPATH": str(root),
    })
    result = subprocess.run(
        [sys.executable, "-c", "from simpliscribe.config import settings; print(settings.ocr_cache_dir); print(settings.paddle_home); print(settings.paddlex_cache_home)"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    assert paths == [
        (root / "tmp/subprocess-ocr-cache").resolve(),
        (root / "tmp/subprocess-ocr-cache/paddle").resolve(),
        (root / "tmp/subprocess-ocr-cache/paddlex").resolve(),
    ]


def test_settings_rejects_unwritable_ocr_cache_path(tmp_path):
    from simpliscribe.config import Settings

    cache_file = tmp_path / "not-a-directory"
    cache_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="OCR_CACHE_DIR must be writable"):
        Settings(ocr_cache_dir=cache_file).validate_runtime()


def test_settings_rejects_unwritable_paddle_override(tmp_path):
    from simpliscribe.config import Settings

    cache_file = tmp_path / "not-a-paddle-directory"
    cache_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PADDLE_HOME must be writable"):
        Settings(paddle_home=cache_file).validate_runtime()


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


def test_validate_document_rejects_pdf_that_would_render_too_large(monkeypatch, tmp_path):
    path = tmp_path / "large-page.pdf"
    document = fitz.open()
    document.new_page(width=100, height=100)
    document.save(path)
    document.close()
    monkeypatch.setattr("simpliscribe.ocr.settings", SimpleNamespace(max_image_pixels=100, max_pdf_pages=10))

    with pytest.raises(ValueError, match="PDF page dimensions are too large"):
        validate_document(path)


def test_validate_document_rejects_pdf_with_excessive_total_render_pixels(monkeypatch, tmp_path):
    path = tmp_path / "many-small-pages.pdf"
    document = fitz.open()
    document.new_page(width=10, height=10)
    document.new_page(width=10, height=10)
    document.save(path)
    document.close()
    monkeypatch.setattr("simpliscribe.ocr.settings", SimpleNamespace(max_image_pixels=500, max_pdf_pages=10))

    with pytest.raises(ValueError, match="PDF page dimensions are too large"):
        validate_document(path)


def test_extract_pdf_pages_removes_partial_images_after_conversion_failure(monkeypatch, tmp_path):
    path = tmp_path / "two-pages.pdf"
    document = fitz.open()
    document.new_page()
    document.new_page()
    document.save(path)
    document.close()
    original_frombytes = Image.frombytes
    calls = 0

    def fail_on_second_page(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Synthetic conversion failure")
        return original_frombytes(*args, **kwargs)

    monkeypatch.setattr("simpliscribe.ocr.Image.frombytes", fail_on_second_page)

    with pytest.raises(RuntimeError, match="Synthetic conversion failure"):
        extract_pdf_pages(path)

    assert not list(tmp_path.glob("two-pages_page_*.png"))


def test_extract_pdf_pages_removes_image_when_save_fails_after_creating_it(monkeypatch, tmp_path):
    path = tmp_path / "one-page.pdf"
    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()

    def write_partial_file_then_fail(_image, output_path, *_args, **_kwargs):
        Path(output_path).write_bytes(b"partial PNG")
        raise RuntimeError("Synthetic save failure")

    monkeypatch.setattr("simpliscribe.ocr.Image.Image.save", write_partial_file_then_fail)

    with pytest.raises(RuntimeError, match="Synthetic save failure"):
        extract_pdf_pages(path)

    assert not list(tmp_path.glob("one-page_page_*.png"))


def test_extract_ocr_result_serializes_shared_reader_calls(monkeypatch, tmp_path):
    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in paths:
        Image.new("RGB", (2, 2), "white").save(path)

    active_calls = 0
    max_active_calls = 0
    state_lock = Lock()

    class Reader:
        def ocr(self, *_args, **_kwargs):
            nonlocal active_calls, max_active_calls
            with state_lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            sleep(0.05)
            with state_lock:
                active_calls -= 1
            return [{"rec_texts": ["Paracetamol"], "rec_scores": [0.99]}]

    monkeypatch.setattr("simpliscribe.ocr.get_ocr_reader", lambda: Reader())
    monkeypatch.setattr("simpliscribe.ocr.settings", SimpleNamespace(min_ocr_confidence=0.8))

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(extract_ocr_result, paths))

    assert max_active_calls == 1


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
