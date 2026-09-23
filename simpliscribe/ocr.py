import os
import logging
import time
from contextvars import ContextVar
from math import ceil
from pathlib import Path
from threading import Lock
from typing import Any
from dataclasses import dataclass

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import fitz
from PIL import Image

from .config import settings


_ocr_reader: Any | None = None
_ocr_reader_lock = Lock()
_ocr_inference_lock = Lock()
_ocr_state = "not_started"
_ocr_error_code: str | None = None
ocr_analysis_id: ContextVar[str] = ContextVar("ocr_analysis_id", default="")
logger = logging.getLogger(__name__)
PDF_RENDER_SCALE = 2


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float | None
    lines: tuple[OCRLine, ...]
    warnings: tuple[str, ...]


def _as_confidence(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= 1 else None


def _collect_paddle_lines(results: Any) -> list[OCRLine]:
    """Support PaddleOCR 2.x nested lists and 3.x PaddleX result objects."""
    pages = results if isinstance(results, (list, tuple)) else [results]
    lines: list[OCRLine] = []
    for page in pages:
        texts = None
        scores = None
        if isinstance(page, dict):
            texts, scores = page.get("rec_texts"), page.get("rec_scores")
        else:
            texts = getattr(page, "rec_texts", None)
            scores = getattr(page, "rec_scores", None)
        if isinstance(texts, (list, tuple)):
            score_values = scores if isinstance(scores, (list, tuple)) else []
            for index, value in enumerate(texts):
                text = str(value).strip()
                if text:
                    score = score_values[index] if index < len(score_values) else None
                    lines.append(OCRLine(text, _as_confidence(score)))
            continue

        if not isinstance(page, list):
            continue
        for item in page:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            candidate = item[1]
            if isinstance(candidate, (list, tuple)) and candidate:
                text = str(candidate[0]).strip()
                if text:
                    lines.append(OCRLine(text, _as_confidence(candidate[1] if len(candidate) > 1 else None)))
    return lines


def _collect_paddle_text(results: Any) -> list[str]:
    return [line.text for line in _collect_paddle_lines(results)]


from contextlib import contextmanager


@contextmanager
def _paddle_runtime_environment():
    cache_dir = Path(getattr(settings, "ocr_cache_dir", Path("tmp/ocr-cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    paddle_home = Path(getattr(settings, "paddle_home", cache_dir / "paddle"))
    paddle_home.mkdir(parents=True, exist_ok=True)
    paddlex_cache = Path(getattr(settings, "paddlex_cache_home", cache_dir / "paddlex"))
    paddlex_cache.mkdir(parents=True, exist_ok=True)
    logger.info("ocr_cache_configured=true ocr_cache_writable=true")

    previous = {
        "USERPROFILE": os.environ.get("USERPROFILE"),
        "HOME": os.environ.get("HOME"),
        "PADDLE_HOME": os.environ.get("PADDLE_HOME"),
        "PADDLE_PDX_CACHE_HOME": os.environ.get("PADDLE_PDX_CACHE_HOME"),
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": os.environ.get("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"),
    }
    try:
        os.environ["USERPROFILE"] = str(cache_dir)
        os.environ["HOME"] = str(cache_dir)
        os.environ["PADDLE_HOME"] = str(paddle_home)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddlex_cache)
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def get_ocr_reader() -> Any:
    global _ocr_reader, _ocr_state, _ocr_error_code
    if _ocr_reader is not None:
        return _ocr_reader

    # Paddle backends can fail when initialized concurrently.
    with _ocr_reader_lock:
        if _ocr_reader is not None:
            return _ocr_reader

        _ocr_state = "initializing"
        _ocr_error_code = None
        logger.info("ocr_state=initializing")
        started = time.monotonic()
        import_started = time.monotonic()

        try:
            with _paddle_runtime_environment():
                try:
                    from paddleocr import PaddleOCR
                except ImportError as exc:
                    raise RuntimeError(
                        "PaddleOCR is not installed. Install `paddlepaddle` and `paddleocr` before running OCR."
                    ) from exc
                if analysis_id := ocr_analysis_id.get():
                    logger.info("analysis_id=%s stage=ocr_import duration_ms=%d provider=paddle status=ok", analysis_id, round((time.monotonic() - import_started) * 1000))

            reader = None
            init_errors: list[Exception] = []
            construction_started = time.monotonic()

            with _paddle_runtime_environment():
                # PaddleOCR v3 dropped use_gpu/use_angle_cls in favor of device/use_textline_orientation.
                for kwargs in (
                    {
                        "lang": settings.ocr_language,
                        "device": "gpu" if settings.ocr_use_gpu else "cpu",
                        "enable_mkldnn": False,
                        "use_textline_orientation": True,
                        "show_log": False,
                    },
                    {
                        "lang": settings.ocr_language,
                        "device": "gpu" if settings.ocr_use_gpu else "cpu",
                        "enable_mkldnn": False,
                        "use_textline_orientation": True,
                    },
                    {
                        "lang": settings.ocr_language,
                        "enable_mkldnn": False,
                        "use_angle_cls": True,
                        "use_gpu": settings.ocr_use_gpu,
                        "show_log": False,
                    },
                    {
                        "lang": settings.ocr_language,
                        "enable_mkldnn": False,
                        "use_angle_cls": True,
                        "use_gpu": settings.ocr_use_gpu,
                    },
                ):
                    try:
                        reader = PaddleOCR(**kwargs)
                        break
                    except (TypeError, ValueError) as exc:
                        init_errors.append(exc)

            if reader is None:
                last_error = init_errors[-1] if init_errors else RuntimeError("Unknown PaddleOCR initialization failure.")
                raise RuntimeError(f"Failed to initialize PaddleOCR: {last_error}") from last_error

            _ocr_reader = reader
            _ocr_state = "ready"
            if analysis_id := ocr_analysis_id.get():
                logger.info("analysis_id=%s stage=ocr_reader_construction duration_ms=%d provider=paddle status=ok", analysis_id, round((time.monotonic() - construction_started) * 1000))
                logger.info("analysis_id=%s stage=ocr_initialization duration_ms=%d provider=paddle status=ok", analysis_id, round((time.monotonic() - started) * 1000))
            return _ocr_reader
        except Exception:
            _ocr_state = "failed"
            _ocr_error_code = "OCR_INIT_FAILED"
            logger.warning("stage=ocr_initialization status=error error_code=OCR_INIT_FAILED")
            raise


def get_ocr_state() -> dict[str, Any]:
    with _ocr_reader_lock:
        ready = _ocr_state == "ready" and _ocr_reader is not None
        state = _ocr_state if _ocr_reader is not None or _ocr_state != "ready" else "not_started"
        return {"state": state, "ready": ready, "error_code": _ocr_error_code if state != "not_started" else None}


def warm_ocr_reader() -> bool:
    started = time.monotonic()
    try:
        get_ocr_reader()
        logger.info("stage=ocr_warmup duration_ms=%d status=ready", round((time.monotonic() - started) * 1000))
        return True
    except Exception:
        logger.warning("stage=ocr_warmup duration_ms=%d status=error error_code=OCR_INIT_FAILED", round((time.monotonic() - started) * 1000))
        return False


def _validate_pdf_document(document: fitz.Document) -> None:
    if document.page_count == 0:
        raise ValueError("The uploaded PDF does not contain any pages.")
    if document.page_count > settings.max_pdf_pages:
        raise ValueError(f"PDF has more than the {settings.max_pdf_pages}-page limit.")
    if document.needs_pass:
        raise ValueError("Password-protected PDFs are not supported.")
    total_rendered_pixels = 0
    for page_index in range(document.page_count):
        rect = document.load_page(page_index).rect
        rendered_pixels = ceil(rect.width * PDF_RENDER_SCALE) * ceil(rect.height * PDF_RENDER_SCALE)
        total_rendered_pixels += rendered_pixels
        if rendered_pixels > settings.max_image_pixels or total_rendered_pixels > settings.max_image_pixels:
            raise ValueError("PDF page dimensions are too large to process safely.")


def extract_pdf_pages(file_path: Path) -> list[Path]:
    image_paths: list[Path] = []
    document = fitz.open(file_path)
    try:
        _validate_pdf_document(document)

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            output_path = file_path.with_name(f"{file_path.stem}_page_{page_index + 1}.png")
            image_paths.append(output_path)
            image.save(output_path)
    except Exception:
        for image_path in image_paths:
            image_path.unlink(missing_ok=True)
        raise
    finally:
        document.close()

    return image_paths


def validate_document(file_path: Path) -> None:
    if file_path.suffix.lower() == ".pdf":
        try:
            with fitz.open(file_path) as document:
                _validate_pdf_document(document)
        except (fitz.FileDataError, fitz.EmptyFileError) as exc:
            raise ValueError("The uploaded file is not a valid PDF.") from exc
        return

    try:
        with Image.open(file_path) as image:
            width, height = image.size
            if width * height > settings.max_image_pixels:
                raise ValueError("Image dimensions are too large to process safely.")
            image.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid supported image.") from exc


def extract_ocr_result(file_path: Path) -> OCRResult:
    reader = get_ocr_reader()
    temp_images: list[Path] = []

    try:
        input_paths = [file_path]
        if file_path.suffix.lower() == ".pdf":
            started = time.monotonic()
            input_paths = extract_pdf_pages(file_path)
            if analysis_id := ocr_analysis_id.get():
                logger.info("analysis_id=%s stage=pdf_conversion duration_ms=%d provider=fitz status=ok", analysis_id, round((time.monotonic() - started) * 1000))
            temp_images = input_paths

        lines: list[OCRLine] = []
        engine_failed = False
        started = time.monotonic()
        inference_duration = 0.0
        lock_wait_duration = 0.0
        postprocessing_started = time.monotonic()
        for path in input_paths:
            results = None
            lock_wait_started = time.monotonic()
            with _ocr_inference_lock:
                lock_wait_duration += time.monotonic() - lock_wait_started
                inference_started = time.monotonic()
                try:
                    try:
                        results = reader.ocr(str(path), cls=True)
                    except TypeError:
                        results = reader.ocr(str(path))
                except Exception:
                    engine_failed = True
                    break
                finally:
                    inference_duration += time.monotonic() - inference_started
            if results is not None:
                postprocessing_started = time.monotonic()
                lines.extend(_collect_paddle_lines(results))

        if analysis_id := ocr_analysis_id.get():
            logger.info("analysis_id=%s stage=ocr_lock_wait duration_ms=%d provider=local status=ok", analysis_id, round(lock_wait_duration * 1000))
            logger.info("analysis_id=%s stage=ocr_inference duration_ms=%d provider=paddle status=%s", analysis_id, round(inference_duration * 1000), "error" if engine_failed else "ok")
            logger.info("analysis_id=%s stage=ocr_postprocessing duration_ms=%d provider=local status=%s", analysis_id, round((time.monotonic() - postprocessing_started) * 1000), "error" if engine_failed else "ok")
            logger.info("analysis_id=%s stage=ocr_execution duration_ms=%d provider=paddle status=%s", analysis_id, round((time.monotonic() - started) * 1000), "error" if engine_failed else "ok")

        text = "\n".join(line.text for line in lines)
        scores = [line.confidence for line in lines if line.confidence is not None]
        confidence = sum(scores) / len(scores) if scores else None
        warnings: list[str] = []
        if engine_failed:
            warnings.append("OCR engine failed; the original prescription must be reviewed.")
        if not text.strip():
            warnings.append("No readable text was detected.")
        if confidence is None:
            warnings.append("OCR confidence was not provided by the installed engine.")
        elif confidence < settings.min_ocr_confidence:
            warnings.append("OCR confidence is low; every medication field requires manual review.")
        return OCRResult(text, confidence, tuple(lines), tuple(warnings))
    finally:
        for path in temp_images:
            if path.exists():
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


def extract_ocr_text(file_path: Path) -> str:
    return extract_ocr_result(file_path).text
