from pathlib import Path
from threading import Lock
from typing import Any

import fitz
from PIL import Image

from .config import settings


_ocr_reader: Any | None = None
_ocr_reader_lock = Lock()


def _collect_paddle_text(results: Any) -> list[str]:
    segments: list[str] = []
    
    # Handle the case where results is a single paddle object or dict instead of a list
    if not isinstance(results, (list, tuple)):
        results = [results]

    for page_result in results:
        # PaddleOCR 3.0+ / Paddlex returns objects or dictionaries with a "rec_text" property
        if hasattr(page_result, "keys") and hasattr(page_result, "__getitem__"):
            # Using dict access
            if "rec_text" in page_result:
                texts = page_result["rec_text"]
                if isinstance(texts, list):
                    segments.extend([str(t).strip() for t in texts if str(t).strip()])
                elif isinstance(texts, str) and texts.strip():
                    segments.append(texts.strip())
                continue

        # If it's a PaddleX result object with attributes
        if hasattr(page_result, "rec_text"):
            texts = page_result.rec_text
            if isinstance(texts, list):
                segments.extend([str(t).strip() for t in texts if str(t).strip()])
            elif isinstance(texts, str) and texts.strip():
                segments.append(texts.strip())
            continue

        # Fallback to PaddleOCR 2.x standard List[List[Box, (Text, Score)]]
        if not isinstance(page_result, list):
            continue
            
        for line in page_result:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            candidate = line[1]
            if isinstance(candidate, (list, tuple)) and candidate:
                text = str(candidate[0]).strip()
                if text:
                    segments.append(text)
            elif isinstance(candidate, str) and candidate.strip():
                segments.append(candidate.strip())

    return segments


def get_ocr_reader() -> Any:
    global _ocr_reader
    if _ocr_reader is not None:
        return _ocr_reader

    # Paddle backends can fail when initialized concurrently.
    with _ocr_reader_lock:
        if _ocr_reader is not None:
            return _ocr_reader

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install `paddlepaddle` and `paddleocr` before running OCR."
            ) from exc

        reader = None
        init_errors: list[Exception] = []

        # PaddleOCR v3 dropped use_gpu/use_angle_cls in favor of device/use_textline_orientation.
        for kwargs in (
            {
                "lang": settings.ocr_language,
                "device": "gpu" if settings.ocr_use_gpu else "cpu",
                "use_textline_orientation": True,
                "show_log": False,
                "enable_mkldnn": False,
            },
            {
                "lang": settings.ocr_language,
                "device": "gpu" if settings.ocr_use_gpu else "cpu",
                "use_textline_orientation": True,
                "enable_mkldnn": False,
            },
            {
                "lang": settings.ocr_language,
                "use_angle_cls": True,
                "use_gpu": settings.ocr_use_gpu,
                "show_log": False,
                "enable_mkldnn": False,
            },
            {
                "lang": settings.ocr_language,
                "use_angle_cls": True,
                "use_gpu": settings.ocr_use_gpu,
                "enable_mkldnn": False,
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
        return _ocr_reader


def extract_pdf_pages(file_path: Path) -> list[Path]:
    image_paths: list[Path] = []
    document = fitz.open(file_path)
    try:
        if document.page_count == 0:
            raise ValueError("The uploaded PDF does not contain any pages.")

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            output_path = file_path.with_name(f"{file_path.stem}_page_{page_index + 1}.png")
            image.save(output_path)
            image_paths.append(output_path)
    finally:
        document.close()

    return image_paths


def extract_ocr_text(file_path: Path) -> str:
    reader = get_ocr_reader()
    temp_images: list[Path] = []

    try:
        input_paths = [file_path]
        if file_path.suffix.lower() == ".pdf":
            input_paths = extract_pdf_pages(file_path)
            temp_images = input_paths

        segments: list[str] = []
        for path in input_paths:
            try:
                results = reader.ocr(str(path), cls=True)
            except TypeError:
                results = reader.ocr(str(path))
            segments.extend(_collect_paddle_text(results))
        return " ".join(segments)
    finally:
        for path in temp_images:
            if path.exists():
                path.unlink()