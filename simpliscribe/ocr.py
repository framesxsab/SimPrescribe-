from functools import lru_cache
from pathlib import Path

import easyocr
import fitz
from PIL import Image


@lru_cache(maxsize=1)
def get_ocr_reader() -> easyocr.Reader:
    return easyocr.Reader(["en"], gpu=False)


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
            results = reader.readtext(str(path), detail=0)
            segments.extend(segment.strip() for segment in results if segment.strip())
        return " ".join(segments)
    finally:
        for path in temp_images:
            if path.exists():
                path.unlink()