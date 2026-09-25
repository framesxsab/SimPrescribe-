"""ASGI entry point for accessibility checks that do not exercise OCR."""

from __future__ import annotations


import simpliscribe.main as main


def _skip_ocr_warmup() -> None:
    return None


# Keep development mode so details pages use the required SQLite medicine index.
# Background OCR model loading is unrelated to these rendered-page checks.
main.warm_ocr_reader = _skip_ocr_warmup
app = main.app
