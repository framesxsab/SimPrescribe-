"""ASGI entry point for accessibility checks that do not exercise OCR."""

from __future__ import annotations

import time

import simpliscribe.main as main
from simpliscribe.inference import hydrate_medication_references


def _skip_ocr_warmup() -> None:
    return None


# Keep development mode so details pages use the required SQLite medicine index.
# Background OCR model loading is unrelated to these rendered-page checks.
main.warm_ocr_reader = _skip_ocr_warmup
app = main.app

# A cold details request hydrates optional medicine references synchronously. Warm the
# same shared read cache used by the synthetic details fixture before browser checks.
_warmup_started = time.perf_counter()
hydrate_medication_references([{"name": "Paracetamol 500"}])
print(f"A11y fixture medicine-reference cache ready in {time.perf_counter() - _warmup_started:.2f}s", flush=True)
