from __future__ import annotations

import asyncio
import csv
import hashlib
import threading
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from httpx import ASGITransport

from simpliscribe import inference, optional_reference_index
from simpliscribe.inference import MedicineEntry, MedicineMatch
from simpliscribe.main import app
from simpliscribe.optional_reference_index import (
    build_index,
    is_current,
    open_current_index,
)


def _source_csv(path: Path) -> None:
    fields = [
        "name",
        *(f"substitute{index}" for index in range(5)),
        *(f"use{index}" for index in range(5)),
        *(f"sideEffect{index}" for index in range(42)),
    ]
    rows = [
        {
            "name": "Paracetamol 500 mg",
            "substitute0": "Dolo 650",
            "substitute1": "Calpol 650",
            "use0": "Pain relief",
            "sideEffect0": "Nausea",
        },
        {
            "name": "Paracetamol-500 mg",
            "substitute0": "Dolo 650",
            "substitute2": "Crocin 650",
            "use0": "Pain relief",
            "use1": "Fever reduction",
            "sideEffect0": "Nausea",
            "sideEffect1": "Dizziness",
        },
        {
            "name": "Ibuprofen 200 mg",
            "substitute0": "Brufen 200",
            "use0": "Pain relief",
            "sideEffect0": "Stomach upset",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _entry(name: str = "Paracetamol 500 mg") -> MedicineEntry:
    return MedicineEntry(
        name=name,
        composition="Paracetamol",
        category="General",
        dosage_form="Tablet",
        manufacturer="",
        pack_size="",
        therapeutic_class="Analgesic",
        chemical_class="",
        action_class="",
        substitutes=(),
        uses=(),
        side_effects=(),
        sources=("India Medicines Dataset",),
    )


def test_optional_index_preserves_legacy_field_merge_and_provenance(tmp_path):
    source = tmp_path / "medicine references.csv"
    index_path = tmp_path / "optional.sqlite"
    _source_csv(source)

    result = build_index(index_path, source)
    assert result["source_rows"] == 3
    index = open_current_index(index_path, source)
    assert index is not None

    fields = index.lookup("paracetamol 500 mg")
    assert fields == {
        "substitutes": ("Dolo 650", "Calpol 650", "Crocin 650"),
        "uses": ("Pain relief", "Fever reduction"),
        "side_effects": ("Nausea", "Dizziness"),
        "provenance": ("Medicine Database",),
    }
    assert index.lookup("unknown medicine") is None


def test_optional_index_build_is_deterministic_and_atomic(tmp_path):
    source = tmp_path / "medicine references.csv"
    index_path = tmp_path / "optional.sqlite"
    _source_csv(source)

    build_index(index_path, source)
    first_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    build_index(index_path, source)
    second_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()

    assert first_hash == second_hash
    assert not list(tmp_path.glob(".optional.sqlite.*.tmp"))


def test_source_fingerprint_invalidates_stale_index(tmp_path):
    source = tmp_path / "medicine references.csv"
    index_path = tmp_path / "optional.sqlite"
    _source_csv(source)
    build_index(index_path, source)

    assert is_current(index_path, source)
    with source.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert not is_current(index_path, source)
    assert open_current_index(index_path, source) is None


def test_process_cache_revalidates_after_source_change(tmp_path, monkeypatch):
    source = tmp_path / "medicine references.csv"
    index_path = tmp_path / "optional.sqlite"
    _source_csv(source)
    build_index(index_path, source)
    monkeypatch.setattr(optional_reference_index, "default_index_path", lambda: index_path)
    monkeypatch.setattr(optional_reference_index, "SOURCE_DATASET", source)
    optional_reference_index.clear_index_cache()

    try:
        assert optional_reference_index.load_current_index() is not None
        with source.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        assert optional_reference_index.load_current_index() is None
    finally:
        optional_reference_index.clear_index_cache()


def test_missing_or_corrupt_optional_index_is_unavailable(tmp_path):
    source = tmp_path / "medicine references.csv"
    index_path = tmp_path / "optional.sqlite"
    _source_csv(source)

    assert not is_current(index_path, source)
    assert open_current_index(index_path, source) is None

    index_path.write_text("not a sqlite database", encoding="utf-8")
    assert not is_current(index_path, source)
    assert open_current_index(index_path, source) is None


def test_hydration_uses_one_targeted_index_lookup_without_csv_read(monkeypatch):
    class CountingIndex:
        def __init__(self):
            self.names = []

        def lookup(self, name):
            self.names.append(name)
            return {
                "substitutes": ("Dolo 650",),
                "uses": ("Pain relief",),
                "side_effects": ("Nausea",),
                "provenance": ("Medicine Database",),
            }

    index = CountingIndex()
    entry = _entry()
    match = MedicineMatch(entry, 1.0, "exact", "paracetamol 500 mg")
    monkeypatch.setattr("simpliscribe.optional_reference_index.load_current_index", lambda: index)
    monkeypatch.setattr(inference, "find_medicine_match", lambda _name: match)
    monkeypatch.setattr(
        inference,
        "_dataset_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("CSV read during indexed hydration")),
    )

    medications = inference.hydrate_medication_references([{"name": entry.name}])
    assert index.names == ["paracetamol 500 mg"]
    assert medications[0]["substitutes"] == ["Dolo 650"]
    assert medications[0]["uses"] == ["Pain relief"]
    assert medications[0]["side_effects"] == ["Nausea"]
    assert "optional_references_unavailable" not in medications[0]


def test_missing_optional_index_keeps_details_renderable(monkeypatch):
    record = {
        "id": "synthetic-optional-reference-missing",
        "filename": "synthetic-prescription.png",
        "prescription_state": "confirmed",
        "patient_review_status": "confirmed",
        "patient_name": "Synthetic Patient",
        "doctor_name": "Synthetic Clinician",
        "date": "N/A",
        "medications": [{
            "name": "Paracetamol 500 mg",
            "type": "Tablet",
            "dosage": "500 mg",
            "frequency": "Once daily",
            "duration": "3 days",
        }],
    }
    match = MedicineMatch(_entry(), 1.0, "exact", "paracetamol 500 mg")
    monkeypatch.setattr("simpliscribe.web.get_analysis_record", lambda *_args, **_kwargs: record.copy())
    monkeypatch.setattr(inference, "find_medicine_match", lambda _name: match)
    monkeypatch.setattr("simpliscribe.optional_reference_index.load_current_index", lambda: None)
    monkeypatch.setattr(inference, "_warn_optional_reference_unavailable", lambda: None)

    response = TestClient(app).get("/details/synthetic-optional-reference-missing")
    assert response.status_code == 200
    assert "Paracetamol 500 mg" in response.text
    assert "Optional medicine reference details are temporarily unavailable" in response.text
    assert "Traceback" not in response.text
    assert str(Path.cwd()) not in response.text


def test_review_page_does_not_initialize_optional_references(monkeypatch):
    record = {
        "id": "synthetic-review-no-optional-index",
        "filename": "synthetic-review.png",
        "prescription_state": "review_required",
        "patient_review_status": "needs_review",
        "medications": [{"name": "Paracetamol", "type": "Tablet", "frequency": "Once daily"}],
        "original_medications": [{"name": "Paracetamol", "type": "Tablet", "frequency": "Once daily", "requires_review": False}],
    }
    monkeypatch.setattr("simpliscribe.web.get_analysis_record", lambda *_args, **_kwargs: record.copy())
    monkeypatch.setattr(
        "simpliscribe.optional_reference_index.load_current_index",
        lambda: (_ for _ in ()).throw(AssertionError("review path opened optional references")),
    )

    response = TestClient(app).get("/details/synthetic-review-no-optional-index")
    assert response.status_code == 200
    assert "Check your medicine list" in response.text


def test_independent_live_request_completes_while_optional_work_is_pending(monkeypatch):
    record = {
        "id": "synthetic-concurrent-details",
        "filename": "synthetic-prescription.png",
        "prescription_state": "confirmed",
        "patient_review_status": "confirmed",
        "medications": [{"name": "Paracetamol"}],
    }
    entered = threading.Event()
    release = threading.Event()

    def slow_hydration(medications):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("test watchdog expired")
        return medications

    monkeypatch.setattr("simpliscribe.web.get_analysis_record", lambda *_args, **_kwargs: record.copy())
    monkeypatch.setattr("simpliscribe.web.hydrate_medication_references", slow_hydration)

    async def run_requests():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            details_task = asyncio.create_task(client.get("/details/synthetic-concurrent-details"))
            entered_in_worker = await asyncio.to_thread(entered.wait, 5)
            assert entered_in_worker

            watchdog = threading.Timer(5, release.set)
            watchdog.start()
            try:
                live_response = await client.get("/api/live")
                completed_before_release = not release.is_set()
            finally:
                release.set()
                watchdog.cancel()

            details_response = await details_task
            return live_response, details_response, completed_before_release

    live, details, completed_before_release = asyncio.run(run_requests())
    assert live.status_code == 200
    assert details.status_code == 200
    assert completed_before_release
