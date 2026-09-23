from simpliscribe import inference
from simpliscribe.inference import MedicineEntry
from simpliscribe.lexicon_index import RequiredLexiconIndex, is_current, write_index


def _entry(name: str) -> MedicineEntry:
    return MedicineEntry(
        name=name,
        composition="Ibuprofen 200 mg",
        category="Analgesics",
        dosage_form="Tablet",
        manufacturer="",
        pack_size="",
        therapeutic_class="Pain",
        chemical_class="",
        action_class="",
        substitutes=(),
        uses=(),
        side_effects=(),
        sources=("test",),
    )


def test_precomputed_index_matches_csv_lookup_semantics(monkeypatch, tmp_path):
    entry = _entry("Ibuprofen Tablet")
    lexicon = {"ibuprofen": entry, "ibu": entry}
    path = tmp_path / "medicine_lexicon.sqlite"
    write_index(path, lexicon, "test-fingerprint")
    index = RequiredLexiconIndex(path)

    monkeypatch.setattr(inference, "load_persistent_lexicon_index", lambda: None)
    monkeypatch.setattr(inference, "load_medicine_lexicon", lambda: lexicon)
    csv_match = inference.find_medicine_match("Ibuprofen 200 mg")
    monkeypatch.setattr(inference, "load_persistent_lexicon_index", lambda: index)
    indexed_match = inference.find_medicine_match("Ibuprofen 200 mg")

    assert csv_match is not None and indexed_match is not None
    assert csv_match.method == indexed_match.method == "exact"
    assert csv_match.matched_alias == indexed_match.matched_alias == "ibuprofen"
    assert indexed_match.entry == csv_match.entry


def test_index_fingerprint_validation_rejects_untrusted_metadata(tmp_path):
    path = tmp_path / "medicine_lexicon.sqlite"
    write_index(path, {"ibuprofen": _entry("Ibuprofen Tablet")}, "wrong-fingerprint")
    assert not is_current(path)
