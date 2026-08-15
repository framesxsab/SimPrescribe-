from __future__ import annotations

from pathlib import Path
from simpliscribe.dataset_validator import compute_file_sha256, validate_all_datasets, validate_csv_dataset, validate_golden_json


def test_compute_file_sha256(tmp_path: Path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("SimpliScribe Test Content", encoding="utf-8")

    checksum = compute_file_sha256(sample_file)
    assert len(checksum) == 64
    assert compute_file_sha256(tmp_path / "non_existent.txt") == ""


def test_validate_csv_dataset(tmp_path: Path):
    valid_csv = tmp_path / "valid.csv"
    valid_csv.write_text("name,dosage,type\nDrugA,500mg,Tablet\nDrugB,250mg,Capsule\n", encoding="utf-8")

    res_valid = validate_csv_dataset(valid_csv, "Test Dataset", required_columns=["name", "dosage"], min_rows=2)
    assert res_valid.is_valid
    assert res_valid.row_count == 2
    assert len(res_valid.errors) == 0

    # Missing column
    res_invalid_cols = validate_csv_dataset(valid_csv, "Test Dataset", required_columns=["name", "price"], min_rows=2)
    assert not res_invalid_cols.is_valid
    assert any("Missing required column" in e for e in res_invalid_cols.errors)


def test_validate_golden_json(tmp_path: Path):
    valid_golden = tmp_path / "golden.json"
    valid_golden.write_text(
        '{"schema_version": "1.0", "cases": [{"id": "c1", "expected_medications": []}]}',
        encoding="utf-8",
    )
    res_golden = validate_golden_json(valid_golden)
    assert res_golden.is_valid
    assert res_golden.row_count == 1

    # Invalid schema version
    invalid_golden = tmp_path / "invalid_golden.json"
    invalid_golden.write_text(
        '{"schema_version": "2.0", "cases": []}',
        encoding="utf-8",
    )
    res_bad = validate_golden_json(invalid_golden)
    assert not res_bad.is_valid
    assert any("Unsupported schema_version" in e for e in res_bad.errors)


def test_validate_all_datasets_live():
    res = validate_all_datasets()
    assert "all_valid" in res
    assert "datasets" in res
    assert len(res["datasets"]) == 3
    assert res["all_valid"] is True
