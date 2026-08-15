from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class DatasetValidationResult:
    dataset_name: str
    file_path: str
    exists: bool
    size_bytes: int
    row_count: int
    sha256: str
    is_valid: bool
    errors: list[str]


def compute_file_sha256(path: Path, max_bytes: int = 20 * 1024 * 1024) -> str:
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        bytes_read = 0
        while chunk := f.read(65536):
            hasher.update(chunk)
            bytes_read += len(chunk)
            if bytes_read >= max_bytes:
                break
    return hasher.hexdigest()


def validate_csv_dataset(
    path: Path,
    dataset_name: str,
    required_columns: list[str],
    min_rows: int = 10,
) -> DatasetValidationResult:
    errors: list[str] = []
    if not path.exists():
        return DatasetValidationResult(
            dataset_name=dataset_name,
            file_path=str(path),
            exists=False,
            size_bytes=0,
            row_count=0,
            sha256="",
            is_valid=False,
            errors=[f"Dataset file does not exist at {path}"],
        )

    size = path.stat().st_size
    row_count = 0
    sha256_hash = compute_file_sha256(path)

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                errors.append("CSV file is empty or missing a header row.")
            else:
                cleaned_header = [col.strip().lower() for col in header]
                for req in required_columns:
                    if req.lower() not in cleaned_header:
                        errors.append(f"Missing required column: '{req}'")

            for _ in reader:
                row_count += 1

        if row_count < min_rows:
            errors.append(f"Row count {row_count} is less than minimum expected {min_rows}.")
    except Exception as exc:
        errors.append(f"Failed to read or parse CSV: {exc}")

    is_valid = len(errors) == 0
    return DatasetValidationResult(
        dataset_name=dataset_name,
        file_path=str(path),
        exists=True,
        size_bytes=size,
        row_count=row_count,
        sha256=sha256_hash,
        is_valid=is_valid,
        errors=errors,
    )


def validate_golden_json(path: Path) -> DatasetValidationResult:
    errors: list[str] = []
    if not path.exists():
        return DatasetValidationResult(
            dataset_name="Golden Cases JSON",
            file_path=str(path),
            exists=False,
            size_bytes=0,
            row_count=0,
            sha256="",
            is_valid=False,
            errors=[f"Golden cases file not found at {path}"],
        )

    size = path.stat().st_size
    sha256_hash = compute_file_sha256(path)
    case_count = 0

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            errors.append("Golden cases file must be a JSON object.")
        else:
            if payload.get("schema_version") != "1.0":
                errors.append(f"Unsupported schema_version: {payload.get('schema_version')}")
            cases = payload.get("cases", [])
            if not isinstance(cases, list) or len(cases) == 0:
                errors.append("Missing or empty 'cases' list.")
            else:
                case_count = len(cases)
                for idx, c in enumerate(cases):
                    if not c.get("id"):
                        errors.append(f"Case at index {idx} missing 'id'.")
                    if "expected_medications" not in c:
                        errors.append(f"Case {c.get('id', idx)} missing 'expected_medications'.")
    except Exception as exc:
        errors.append(f"Failed to parse golden cases JSON: {exc}")

    is_valid = len(errors) == 0
    return DatasetValidationResult(
        dataset_name="Golden Cases JSON",
        file_path=str(path),
        exists=True,
        size_bytes=size,
        row_count=case_count,
        sha256=sha256_hash,
        is_valid=is_valid,
        errors=errors,
    )


def validate_all_datasets() -> dict[str, Any]:
    india_res = validate_csv_dataset(
        settings.india_medicine_dataset,
        "A-Z Medicines Dataset of India",
        required_columns=["name", "price(₹)", "is_discontinued", "manufacturer_name", "type", "pack_size_label", "short_composition1"],
        min_rows=1000,
    )
    all_meds_res = validate_csv_dataset(
        settings.medicine_database_dataset,
        "All Medicine Database",
        required_columns=[],
        min_rows=1000,
    )
    golden_res = validate_golden_json(settings.root_dir / "data" / "golden_cases.v1.json")

    all_valid = india_res.is_valid and all_meds_res.is_valid and golden_res.is_valid
    return {
        "all_valid": all_valid,
        "datasets": [
            {
                "name": r.dataset_name,
                "path": r.file_path,
                "exists": r.exists,
                "size_mb": round(r.size_bytes / (1024 * 1024), 2),
                "row_count": r.row_count,
                "sha256_prefix": r.sha256[:16] if r.sha256 else "",
                "valid": r.is_valid,
                "errors": r.errors,
            }
            for r in (india_res, all_meds_res, golden_res)
        ],
    }
