from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inference import structure_medications
from .ocr import extract_ocr_text

DEFAULT_BENCHMARK_CASES = Path(__file__).resolve().parent.parent / "data" / "benchmark_cases.sample.json"
DEFAULT_GOLDEN_CASES = Path(__file__).resolve().parent.parent / "data" / "golden_cases.v1.json"
DEFAULT_BENCHMARK_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark_runs"
SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
SCORABLE_FIELDS = ("name", "type", "dosage", "frequency", "duration")
SUPPORTED_PARQUET_SUFFIXES = {".parquet"}

PARQUET_SECTION_PATTERN = re.compile(r"medications:\s*(.*?)(?:\s*signature:|$)", flags=re.IGNORECASE)
PARQUET_PART_DELIMITER = re.compile(r"\s+-\s+")
DOSAGE_PATTERN = re.compile(
    r"(?P<dosage>\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?(?:\s*(?:mg|ml|mcg|g|iu|units))(?:\s*/\s*\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|iu|units))?)$",
    flags=re.IGNORECASE,
)

PARQUET_FREQUENCY_MAP = {
    "take once daily": "once daily",
    "take twice daily": "twice daily",
    "every 12 hours": "twice daily",
    "every 8 hours": "three times daily",
    "at bedtime": "at bedtime",
    "as needed for pain": "as needed",
    "as directed": "Refer to prescription",
    "after meals": "Refer to prescription",
    "before meals": "Refer to prescription",
    "with food": "Refer to prescription",
}


def normalize_for_score(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_ground_truth_text(raw_text: str) -> str:
    cleaned = str(raw_text or "").replace("<s_ocr>", "").replace("</s>", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_medication_section(raw_text: str) -> str:
    match = PARQUET_SECTION_PATTERN.search(raw_text)
    if not match:
        return ""
    return match.group(1).strip()


def split_medication_parts(medication_section: str) -> list[str]:
    normalized_section = re.sub(r"^\s*-\s*", "", medication_section.strip())
    if not normalized_section:
        return []
    return [part.strip() for part in PARQUET_PART_DELIMITER.split(normalized_section) if part.strip()]


def parse_medication_line(line: str) -> tuple[str, str]:
    normalized_line = re.sub(r"\s+", " ", line.strip())
    match = DOSAGE_PATTERN.search(normalized_line)
    if not match:
        return normalized_line, ""

    name = normalized_line[:match.start()].strip()
    dosage = re.sub(r"\s+", " ", match.group("dosage")).strip()
    return name.title(), dosage


def normalize_instruction(instruction: str) -> str:
    normalized = normalize_for_score(instruction)
    return PARQUET_FREQUENCY_MAP.get(normalized, "Refer to prescription")


def parquet_ground_truth_to_case(raw_text: str, case_id: str, label: str) -> dict[str, Any]:
    normalized_text = normalize_ground_truth_text(raw_text)
    medication_section = extract_medication_section(normalized_text)
    parts = split_medication_parts(medication_section)

    if not parts:
        raise ValueError(f"Parquet benchmark row {case_id} does not contain a parseable medications section.")

    raw_lines: list[str] = []
    expected_medications: list[dict[str, Any]] = []
    index = 0
    while index < len(parts):
        medication_line = parts[index]
        instruction = parts[index + 1] if index + 1 < len(parts) else ""
        name, dosage = parse_medication_line(medication_line)
        raw_lines.append(f"{medication_line} {instruction}".strip())
        expected_medications.append(
            {
                "name": name,
                "type": "",
                "dosage": dosage,
                "frequency": normalize_instruction(instruction),
                "duration": "N/A",
            }
        )
        index += 2

    return {
        "id": case_id,
        "label": label,
        "raw_text": "\n".join(raw_lines),
        "expected_medications": expected_medications,
    }


def load_parquet_cases(cases_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Parquet benchmark input requires pandas and pyarrow installed.") from exc

    dataframe = pd.read_parquet(cases_path)
    if "ground_truth" not in dataframe.columns:
        raise ValueError("Parquet benchmark input must contain a ground_truth column.")

    cases: list[dict[str, Any]] = []
    for row_index, row in enumerate(dataframe.itertuples(index=False), start=1):
        if limit is not None and len(cases) >= limit:
            break
        case_id = f"{cases_path.stem}-{row_index}"
        label = f"{cases_path.stem} row {row_index}"
        cases.append(parquet_ground_truth_to_case(getattr(row, "ground_truth", ""), case_id, label))
    return cases


@dataclass
class CaseScore:
    case_id: str
    label: str
    score: float
    matched_fields: int
    total_fields: int
    expected_count: int
    actual_count: int
    field_results: list[dict[str, Any]]
    raw_text: str
    source_kind: str
    source_path: str
    actual: list[dict[str, Any]]
    medication_results: list[dict[str, Any]]
    tags: list[str]
    expected_rejection: bool
    rejected: bool
    error: str = ""


def load_case_bundle(cases_path: Path, limit: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if cases_path.suffix.lower() in SUPPORTED_PARQUET_SUFFIXES:
        return {"schema_version": "legacy", "dataset": cases_path.stem}, load_parquet_cases(cases_path, limit=limit)

    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {"schema_version": "legacy", "dataset": cases_path.stem}
    if isinstance(payload, dict):
        schema_version = str(payload.get("schema_version") or "")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported benchmark schema_version: {schema_version or 'missing'}.")
        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise ValueError("Versioned benchmark files must contain a cases array.")
        metadata = {key: value for key, value in payload.items() if key != "cases"}
        payload = cases
    if not isinstance(payload, list):
        raise ValueError("Benchmark cases file must contain a JSON array or versioned benchmark object.")
    seen_ids: set[str] = set()
    for index, case in enumerate(payload):
        if not isinstance(case, dict):
            raise ValueError(f"Benchmark case at index {index} must be an object.")
        case_id = str(case.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"Benchmark case at index {index} is missing an id.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate benchmark case id: {case_id}.")
        seen_ids.add(case_id)
        if not isinstance(case.get("expected_medications", []), list):
            raise ValueError(f"Benchmark case {case_id} expected_medications must be a list.")
    if limit is not None:
        payload = payload[:limit]
    return metadata, payload


def load_cases(cases_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    return load_case_bundle(cases_path, limit=limit)[1]


def score_case(case: dict[str, Any], actual: list[dict[str, Any]]) -> CaseScore:
    expected = case.get("expected_medications") or []
    if not isinstance(expected, list):
        raise ValueError("expected_medications must be a list.")

    field_results: list[dict[str, Any]] = []
    medication_results: list[dict[str, Any]] = []
    matched_fields = 0
    total_fields = 0

    # Extraction order is not clinically meaningful. Pair each expected row
    # with the remaining actual row that matches the most labelled fields,
    # then score any unmatched actual rows as false positives.
    unmatched_actual = list(enumerate(actual))
    aligned_rows: list[tuple[dict[str, Any], dict[str, Any], int | None]] = []
    for expected_med in expected:
        if unmatched_actual:
            actual_position, actual_med = max(
                unmatched_actual,
                key=lambda item: sum(
                    normalize_for_score(expected_med.get(field, ""))
                    == normalize_for_score(item[1].get(field, ""))
                    and bool(normalize_for_score(expected_med.get(field, "")))
                    for field in SCORABLE_FIELDS
                ),
            )
            unmatched_actual.remove((actual_position, actual_med))
            aligned_rows.append((expected_med, actual_med, actual_position))
        else:
            aligned_rows.append((expected_med, {}, None))
    aligned_rows.extend(({}, actual_med, actual_position) for actual_position, actual_med in unmatched_actual)

    for index, (expected_med, actual_med, actual_position) in enumerate(aligned_rows):
        name_matched = bool(expected_med) and normalize_for_score(expected_med.get("name")) == normalize_for_score(actual_med.get("name"))
        medication_results.append({
            "expected_medication_index": index if expected_med else None,
            "actual_medication_index": actual_position,
            "name_matched": name_matched,
            "requires_review_expected": bool(expected_med.get("requires_review", False)),
            "requires_review_actual": bool(actual_med.get("requires_review", False)),
        })
        for field in SCORABLE_FIELDS:
            expected_value = normalize_for_score(expected_med.get(field, ""))
            actual_value = normalize_for_score(actual_med.get(field, ""))
            # Expected fields measure extraction accuracy. Fields on an extra
            # medication are false positives and must also count against the
            # score; otherwise a model can hallucinate drugs and still earn 1.0.
            if expected_value or (not expected_med and actual_value):
                total_fields += 1
            matched = expected_value == actual_value and expected_value != ""
            if matched:
                matched_fields += 1
            field_results.append(
                {
                    "medication_index": index,
                    "actual_medication_index": actual_position,
                    "field": field,
                    "matched": matched,
                    "expected": expected_med.get(field, ""),
                    "actual": actual_med.get(field, ""),
                }
            )

    score = 1.0 if total_fields == 0 else matched_fields / total_fields
    has_file_path = bool(str(case.get("file_path") or "").strip())
    return CaseScore(
        case_id=str(case.get("id") or f"case-{case.get('label', 'unknown')}").strip(),
        label=str(case.get("label") or case.get("id") or "Unlabeled case").strip(),
        score=score,
        matched_fields=matched_fields,
        total_fields=total_fields,
        expected_count=len(expected),
        actual_count=len(actual),
        field_results=field_results,
        raw_text=str(case.get("raw_text") or ""),
        source_kind="file" if has_file_path else "text",
        source_path=str(case.get("file_path") or ""),
        actual=actual,
        medication_results=medication_results,
        tags=[str(tag) for tag in case.get("tags", [])],
        expected_rejection=bool(case.get("expected_rejection", False)),
        rejected=bool(case.get("expected_rejection", False) and not actual),
    )


def build_failed_case_score(case: dict[str, Any], error: str) -> CaseScore:
    expected = case.get("expected_medications") or []
    has_file_path = bool(str(case.get("file_path") or "").strip())
    return CaseScore(
        case_id=str(case.get("id") or f"case-{case.get('label', 'unknown')}").strip(),
        label=str(case.get("label") or case.get("id") or "Unlabeled case").strip(),
        score=0.0,
        matched_fields=0,
        total_fields=max(len(expected), 1) * len(SCORABLE_FIELDS),
        expected_count=len(expected) if isinstance(expected, list) else 0,
        actual_count=0,
        field_results=[],
        raw_text=str(case.get("raw_text") or ""),
        source_kind="file" if has_file_path else "text",
        source_path=str(case.get("file_path") or ""),
        actual=[],
        medication_results=[],
        tags=[str(tag) for tag in case.get("tags", [])],
        expected_rejection=bool(case.get("expected_rejection", False)),
        rejected=bool(case.get("expected_rejection", False)),
        error=error,
    )


def resolve_case_raw_text(case: dict[str, Any], base_dir: Path) -> str:
    raw_text = str(case.get("raw_text") or "").strip()
    if raw_text:
        return raw_text

    file_path_value = str(case.get("file_path") or "").strip()
    if not file_path_value:
        raise ValueError(f"Benchmark case {case.get('id', 'unknown')} must include raw_text or file_path.")

    file_path = Path(file_path_value)
    if not file_path.is_absolute():
        file_path = (base_dir / file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark file does not exist: {file_path}")

    return extract_ocr_text(file_path)


def run_case(case: dict[str, Any], base_dir: Path | None = None, retries: int = 1, retry_delay_seconds: float = 1.5) -> CaseScore:
    resolved_base_dir = base_dir or Path.cwd()
    raw_text = resolve_case_raw_text(case, resolved_base_dir)

    last_error = ""
    for attempt in range(retries + 1):
        try:
            result = structure_medications(raw_text)
            actual = result.get("medications", []) if isinstance(result, dict) else result
            if not isinstance(actual, list):
                raise ValueError("Extraction pipeline returned an invalid medication list.")
            hydrated_case = dict(case)
            hydrated_case["raw_text"] = raw_text
            return score_case(hydrated_case, actual)
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(retry_delay_seconds)

    hydrated_case = dict(case)
    hydrated_case["raw_text"] = raw_text
    return build_failed_case_score(hydrated_case, last_error)


def calculate_metrics(case_scores: list[CaseScore]) -> dict[str, Any]:
    true_positives = sum(sum(1 for row in item.medication_results if row["name_matched"]) for item in case_scores)
    expected_total = sum(item.expected_count for item in case_scores)
    actual_total = sum(item.actual_count for item in case_scores)
    false_positives = max(actual_total - true_positives, 0)
    false_negatives = max(expected_total - true_positives, 0)
    precision = true_positives / actual_total if actual_total else (1.0 if expected_total == 0 else 0.0)
    recall = true_positives / expected_total if expected_total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    field_accuracy: dict[str, dict[str, Any]] = {}
    for field in SCORABLE_FIELDS:
        labelled = [
            result for item in case_scores for result in item.field_results
            if result["field"] == field and normalize_for_score(result["expected"])
        ]
        matched = sum(1 for result in labelled if result["matched"])
        field_accuracy[field] = {
            "matched": matched,
            "total": len(labelled),
            "accuracy": round(matched / len(labelled), 4) if labelled else None,
        }

    review_rows = [
        row for item in case_scores for row in item.medication_results if row["requires_review_expected"]
    ]
    rejection_cases = [item for item in case_scores if item.expected_rejection]
    return {
        "medication_detection": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "hallucination_rate": round(false_positives / actual_total, 4) if actual_total else 0.0,
        },
        "field_accuracy": field_accuracy,
        "review_flag_recall": round(
            sum(1 for row in review_rows if row["requires_review_actual"]) / len(review_rows), 4
        ) if review_rows else None,
        "unreadable_rejection_rate": round(
            sum(1 for item in rejection_cases if item.rejected) / len(rejection_cases), 4
        ) if rejection_cases else None,
    }


def run_benchmark(
    cases: list[dict[str, Any]], base_dir: Path | None = None, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    case_scores: list[CaseScore] = []
    for case in cases:
        case_scores.append(run_case(case, base_dir=base_dir))

    average_score = sum(item.score for item in case_scores) / len(case_scores) if case_scores else 0.0
    result = {
        "schema_version": "1.0",
        "dataset": metadata or {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(case_scores),
        "success_count": sum(1 for item in case_scores if not item.error),
        "failure_count": sum(1 for item in case_scores if item.error),
        "average_score": average_score,
        "cases": [
            {
                "id": item.case_id,
                "label": item.label,
                "score": round(item.score, 4),
                "matched_fields": item.matched_fields,
                "total_fields": item.total_fields,
                "expected_count": item.expected_count,
                "actual_count": item.actual_count,
                "source_kind": item.source_kind,
                "source_path": item.source_path,
                "raw_text": item.raw_text,
                "actual": item.actual,
                "medication_results": item.medication_results,
                "field_results": item.field_results,
                "tags": item.tags,
                "expected_rejection": item.expected_rejection,
                "rejected": item.rejected,
                "error": item.error,
            }
            for item in case_scores
        ],
    }
    result["metrics"] = calculate_metrics(case_scores)
    return result


def evaluate_quality_gates(result: dict[str, Any], minimum_f1: float, maximum_hallucination_rate: float) -> list[str]:
    detection = result["metrics"]["medication_detection"]
    failures: list[str] = []
    if detection["f1"] < minimum_f1:
        failures.append(f"medication F1 {detection['f1']:.4f} is below {minimum_f1:.4f}")
    if detection["hallucination_rate"] > maximum_hallucination_rate:
        failures.append(
            f"hallucination rate {detection['hallucination_rate']:.4f} exceeds {maximum_hallucination_rate:.4f}"
        )
    return failures


def save_benchmark_result(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def print_summary(result: dict[str, Any]) -> None:
    print(json.dumps({
        "generated_at": result["generated_at"],
        "case_count": result["case_count"],
        "success_count": result["success_count"],
        "failure_count": result["failure_count"],
        "average_score": round(result["average_score"], 4),
    }, indent=2))
    for case in result["cases"]:
        suffix = f" error={case['error']}" if case.get("error") else ""
        print(f"- {case['label']}: score={case['score']:.4f} ({case['matched_fields']}/{case['total_fields']}){suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local extraction benchmark against sample prescription cases.")
    parser.add_argument("--cases", default=str(DEFAULT_BENCHMARK_CASES), help="Path to benchmark cases JSON file.")
    parser.add_argument("--output", default="", help="Optional path to write benchmark results JSON.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of cases to run. Use 0 for all cases.")
    parser.add_argument("--min-f1", type=float, default=0.0, help="Fail when medication-name F1 is below this value.")
    parser.add_argument("--max-hallucination-rate", type=float, default=1.0, help="Fail when the hallucination rate exceeds this value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    output_path = Path(args.output) if args.output else DEFAULT_BENCHMARK_OUTPUT_DIR / "latest.json"

    metadata, cases = load_case_bundle(cases_path, limit=args.limit or None)
    result = run_benchmark(cases, base_dir=cases_path.parent, metadata=metadata)
    save_benchmark_result(result, output_path)
    print_summary(result)
    print(f"Saved benchmark report to {output_path}")
    failures = evaluate_quality_gates(result, args.min_f1, args.max_hallucination_rate)
    if failures:
        raise SystemExit("Benchmark quality gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
