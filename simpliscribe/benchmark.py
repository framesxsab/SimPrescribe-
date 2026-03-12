from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inference import structure_medications
from .ocr import extract_ocr_text

DEFAULT_BENCHMARK_CASES = Path(__file__).resolve().parent.parent / "data" / "benchmark_cases.sample.json"
DEFAULT_BENCHMARK_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark_runs"
SCORABLE_FIELDS = ("name", "type", "dosage", "frequency", "duration")


def normalize_for_score(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


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
    error: str = ""


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark cases file must contain a JSON array.")
    return payload


def score_case(case: dict[str, Any], actual: list[dict[str, Any]]) -> CaseScore:
    expected = case.get("expected_medications") or []
    if not isinstance(expected, list):
        raise ValueError("expected_medications must be a list.")

    field_results: list[dict[str, Any]] = []
    matched_fields = 0
    total_fields = max(len(expected), len(actual)) * len(SCORABLE_FIELDS)

    for index in range(max(len(expected), len(actual))):
        expected_med = expected[index] if index < len(expected) else {}
        actual_med = actual[index] if index < len(actual) else {}
        for field in SCORABLE_FIELDS:
            expected_value = normalize_for_score(expected_med.get(field, ""))
            actual_value = normalize_for_score(actual_med.get(field, ""))
            matched = expected_value == actual_value and expected_value != ""
            if matched:
                matched_fields += 1
            field_results.append(
                {
                    "medication_index": index,
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
            actual = structure_medications(raw_text)
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


def run_benchmark(cases: list[dict[str, Any]], base_dir: Path | None = None) -> dict[str, Any]:
    case_scores: list[CaseScore] = []
    for case in cases:
        case_scores.append(run_case(case, base_dir=base_dir))

    average_score = sum(item.score for item in case_scores) / len(case_scores) if case_scores else 0.0
    return {
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
                "field_results": item.field_results,
                "error": item.error,
            }
            for item in case_scores
        ],
    }


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    output_path = Path(args.output) if args.output else DEFAULT_BENCHMARK_OUTPUT_DIR / "latest.json"

    cases = load_cases(cases_path)
    result = run_benchmark(cases, base_dir=cases_path.parent)
    save_benchmark_result(result, output_path)
    print_summary(result)
    print(f"Saved benchmark report to {output_path}")


if __name__ == "__main__":
    main()