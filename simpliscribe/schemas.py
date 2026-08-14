from __future__ import annotations

from typing import Any, Mapping

HEADER_FIELDS = ("patient_name", "doctor_name", "date")
PIPELINE_CORE_FIELDS = (
    "requested_provider",
    "used_provider",
    "warnings",
    "human_review_required",
)


def empty_extraction_result() -> dict[str, Any]:
    return {
        "patient_name": "N/A",
        "doctor_name": "N/A",
        "date": "N/A",
        "medications": [],
    }


def coerce_header_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"na", "n/a", "none", "unknown"}:
        return "N/A"
    return text


def coerce_extraction_result(result: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = empty_extraction_result()
    if not isinstance(result, Mapping):
        return payload
    for field in HEADER_FIELDS:
        payload[field] = coerce_header_value(result.get(field))
    medications = result.get("medications")
    payload["medications"] = medications if isinstance(medications, list) else []
    if isinstance(result.get("pipeline"), dict):
        payload["pipeline"] = dict(result["pipeline"])
    return payload


def build_pipeline_metadata(
    *,
    requested_provider: str,
    used_provider: str,
    warnings: list[str] | None = None,
    degraded: bool = False,
    error_code: str | None = None,
) -> dict[str, Any]:
    pipeline: dict[str, Any] = {
        "requested_provider": requested_provider,
        "used_provider": used_provider,
        "warnings": list(warnings or []),
        "human_review_required": True,
    }
    if degraded:
        pipeline["degraded"] = True
    if error_code:
        pipeline["error_code"] = error_code
    return pipeline


def analysis_output_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload.setdefault("id", "")
    payload.setdefault("filename", "upload")
    payload.setdefault("created_at", "")
    payload.setdefault("raw_text", "")
    payload["patient_name"] = coerce_header_value(payload.get("patient_name"))
    payload["doctor_name"] = coerce_header_value(payload.get("doctor_name"))
    payload["date"] = coerce_header_value(payload.get("date"))
    medications = payload.get("medications")
    payload["medications"] = medications if isinstance(medications, list) else []
    pipeline = payload.get("pipeline")
    payload["pipeline"] = dict(pipeline) if isinstance(pipeline, dict) else {}
    payload["pipeline"]["human_review_required"] = True
    payload.setdefault("review_status", "needs_review")
    return payload
