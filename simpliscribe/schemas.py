from __future__ import annotations

from typing import Any, Mapping
from pydantic import BaseModel, Field

HEADER_FIELDS = ("patient_name", "doctor_name", "date")
PIPELINE_CORE_FIELDS = (
    "requested_provider",
    "used_provider",
    "warnings",
    "human_review_required",
)


class LiveResponse(BaseModel):
    status: str = Field(default="alive", description="Service liveness state.")


class HealthResponse(BaseModel):
    status: str = Field(description="Service readiness state ('ready' or 'degraded').")
    datasets_ready: bool = Field(description="Whether local medicine reference datasets are loaded.")
    database_ready: bool = Field(description="Whether primary database ping succeeded.")
    configured_provider: str = Field(description="Inference provider setting (huggingface/fallback/local).")
    provider_ready: bool = Field(description="Whether the configured inference credentials/model are ready.")
    clinical_use: str = Field(default="human_review_required", description="Clinical safety statement.")
    authentication_required: bool = Field(description="Whether session/OIDC auth is enforced.")


class SimilarPrescriptionItem(BaseModel):
    id: str = Field(description="Unique identifier of matching prescription.")
    similarity: float = Field(description="Cosine similarity score (0.0 to 1.0).")
    raw_text: str = Field(description="Original or synthesized prescription text.")
    medicines: list[dict[str, Any]] = Field(default_factory=list, description="Extracted medications list.")
    source: str = Field(default="", description="Provenance dataset source.")
    tags: list[str] = Field(default_factory=list, description="Diagnostic tags.")


class SimilarPrescriptionsResponse(BaseModel):
    query: str = Field(description="Query string searched.")
    count: int = Field(description="Number of matching results returned.")
    results: list[SimilarPrescriptionItem] = Field(description="Top-k matching prescription records.")


class CacheStatsResponse(BaseModel):
    in_memory_entries: int = Field(description="Active vector index items in memory.")
    exact_cache_size: int = Field(description="Exact hash lookup cache items.")
    session_hits: int = Field(description="Total cache hits during current server session.")
    session_misses: int = Field(description="Total cache misses during current server session.")
    hit_ratio: float = Field(description="Cache hit ratio (0.0 to 1.0).")
    total_db_entries: int | None = Field(default=0, description="Total cached records in persistent DB.")
    total_db_hits: int | None = Field(default=0, description="Historical cumulative DB cache hits.")


class ReviewRequest(BaseModel):
    status: str = Field(description="Review decision ('confirmed', 'corrected', 'rejected').")
    medications: list[dict[str, Any]] | None = Field(default=None, description="Optional medication field corrections.")


class ReviewResponse(BaseModel):
    analysis_id: str = Field(description="Analysis UUID.")
    review_status: str = Field(description="Updated review status.")
    reviewed_at: str = Field(description="ISO timestamp of review.")
    review_version: int = Field(description="Incremental review revision number.")


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
