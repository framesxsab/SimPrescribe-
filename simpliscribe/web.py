import asyncio
import logging
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from typing import Any

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import settings
from .inference import structure_medications
from .ocr import OCRResult, extract_ocr_result, validate_document
from .reporting import build_pdf_report
from .retrieval import get_vector_cache
from .schemas import analysis_output_contract
from .security import owner_id, public_user_context, require_edit_role, verify_csrf
from .storage import append_audit_event, get_analysis_record, load_history, try_append_history, update_analysis_record


logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(filename: str) -> str:
    basename = Path(filename or "upload").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", basename)
    return cleaned or "upload"


async def save_upload(file: UploadFile) -> Path:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a name.")

    safe_name = sanitize_filename(file.filename)
    extension = Path(safe_name).suffix.lower()
    allowed_extensions = {".png", ".jpg", ".jpeg", ".pdf", ".webp"}
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(status_code=400, detail=f"Uploaded file exceeds the {settings.max_upload_mb} MB limit.")

    stored_name = f"{uuid.uuid4()}_{safe_name}"
    file_path = settings.uploads_dir / stored_name
    file_path.write_bytes(contents)
    try:
        validate_document(file_path)
    except ValueError as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return file_path


async def render_dashboard(request: Request, templates) -> HTMLResponse:
    owner = owner_id(request)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "recent_analyses": load_history(owner)[:5],
            "max_upload_mb": settings.max_upload_mb,
            "app_name": settings.app_name,
            "alternatives_enabled": settings.alternatives_enabled,
            **public_user_context(request),
        },
    )


async def render_history(request: Request, templates) -> HTMLResponse:
    owner = owner_id(request)
    return templates.TemplateResponse(request, "history.html", {"analyses": load_history(owner), "app_name": settings.app_name, **public_user_context(request)})


async def render_details(request: Request, analysis_id: str, templates) -> HTMLResponse:
    analysis = get_analysis_record(analysis_id, owner_id(request))
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return templates.TemplateResponse(request, "details.html", {
        "analysis": analysis,
        "app_name": settings.app_name,
        "alternatives_enabled": settings.alternatives_enabled,
        **public_user_context(request),
    })


async def history_payload(request: Request) -> dict[str, Any]:
    return {"analyses": load_history(owner_id(request))}


async def download_report(request: Request, analysis_id: str) -> Response:
    owner = owner_id(request)
    analysis = get_analysis_record(analysis_id, owner)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    try:
        pdf_bytes = build_pdf_report(analysis, settings.app_name)
    except Exception:
        logger.exception("PDF report generation failed.")
        return JSONResponse(
            status_code=503,
            content={
                "error": "The PDF report is temporarily unavailable. Review the on-screen analysis against the original prescription.",
                "error_code": "REPORT_UNAVAILABLE",
                "analysis_id": analysis_id,
            },
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
    append_audit_event(str(uuid.uuid4()), owner, "report_downloaded", analysis_id)
    safe_name = sanitize_filename(str(analysis.get("filename") or "analysis"))
    download_name = f"{Path(safe_name).stem}_report.pdf"
    encoded_name = quote(download_name)
    headers = {
        "Content-Disposition": f"attachment; filename=\"{download_name}\"; filename*=UTF-8''{encoded_name}",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


async def analyze(
    request: Request,
    file: UploadFile = File(...),
    consent: bool = Form(False),
    csrf: str | None = Form(None),
) -> JSONResponse:
    require_edit_role(request)
    owner = owner_id(request)
    verify_csrf(request, request.headers.get("X-CSRF-Token") or csrf)
    if not consent:
        raise HTTPException(status_code=400, detail="Explicit processing consent is required.")
    stored_file = await save_upload(file)
    try:
        try:
            ocr_result = await asyncio.to_thread(extract_ocr_result, stored_file)
        except Exception:
            logger.exception("OCR engine failed.")
            ocr_result = OCRResult(
                "",
                None,
                (),
                ("OCR engine failed; the original prescription must be reviewed.",),
            )
        if not str(ocr_result.text or "").strip():
            raise ValueError("No readable text was extracted from the uploaded document.")
        cache_hit = get_vector_cache().lookup(ocr_result.text, threshold=0.98)
        if cache_hit is not None:
            cached_payload, similarity = cache_hit
            parsed = deepcopy(cached_payload)
            pipeline = dict(parsed.get("pipeline") or {})
            pipeline["cached_vector_match"] = True
            pipeline["cached_vector_similarity"] = round(similarity, 4)
            parsed["pipeline"] = pipeline
        else:
            try:
                parsed = await asyncio.to_thread(structure_medications, ocr_result.text)
                if not parsed.get("pipeline", {}).get("degraded", False):
                    get_vector_cache().store(ocr_result.text, parsed)
            except ValueError:
                raise
            except Exception:
                logger.exception("Medication structuring failed; returning an empty review payload.")
                parsed = {
                    "patient_name": "N/A",
                    "doctor_name": "N/A",
                    "date": "N/A",
                    "medications": [],
                    "pipeline": {
                        "requested_provider": settings.inference_provider,
                        "used_provider": "fallback",
                        "warnings": ["Medication structuring failed; every field requires manual review."],
                        "human_review_required": True,
                        "degraded": True,
                        "error_code": "STRUCTURING_FAILED",
                    },
                }
        medications = parsed.get("medications", [])
        if not isinstance(medications, list):
            raise ValueError("The extraction pipeline returned an invalid medication list.")
        pipeline = dict(parsed.get("pipeline") or {})
        pipeline["ocr_confidence"] = round(ocr_result.confidence, 4) if ocr_result.confidence is not None else None
        pipeline["ocr_warnings"] = list(ocr_result.warnings)
        pipeline["human_review_required"] = True
        analysis_id = str(uuid.uuid4())
        record = analysis_output_contract({
            "id": analysis_id,
            "filename": stored_file.name.split("_", 1)[1] if "_" in stored_file.name else stored_file.name,
            "created_at": utc_now_iso(),
            "raw_text": ocr_result.text,
            "patient_name": parsed.get("patient_name", "N/A"),
            "doctor_name": parsed.get("doctor_name", "N/A"),
            "date": parsed.get("date", "N/A"),
            "medications": medications,
            "pipeline": pipeline,
            "review_status": "needs_review",
        })
        stored = try_append_history(record, owner_id=owner)
        if not stored:
            pipeline = dict(record["pipeline"])
            pipeline["warnings"] = list(pipeline.get("warnings") or []) + [
                "Analysis could not be saved; this result is not in history."
            ]
            pipeline["degraded"] = True
            pipeline["error_code"] = "STORAGE_FAILED"
            record["pipeline"] = pipeline
            logger.warning("Analysis persistence failed after retry.")
            return JSONResponse(
                status_code=503,
                content={"analysis_id": analysis_id, **record},
                headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
            )
        append_audit_event(str(uuid.uuid4()), owner, "analysis_created", analysis_id, provider=pipeline.get("used_provider", "unknown"))
        return JSONResponse(
            content={"analysis_id": analysis_id, **record},
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
    except HTTPException:
        raise
    except ValueError:
        logger.exception("Prescription analysis rejected because the result was not usable.")
        return JSONResponse(
            status_code=422,
            content={
                "error": "No reliable prescription text could be extracted. Try a clearer scan and review the original prescription.",
                "error_code": "UNUSABLE_PRESCRIPTION",
                "medications": [],
            },
            headers={"Cache-Control": "no-store"},
        )
    except Exception:
        logger.exception("Prescription analysis failed.")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prescription analysis is temporarily unavailable. Please retry without relying on a partial result.",
                "error_code": "ANALYSIS_FAILED",
                "medications": [],
            },
            headers={"Cache-Control": "no-store"},
        )
    finally:
        if stored_file.exists():
            stored_file.unlink()


async def review_analysis(request: Request, analysis_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    require_edit_role(request)
    owner = owner_id(request)
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    analysis = get_analysis_record(analysis_id, owner)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    previous_analysis = deepcopy(analysis)
    status = str(payload.get("status") or "").strip()
    if status not in {"confirmed", "corrected", "rejected"}:
        raise HTTPException(status_code=400, detail="Invalid review status.")
    review_versions = list(analysis.get("review_versions") or [])[:20]
    review_versions.append({
        "version": len(review_versions) + 1,
        "recorded_at": utc_now_iso(),
        "status": str(analysis.get("review_status") or "needs_review"),
        "reviewed_at": analysis.get("reviewed_at"),
        "reviewed_by": analysis.get("reviewed_by"),
        "medications": deepcopy(analysis.get("medications") or []),
    })
    medications = payload.get("medications")
    if medications is not None:
        if not isinstance(medications, list) or len(medications) > 50:
            raise HTTPException(status_code=400, detail="Invalid medications payload.")
        allowed = {"name", "type", "dosage", "frequency", "duration"}
        for index, correction in enumerate(medications):
            if not isinstance(correction, dict) or index >= len(analysis.get("medications", [])):
                raise HTTPException(status_code=400, detail="Invalid medication correction.")
            for field in allowed:
                if field in correction:
                    analysis["medications"][index][field] = str(correction[field]).strip()[:500]
    analysis["review_status"] = status
    analysis["reviewed_at"] = utc_now_iso()
    analysis["reviewed_by"] = owner
    analysis["review_versions"] = review_versions
    if not update_analysis_record(analysis_id, owner, analysis, expected_record=previous_analysis):
        raise HTTPException(status_code=409, detail="Analysis was updated by another reviewer. Reload and try again.")
    append_audit_event(str(uuid.uuid4()), owner, "analysis_reviewed", analysis_id, status=status, review_version=len(review_versions))
    return {"analysis_id": analysis_id, "review_status": status, "reviewed_at": analysis["reviewed_at"], "review_version": len(review_versions)}


def export_audit_csv(events: list[dict[str, Any]]) -> str:
    import csv
    import io
    import json

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "created_at", "event_type", "analysis_id", "metadata"])
    for event in events:
        writer.writerow([
            event.get("id", ""),
            event.get("created_at", ""),
            event.get("event_type", ""),
            event.get("analysis_id", "") or "",
            json.dumps(event.get("metadata", {})),
        ])
    return output.getvalue()

