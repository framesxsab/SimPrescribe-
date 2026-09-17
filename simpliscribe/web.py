import asyncio
import hashlib
import logging
import re
import shutil
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from typing import Any

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .config import settings
from .inference import structure_medications
from .ocr import OCRResult, extract_ocr_result, validate_document
from .reporting import build_pdf_report
from .retrieval import get_vector_cache
from .schemas import analysis_output_contract
from .marketplace import pharmacy_can_access_analysis
from .security import current_user, owner_id, public_user_context, require_edit_role, require_role, verify_csrf
from .storage import (
    append_audit_event,
    get_analysis_record,
    get_prescription_file,
    load_history,
    save_prescription_file,
    try_append_history,
    update_analysis_record,
)


logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(filename: str) -> str:
    basename = Path(filename or "upload").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", basename)
    return cleaned or "upload"


def safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            path.unlink(missing_ok=True)
    except OSError:
        time.sleep(0.05)
        try:
            if path.exists():
                path.unlink(missing_ok=True)
        except OSError:
            logger.warning(f"Could not immediately delete temporary file {path}")


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
        safe_unlink(file_path)
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
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "analyses": load_history(owner),
            "app_name": settings.app_name,
            **public_user_context(request),
        },
    )


async def render_details(request: Request, analysis_id: str, templates) -> HTMLResponse:
    owner = owner_id(request)
    analysis = get_analysis_record(analysis_id, owner)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    user = current_user(request)
    # In dev mode (no auth), allow editing and patient-flow features
    can_edit = False
    show_patient_marketplace = False
    if user:
        can_edit = user.get("role") in {"admin", "reviewer"}
        show_patient_marketplace = user.get("role") == "patient"
    elif not settings.authentication_enabled:
        can_edit = True
        show_patient_marketplace = True
    breadcrumbs = [
        {"href": "/", "label": "Dashboard"},
        {"href": "/history", "label": "Prescriptions"},
        {"href": "", "label": analysis.get("filename", "Analysis")},
    ]
    return templates.TemplateResponse(
        request,
        "details.html",
        {
            "analysis": analysis,
            "app_name": settings.app_name,
            "alternatives_enabled": settings.alternatives_enabled,
            "can_edit": can_edit,
            "show_patient_marketplace": show_patient_marketplace,
            "breadcrumbs": breadcrumbs,
            **public_user_context(request),
        },
    )


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
    if settings.authentication_enabled:
        user = current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if user.get("role") not in {"patient", "admin", "reviewer"}:
            raise HTTPException(status_code=403, detail="Reviewer role required.")
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
            "patient_review_status": "needs_review",
            "patient_review_versions": [],
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
        settings.prescription_storage_dir.mkdir(parents=True, exist_ok=True)
        storage_name = f"{uuid.uuid4().hex}{stored_file.suffix.lower()}"
        protected_path = settings.prescription_storage_dir / storage_name
        try:
            shutil.copy2(stored_file, protected_path)
            save_prescription_file({
                "id": str(uuid.uuid4()),
                "analysis_id": analysis_id,
                "owner_id": owner,
                "storage_name": storage_name,
                "original_name": record["filename"],
                "content_type": file.content_type or "application/octet-stream",
                "sha256": hashlib.sha256(protected_path.read_bytes()).hexdigest(),
                "created_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.retention_days),
            })
        except Exception:
            safe_unlink(protected_path)
            pipeline = dict(record["pipeline"])
            pipeline["warnings"] = list(pipeline.get("warnings") or []) + [
                "Original prescription could not be retained; ordering is unavailable."
            ]
            pipeline["degraded"] = True
            pipeline["error_code"] = "PRESCRIPTION_STORAGE_FAILED"
            record["pipeline"] = pipeline
            update_analysis_record(analysis_id, owner, record)
        append_audit_event(
            str(uuid.uuid4()),
            owner,
            "analysis_created",
            analysis_id,
            provider=pipeline.get("used_provider", "unknown"),
        )
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
        safe_unlink(stored_file)


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
    append_audit_event(
        str(uuid.uuid4()),
        owner,
        "analysis_reviewed",
        analysis_id,
        status=status,
        review_version=len(review_versions),
    )
    return {
        "analysis_id": analysis_id,
        "review_status": status,
        "reviewed_at": analysis["reviewed_at"],
        "review_version": len(review_versions),
    }


async def patient_review_analysis(request: Request, analysis_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    user = current_user(request)
    if settings.authentication_enabled or user is not None:
        user = require_role(request, "patient")
        user_id = user["id"]
    else:
        user_id = "local"
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    analysis = get_analysis_record(analysis_id, user_id)
    if analysis is None:
        analysis = get_analysis_record(analysis_id, "local")
    if analysis is None:
        raise HTTPException(status_code=404, detail="Prescription analysis not found.")
    medications = payload.get("medications")
    status = str(payload.get("status") or "")
    if status not in {"confirmed", "corrected"} or not isinstance(medications, list) or len(medications) != len(analysis.get("medications") or []):
        raise HTTPException(status_code=400, detail="Review every extracted medicine before confirmation.")
    allowed = {"name", "type", "dosage", "frequency", "duration"}
    cleaned = []
    for medication in medications:
        if not isinstance(medication, dict) or not str(medication.get("name") or "").strip():
            raise HTTPException(status_code=400, detail="Every medicine requires a name.")
        current = dict(analysis["medications"][len(cleaned)])
        for field in allowed:
            if field in medication:
                current[field] = str(medication[field]).strip()[:500]
        cleaned.append(current)
    analysis.setdefault("original_medications", deepcopy(analysis.get("medications") or []))
    versions = list(analysis.get("patient_review_versions") or [])[:20]
    versions.append({
        "recorded_at": utc_now_iso(),
        "status": analysis.get("patient_review_status", "needs_review"),
        "medications": deepcopy(analysis.get("medications") or []),
    })
    analysis["medications"] = cleaned
    analysis["patient_review_status"] = status
    analysis["patient_confirmed_at"] = utc_now_iso()
    analysis["patient_review_versions"] = versions
    if not update_analysis_record(analysis_id, user_id, analysis):
        raise HTTPException(status_code=409, detail="Prescription changed. Reload and try again.")
    append_audit_event(str(uuid.uuid4()), user_id, "patient_prescription_confirmed", analysis_id, status=status)
    return {"analysis_id": analysis_id, "patient_review_status": status, "version": len(versions)}


async def prescription_source(request: Request, analysis_id: str) -> FileResponse:
    user = current_user(request)
    if user is None:
        if not settings.authentication_enabled:
            user = {"id": "local", "role": "reviewer"}
        else:
            raise HTTPException(status_code=401, detail="Authentication required.")
    source = get_prescription_file(analysis_id)
    if not source:
        raise HTTPException(status_code=404, detail="Prescription source not found or expired.")
    allowed = user["role"] in {"admin", "reviewer"} or source["owner_id"] == user["id"]
    if user["role"] == "pharmacy":
        allowed = pharmacy_can_access_analysis(user["id"], analysis_id)
    if not allowed:
        raise HTTPException(status_code=403, detail="Prescription source access denied.")
    path = (settings.prescription_storage_dir / source["storage_name"]).resolve()
    if path.parent != settings.prescription_storage_dir.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Prescription source not found or expired.")
    append_audit_event(str(uuid.uuid4()), user["id"], "prescription_source_viewed", analysis_id)
    return FileResponse(
        path,
        media_type=source["content_type"],
        filename=source["original_name"],
        headers={"Cache-Control": "no-store"},
    )


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
