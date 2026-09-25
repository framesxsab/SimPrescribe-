import asyncio
import hashlib
import logging
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from typing import Any

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .config import settings
from .inference import hydrate_medication_references, structure_medications
from .ocr import extract_ocr_result, get_ocr_state, ocr_analysis_id, validate_document
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


@contextmanager
def timed_stage(analysis_id: str, stage: str, provider: str = "local"):
    started = time.monotonic()
    try:
        yield
    except Exception:
        logger.warning("analysis_id=%s stage=%s duration_ms=%d provider=%s status=error error_code=STAGE_FAILED", analysis_id, stage, round((time.monotonic() - started) * 1000), provider)
        raise
    else:
        logger.info("analysis_id=%s stage=%s duration_ms=%d provider=%s status=ok", analysis_id, stage, round((time.monotonic() - started) * 1000), provider)


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
            logger.warning("Could not immediately delete temporary processing file")


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
    try:
        file_path.write_bytes(contents)
        validate_document(file_path)
    except ValueError as exc:
        safe_unlink(file_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        safe_unlink(file_path)
        raise
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
    if analysis.get("prescription_state") in {"uploaded", "processing"}:
        started = analysis.get("processing_started_at") or analysis.get("created_at")
        try:
            stale = bool(started and datetime.now(timezone.utc) - datetime.fromisoformat(started) > timedelta(minutes=5))
        except ValueError:
            stale = False
        if stale:
            previous = deepcopy(analysis)
            analysis["prescription_state"] = "processing_failed"
            analysis["error_code"] = "PROCESSING_INTERRUPTED"
            if not update_analysis_record(analysis_id, owner, analysis, expected_record=previous):
                analysis = get_analysis_record(analysis_id, owner) or analysis
    if analysis.get("prescription_state") in {"uploaded", "processing", "processing_failed"}:
        with timed_stage(analysis_id, "template_render"):
            return templates.TemplateResponse(request, "processing.html", {
                "analysis": analysis, "app_name": settings.app_name, **public_user_context(request),
            })
    user = current_user(request)
    if analysis.get("prescription_state") == "review_required" and (user and user.get("role") == "patient" or not user and not settings.authentication_enabled):
        with timed_stage(analysis_id, "template_render"):
            return templates.TemplateResponse(request, "patient_review.html", {
                "analysis": analysis, "app_name": settings.app_name, **public_user_context(request),
            })
    # In dev mode (no auth), allow editing and patient-flow features
    can_edit = False
    show_patient_marketplace = False
    if user:
        can_edit = user.get("role") in {"admin", "reviewer"}
        show_patient_marketplace = user.get("role") == "patient"
    elif not settings.authentication_enabled:
        can_edit = True
        show_patient_marketplace = True
    analysis["medications"] = await asyncio.to_thread(
        hydrate_medication_references, analysis.get("medications") or []
    )
    breadcrumbs = [
        {"href": "/", "label": "Dashboard"},
        {"href": "/history", "label": "Prescriptions"},
        {"href": "", "label": analysis.get("filename", "Analysis")},
    ]
    with timed_stage(analysis_id, "template_render"):
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
    if analysis.get("prescription_state") in {"uploaded", "processing", "processing_failed"}:
        return JSONResponse(status_code=503, content={"error_code": "REPORT_UNAVAILABLE", "analysis_id": analysis_id, "error": "The prescription is not ready for a report."})

    try:
        analysis["medications"] = await asyncio.to_thread(
            hydrate_medication_references, analysis.get("medications") or []
        )
        with timed_stage(analysis_id, "report_generation"):
            pdf_bytes = build_pdf_report(analysis, settings.app_name)
    except Exception:
        logger.warning("analysis_id=%s stage=report_generation status=error error_code=REPORT_UNAVAILABLE", analysis_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": "The PDF report is temporarily unavailable. Review the on-screen analysis against the original prescription.",
                "error_code": "REPORT_UNAVAILABLE",
                "analysis_id": analysis_id,
            },
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
    try:
        append_audit_event(str(uuid.uuid4()), owner, "report_downloaded", analysis_id)
    except Exception:
        logger.warning("analysis_id=%s stage=report_audit status=error error_code=AUDIT_FAILED", analysis_id)
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
    analysis_id = str(uuid.uuid4())
    try:
        with timed_stage(analysis_id, "upload_validation"):
            stored_file = await save_upload(file)
    except OSError:
        return JSONResponse(status_code=503, content={"error_code": "STORAGE_FAILED", "error": "Upload storage is unavailable. Please retry."})
    record = analysis_output_contract({
        "id": analysis_id,
        "filename": stored_file.name.split("_", 1)[1] if "_" in stored_file.name else stored_file.name,
        "created_at": utc_now_iso(),
        "prescription_state": "uploaded",
        "processing_stage": "upload_saved",
        "patient_review_status": "needs_review",
        "patient_review_versions": [],
    })
    try:
        with timed_stage(analysis_id, "initial_database_write"):
            if not try_append_history(record, owner_id=owner):
                record["pipeline"] = {"error_code": "STORAGE_FAILED", "human_review_required": True, "degraded": True}
                return JSONResponse(status_code=503, content={"analysis_id": analysis_id, "persisted": False, "error_code": "STORAGE_FAILED", "error": "Prescription could not be saved. Please retry.", **record})
        record["processing_stage"] = "private_storage"
        settings.prescription_storage_dir.mkdir(parents=True, exist_ok=True)
        storage_name = f"{uuid.uuid4().hex}{stored_file.suffix.lower()}"
        protected_path = settings.prescription_storage_dir / storage_name
        with timed_stage(analysis_id, "private_source_storage"):
            try:
                shutil.copy2(stored_file, protected_path)
                save_prescription_file({
                    "id": str(uuid.uuid4()), "analysis_id": analysis_id, "owner_id": owner,
                    "storage_name": storage_name, "original_name": record["filename"],
                    "content_type": file.content_type or "application/octet-stream",
                    "sha256": hashlib.sha256(protected_path.read_bytes()).hexdigest(),
                    "created_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.retention_days),
                })
            except Exception:
                safe_unlink(protected_path)
                raise
        record["prescription_state"] = "processing"
        record["processing_stage"] = "ocr_initializing" if not get_ocr_state()["ready"] else "ocr"
        record["processing_started_at"] = utc_now_iso()
        if not update_analysis_record(analysis_id, owner, record):
            raise RuntimeError("Could not persist processing state")
        try:
            with timed_stage(analysis_id, "ocr", "paddle"):
                token = ocr_analysis_id.set(analysis_id)
                try:
                    ocr_result = await asyncio.to_thread(extract_ocr_result, stored_file)
                finally:
                    ocr_analysis_id.reset(token)
        except Exception:
            raise ValueError("OCR engine failed")
        if not str(ocr_result.text or "").strip():
            raise ValueError("No readable text was extracted from the uploaded document.")
        record["raw_text"] = ocr_result.text
        record["ocr_confidence"] = ocr_result.confidence
        record["ocr_lines"] = [{"text": line.text, "confidence": line.confidence} for line in ocr_result.lines]
        record["processing_stage"] = "structuring"
        with timed_stage(analysis_id, "ocr_persistence"):
            if not update_analysis_record(analysis_id, owner, record):
                raise RuntimeError("Could not persist OCR output")
        try:
            with timed_stage(analysis_id, "cache_lookup"):
                cache_hit = get_vector_cache().lookup(ocr_result.text, threshold=0.98)
        except Exception:
            cache_hit = None
        if cache_hit is not None:
            cached_payload, similarity = cache_hit
            parsed = deepcopy(cached_payload)
            pipeline = dict(parsed.get("pipeline") or {})
            pipeline["cached_vector_match"] = True
            pipeline["cached_vector_similarity"] = round(similarity, 4)
            parsed["pipeline"] = pipeline
        else:
            try:
                with timed_stage(analysis_id, "structuring", settings.inference_provider):
                    parsed = await asyncio.to_thread(structure_medications, ocr_result.text)
                if not parsed.get("pipeline", {}).get("degraded", False):
                    try:
                        get_vector_cache().store(ocr_result.text, parsed)
                    except Exception:
                        logger.warning("analysis_id=%s stage=cache_store status=error error_code=CACHE_FAILED", analysis_id)
            except ValueError:
                raise RuntimeError("Medication structuring rejected the input")
            except Exception:
                raise RuntimeError("Medication structuring failed")
        medications = parsed.get("medications", [])
        if not isinstance(medications, list) or not medications:
            raise RuntimeError("The extraction pipeline returned no medicines.")
        pipeline = dict(parsed.get("pipeline") or {})
        pipeline["ocr_confidence"] = round(ocr_result.confidence, 4) if ocr_result.confidence is not None else None
        pipeline["ocr_warnings"] = list(ocr_result.warnings)
        pipeline["human_review_required"] = True
        record.update({
            "patient_name": parsed.get("patient_name", "N/A"),
            "doctor_name": parsed.get("doctor_name", "N/A"),
            "date": parsed.get("date", "N/A"),
            "medications": medications,
            "original_medications": deepcopy(medications),
            "pipeline": pipeline,
            "prescription_state": "review_required",
            "processing_stage": "review_ready",
        })
        with timed_stage(analysis_id, "final_database_write"):
            if not update_analysis_record(analysis_id, owner, record):
                raise RuntimeError("Could not save structured result")
        try:
            append_audit_event(
                str(uuid.uuid4()), owner, "analysis_created", analysis_id,
                provider=pipeline.get("used_provider", "unknown"),
            )
        except Exception:
            logger.warning("analysis_id=%s stage=audit status=error error_code=AUDIT_FAILED", analysis_id)
        return JSONResponse(
            content={"analysis_id": analysis_id, **record},
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
    except asyncio.CancelledError:
        record["prescription_state"] = "processing_failed"
        record["error_code"] = "PROCESSING_INTERRUPTED"
        try:
            update_analysis_record(analysis_id, owner, record)
        except Exception:
            pass
        raise
    except HTTPException:
        raise
    except ValueError:
        logger.warning("analysis_id=%s stage=ocr status=error error_code=UNUSABLE_PRESCRIPTION", analysis_id)
        record["prescription_state"] = "processing_failed"
        record["error_code"] = "UNUSABLE_PRESCRIPTION"
        try:
            if not update_analysis_record(analysis_id, owner, record):
                raise RuntimeError("Could not persist failure state")
        except Exception:
            return JSONResponse(status_code=503, content={"error_code": "STORAGE_FAILED", "error": "Prescription state could not be saved."})
        return JSONResponse(
            status_code=422,
            content={
                "error": "No reliable prescription text could be extracted. Try a clearer scan and review the original prescription.",
                "error_code": "UNUSABLE_PRESCRIPTION",
                "analysis_id": analysis_id,
                "medications": [],
            },
            headers={"Cache-Control": "no-store"},
        )
    except Exception:
        logger.warning("analysis_id=%s stage=%s status=error error_code=ANALYSIS_FAILED", analysis_id, record.get("processing_stage"))
        record["prescription_state"] = "processing_failed"
        record["error_code"] = "STORAGE_FAILED" if record.get("processing_stage") in {"upload_saved", "private_storage", "review_ready"} else "STRUCTURING_FAILED" if record.get("raw_text") else "UNUSABLE_PRESCRIPTION"
        try:
            update_analysis_record(analysis_id, owner, record)
        except Exception:
            pass
        return JSONResponse(
            status_code=503,
            content={
                "error": "Prescription processing failed. The saved source and OCR can be retried from history.",
                "error_code": record["error_code"],
                "analysis_id": analysis_id,
                "medications": [],
            },
            headers={"Cache-Control": "no-store"},
        )
    finally:
        safe_unlink(stored_file)


async def start_analysis(request: Request, file: UploadFile, consent: bool, csrf: str | None) -> JSONResponse:
    if settings.authentication_enabled:
        user = current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if user.get("role") not in {"patient", "admin", "reviewer"}:
            raise HTTPException(status_code=403, detail="Patient or reviewer account required.")
    owner = owner_id(request)
    verify_csrf(request, request.headers.get("X-CSRF-Token") or csrf)
    if not consent:
        raise HTTPException(status_code=400, detail="Explicit processing consent is required.")
    analysis_id = str(uuid.uuid4())
    try:
        with timed_stage(analysis_id, "upload_validation"):
            stored_file = await save_upload(file)
    except OSError:
        return JSONResponse(status_code=503, content={"error_code": "STORAGE_FAILED", "error": "Upload storage is unavailable. Please retry."})
    record = analysis_output_contract({
        "id": analysis_id,
        "filename": stored_file.name.split("_", 1)[1] if "_" in stored_file.name else stored_file.name,
        "created_at": utc_now_iso(),
        "prescription_state": "uploaded",
        "processing_stage": "upload_saved",
        "patient_review_status": "needs_review",
        "patient_review_versions": [],
    })
    try:
        with timed_stage(analysis_id, "initial_database_write"):
            if not try_append_history(record, owner_id=owner):
                return JSONResponse(status_code=503, content={"error_code": "STORAGE_FAILED", "error": "Prescription could not be saved. Please retry."})
        settings.prescription_storage_dir.mkdir(parents=True, exist_ok=True)
        storage_name = f"{uuid.uuid4().hex}{stored_file.suffix.lower()}"
        protected_path = settings.prescription_storage_dir / storage_name
        with timed_stage(analysis_id, "private_source_storage"):
            try:
                shutil.copy2(stored_file, protected_path)
                save_prescription_file({
                    "id": str(uuid.uuid4()), "analysis_id": analysis_id, "owner_id": owner,
                    "storage_name": storage_name, "original_name": record["filename"],
                    "content_type": file.content_type or "application/octet-stream",
                    "sha256": hashlib.sha256(protected_path.read_bytes()).hexdigest(),
                    "created_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc) + timedelta(days=settings.retention_days),
                })
            except Exception:
                safe_unlink(protected_path)
                raise
        return JSONResponse(status_code=201, content={"analysis_id": analysis_id, "prescription_state": "uploaded"})
    except Exception:
        logger.warning("analysis_id=%s stage=private_source_storage status=error error_code=STORAGE_FAILED", analysis_id)
        record["prescription_state"] = "processing_failed"
        record["error_code"] = "STORAGE_FAILED"
        try:
            update_analysis_record(analysis_id, owner, record)
        except Exception:
            pass
        return JSONResponse(status_code=503, content={"error_code": "STORAGE_FAILED", "error": "Prescription source could not be stored."})
    finally:
        safe_unlink(stored_file)


async def retry_processing(request: Request, analysis_id: str, *, allow_uploaded: bool = False) -> JSONResponse:
    owner = owner_id(request)
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    record = get_analysis_record(analysis_id, owner)
    if record is None:
        raise HTTPException(status_code=404, detail="Prescription analysis not found.")
    allowed = {"uploaded"} if allow_uploaded else {"processing_failed"}
    if record.get("prescription_state") not in allowed:
        raise HTTPException(status_code=409, detail="Prescription processing has already started or is not ready for retry.")
    original = deepcopy(record)
    record["prescription_state"] = "processing"
    record["processing_stage"] = "structuring" if record.get("raw_text") else ("ocr_initializing" if not get_ocr_state()["ready"] else "ocr")
    record["processing_started_at"] = utc_now_iso()
    if not update_analysis_record(analysis_id, owner, record, expected_record=original):
        raise HTTPException(status_code=409, detail="Prescription changed. Reload and try again.")
    try:
        if not record.get("raw_text"):
            source = get_prescription_file(analysis_id)
            if not source or source["owner_id"] != owner:
                raise FileNotFoundError("Protected source is unavailable")
            path = (settings.prescription_storage_dir / source["storage_name"]).resolve()
            if path.parent != settings.prescription_storage_dir.resolve() or not path.is_file():
                raise FileNotFoundError("Protected source is unavailable")
            with timed_stage(analysis_id, "ocr_retry", "paddle"):
                token = ocr_analysis_id.set(analysis_id)
                try:
                    ocr_result = await asyncio.to_thread(extract_ocr_result, path)
                finally:
                    ocr_analysis_id.reset(token)
            if not ocr_result.text.strip():
                raise ValueError("Unreadable prescription")
            record["raw_text"] = ocr_result.text
            record["ocr_confidence"] = ocr_result.confidence
            record["ocr_lines"] = [{"text": line.text, "confidence": line.confidence} for line in ocr_result.lines]
            record["processing_stage"] = "structuring"
            if not update_analysis_record(analysis_id, owner, record):
                raise RuntimeError("Could not save OCR output")
        with timed_stage(analysis_id, "structuring_retry", settings.inference_provider):
            parsed = await asyncio.to_thread(structure_medications, record["raw_text"])
        medications = parsed.get("medications")
        if not isinstance(medications, list) or not medications:
            raise RuntimeError("No structured medicines")
        record.update({
            "patient_name": parsed.get("patient_name", "N/A"),
            "doctor_name": parsed.get("doctor_name", "N/A"),
            "date": parsed.get("date", "N/A"),
            "medications": medications,
            "original_medications": deepcopy(medications),
            "pipeline": {**(parsed.get("pipeline") or {}), "ocr_confidence": record.get("ocr_confidence"), "human_review_required": True},
            "prescription_state": "review_required",
            "processing_stage": "review_ready",
        })
        record.pop("error_code", None)
        if not update_analysis_record(analysis_id, owner, record):
            raise RuntimeError("Could not save structured result")
        return JSONResponse(content={"analysis_id": analysis_id, "prescription_state": "review_required"})
    except asyncio.CancelledError:
        record["prescription_state"] = "processing_failed"
        record["error_code"] = "PROCESSING_INTERRUPTED"
        try:
            update_analysis_record(analysis_id, owner, record)
        except Exception:
            pass
        raise
    except FileNotFoundError:
        record["prescription_state"] = "processing_failed"
        record["error_code"] = "STORAGE_FAILED"
        update_analysis_record(analysis_id, owner, record)
        return JSONResponse(status_code=503, content={"analysis_id": analysis_id, "error_code": "STORAGE_FAILED"})
    except ValueError:
        record["prescription_state"] = "processing_failed"
        record["error_code"] = "UNUSABLE_PRESCRIPTION"
        update_analysis_record(analysis_id, owner, record)
        return JSONResponse(status_code=422, content={"analysis_id": analysis_id, "error_code": "UNUSABLE_PRESCRIPTION"})
    except Exception:
        logger.warning("analysis_id=%s stage=%s status=error error_code=PROCESSING_FAILED", analysis_id, record["processing_stage"])
        record["prescription_state"] = "processing_failed"
        record["error_code"] = "STRUCTURING_FAILED" if record.get("raw_text") else "UNUSABLE_PRESCRIPTION"
        try:
            update_analysis_record(analysis_id, owner, record)
        except Exception:
            pass
        return JSONResponse(status_code=503, content={"analysis_id": analysis_id, "error_code": record["error_code"]})


async def save_patient_draft(request: Request, analysis_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    user = current_user(request)
    if settings.authentication_enabled or user is not None:
        owner = require_role(request, "patient")["id"]
    else:
        owner = "local"
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    analysis = get_analysis_record(analysis_id, owner)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Prescription analysis not found.")
    if analysis.get("prescription_state") != "review_required":
        raise HTTPException(status_code=409, detail="Prescription is not open for edits.")
    incoming = payload.get("medications")
    originals = analysis.get("original_medications") or analysis.get("medications") or []
    if not isinstance(incoming, list) or not originals or len(incoming) != len(originals):
        raise HTTPException(status_code=400, detail="Review every medicine before saving edits.")
    cleaned = []
    fields = {"name", "type", "dosage", "frequency", "duration"}
    for medication in incoming:
        if not isinstance(medication, dict) or not str(medication.get("name") or "").strip():
            raise HTTPException(status_code=400, detail="Every medicine requires a name.")
        cleaned.append({field: str(medication.get(field) or "").strip()[:500] for field in fields})
    previous = deepcopy(analysis)
    versions = list(analysis.get("patient_draft_versions") or [])
    versions.append({"recorded_at": utc_now_iso(), "medications": deepcopy(analysis.get("patient_draft_medications") or originals)})
    analysis["patient_draft_medications"] = cleaned
    analysis["patient_draft_versions"] = versions
    if not update_analysis_record(analysis_id, owner, analysis, expected_record=previous):
        raise HTTPException(status_code=409, detail="Prescription changed. Reload and try again.")
    return {"analysis_id": analysis_id, "version": len(versions)}


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
        raise HTTPException(status_code=404, detail="Prescription analysis not found.")
    if analysis.get("prescription_state", "review_required") != "review_required" or analysis.get("patient_review_status") in {"confirmed", "corrected"}:
        raise HTTPException(status_code=409, detail="Prescription is already confirmed or is not ready for review.")
    previous_analysis = deepcopy(analysis)
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
    versions = list(analysis.get("patient_review_versions") or [])
    versions.append({
        "recorded_at": utc_now_iso(),
        "status": analysis.get("patient_review_status", "needs_review"),
        "medications": deepcopy(analysis.get("medications") or []),
    })
    analysis["medications"] = cleaned
    analysis["patient_review_status"] = status
    analysis["patient_confirmed_at"] = utc_now_iso()
    analysis["patient_review_versions"] = versions
    analysis["prescription_state"] = "confirmed"
    if not update_analysis_record(analysis_id, user_id, analysis, expected_record=previous_analysis):
        raise HTTPException(status_code=409, detail="Prescription changed. Reload and try again.")
    try:
        append_audit_event(str(uuid.uuid4()), user_id, "patient_prescription_confirmed", analysis_id, status=status)
    except Exception:
        logger.warning("analysis_id=%s stage=confirmation_audit status=error error_code=AUDIT_FAILED", analysis_id)
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
