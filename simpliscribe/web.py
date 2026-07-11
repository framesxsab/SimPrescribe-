import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from typing import Any

from fastapi import File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import settings
from .inference import structure_medications
from .ocr import extract_ocr_result, validate_document
from .reporting import build_pdf_report
from .storage import append_history, get_analysis_record, load_history


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
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "recent_analyses": load_history()[:5],
            "max_upload_mb": settings.max_upload_mb,
            "app_name": settings.app_name,
        },
    )


async def render_history(request: Request, templates) -> HTMLResponse:
    return templates.TemplateResponse(request, "history.html", {"analyses": load_history(), "app_name": settings.app_name})


async def render_details(request: Request, analysis_id: str, templates) -> HTMLResponse:
    analysis = get_analysis_record(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return templates.TemplateResponse(request, "details.html", {"analysis": analysis, "app_name": settings.app_name})


async def history_payload() -> dict[str, Any]:
    return {"analyses": load_history()}


async def download_report(analysis_id: str) -> Response:
    analysis = get_analysis_record(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    pdf_bytes = build_pdf_report(analysis, settings.app_name)
    safe_name = sanitize_filename(str(analysis.get("filename") or "analysis"))
    download_name = f"{Path(safe_name).stem}_report.pdf"
    encoded_name = quote(download_name)
    headers = {
        "Content-Disposition": f"attachment; filename=\"{download_name}\"; filename*=UTF-8''{encoded_name}",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    stored_file = await save_upload(file)
    try:
        ocr_result = extract_ocr_result(stored_file)
        parsed = structure_medications(ocr_result.text)
        medications = parsed.get("medications", [])
        if not isinstance(medications, list):
            raise ValueError("The extraction pipeline returned an invalid medication list.")
        pipeline = dict(parsed.get("pipeline") or {})
        pipeline["ocr_confidence"] = round(ocr_result.confidence, 4) if ocr_result.confidence is not None else None
        pipeline["ocr_warnings"] = list(ocr_result.warnings)
        pipeline["human_review_required"] = True
        analysis_id = str(uuid.uuid4())
        record = {
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
        }
        append_history(record)
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
