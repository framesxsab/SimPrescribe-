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
from .ocr import extract_ocr_text
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
        raw_text = extract_ocr_text(stored_file)
        medications = structure_medications(raw_text)
        analysis_id = str(uuid.uuid4())
        record = {
            "id": analysis_id,
            "filename": stored_file.name.split("_", 1)[1] if "_" in stored_file.name else stored_file.name,
            "created_at": utc_now_iso(),
            "raw_text": raw_text,
            "medications": medications,
        }
        append_history(record)
        return JSONResponse(content={"analysis_id": analysis_id, **record})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prescription analysis failed.")
        message = str(exc)
        if "PDX has already been initialized" in message:
            message = "OCR engine is warming up. Please retry in a few seconds."
        return JSONResponse(status_code=500, content={"error": message, "medications": []})
    finally:
        if stored_file.exists():
            stored_file.unlink()