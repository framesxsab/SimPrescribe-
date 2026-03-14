from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .storage import load_history
from .web import analyze, download_report, history_payload, render_dashboard, render_details, render_history

load_dotenv()
load_history()

# Pre-warm heavy resources at startup to avoid slow first request
import threading
def _preload():
    try:
        from .inference import load_medicine_lexicon
        load_medicine_lexicon()
    except Exception:
        pass
    try:
        from .ocr import get_ocr_reader
        get_ocr_reader()
    except Exception:
        pass

threading.Thread(target=_preload, daemon=True).start()

app = FastAPI(title=f"{settings.app_name} API")
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
templates = Jinja2Templates(directory=str(settings.templates_dir))


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request) -> HTMLResponse:
    return await render_dashboard(request, templates)


@app.get("/history", response_class=HTMLResponse)
async def serve_history(request: Request) -> HTMLResponse:
    return await render_history(request, templates)


@app.get("/details/{analysis_id}", response_class=HTMLResponse)
async def serve_details(request: Request, analysis_id: str) -> HTMLResponse:
    return await render_details(request, analysis_id, templates)


@app.get("/api/history")
async def get_history() -> dict:
    return await history_payload()


@app.post("/api/analyze")
async def analyze_prescription(file: UploadFile):
    return await analyze(file)


@app.get("/api/report/{analysis_id}")
async def get_report(analysis_id: str):
    return await download_report(analysis_id)