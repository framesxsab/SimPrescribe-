from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .storage import load_history
from .web import analyze, history_payload, render_dashboard, render_details, render_history

load_dotenv()
load_history()

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