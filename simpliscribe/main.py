from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .storage import load_history
from .web import analyze, download_report, history_payload, render_dashboard, render_details, render_history

load_history()

app = FastAPI(title=f"{settings.app_name} API")
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
templates = Jinja2Templates(directory=str(settings.templates_dir))


@app.middleware("http")
async def protect_health_data_responses(request: Request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


@app.get("/api/health")
async def health() -> dict:
    datasets_ready = settings.india_medicine_dataset.exists() and settings.medicine_database_dataset.exists()
    provider_ready = settings.inference_provider == "fallback" or bool(
        settings.hf_token if settings.inference_provider == "huggingface" else settings.model_api_url
    )
    return {
        "status": "ready" if datasets_ready and provider_ready else "degraded",
        "datasets_ready": datasets_ready,
        "configured_provider": settings.inference_provider,
        "provider_ready": provider_ready,
        "clinical_use": "human_review_required",
    }


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
