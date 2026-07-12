import asyncio
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .config import settings
from .security import authenticate, current_user, csrf_token, verify_csrf
from .storage import append_audit_event, load_history
from .web import analyze, download_report, history_payload, render_dashboard, render_details, render_history, review_analysis

settings.validate_runtime()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
load_history()

app = FastAPI(title=f"{settings.app_name} API")
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
templates = Jinja2Templates(directory=str(settings.templates_dir))
_request_times: dict[str, deque[float]] = defaultdict(deque)
_analysis_slots = asyncio.Semaphore(2)


@app.middleware("http")
async def protect_health_data_responses(request: Request, call_next):
    if settings.authentication_enabled and request.url.path not in {"/login", "/api/health", "/api/live"} and not request.url.path.startswith("/static/"):
        if current_user(request) is None:
            if request.url.path.startswith("/api/"):
                return JSONResponse(status_code=401, content={"detail": "Authentication required."})
            return RedirectResponse("/login", status_code=303)
    response = await call_next(request)
    if not request.url.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self'")
        if settings.secure_transport:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


# Added after the HTTP middleware so signed session data is decoded before
# access control and CSRF checks run.
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    max_age=settings.session_max_age_seconds,
    same_site="lax",
    https_only=settings.secure_transport,
)


@app.get("/api/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


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
        "authentication_required": settings.authentication_enabled,
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {"app_name": settings.app_name, "csrf_token": csrf_token(request), "error": ""})


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, email: str = Form(...), password: str = Form(...), csrf: str = Form(...)):
    verify_csrf(request, csrf)
    user = authenticate(email, password)
    if user is None:
        return templates.TemplateResponse(request, "login.html", {"app_name": settings.app_name, "csrf_token": csrf_token(request), "error": "Invalid email or password."}, status_code=401)
    request.session.clear()
    request.session["user"] = user
    csrf_token(request)
    append_audit_event(str(uuid.uuid4()), user["id"], "login_succeeded")
    return RedirectResponse("/", status_code=303)


@app.post("/logout")
async def logout(request: Request, csrf: str = Form(...)):
    verify_csrf(request, csrf)
    user = current_user(request)
    if user:
        append_audit_event(str(uuid.uuid4()), user["id"], "logout")
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


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
async def get_history(request: Request) -> dict:
    return await history_payload(request)


@app.post("/api/analyze")
async def analyze_prescription(request: Request, file: UploadFile, consent: bool = Form(False), csrf: str | None = Form(None)):
    key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _request_times[key]
    while bucket and bucket[0] < now - 60:
        bucket.popleft()
    if len(bucket) >= 10:
        return JSONResponse(status_code=429, content={"detail": "Too many analysis requests. Try again later."})
    bucket.append(now)
    async with _analysis_slots:
        return await analyze(request, file, consent, csrf)


@app.get("/api/report/{analysis_id}")
async def get_report(request: Request, analysis_id: str):
    return await download_report(request, analysis_id)


@app.patch("/api/analyses/{analysis_id}/review")
async def review(request: Request, analysis_id: str):
    return await review_analysis(request, analysis_id, await request.json())
