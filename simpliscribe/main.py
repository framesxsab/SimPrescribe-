import asyncio
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .config import settings
from .metrics import generate_prometheus_metrics, get_metrics_snapshot, record_http_request
from .retrieval import get_retriever, get_vector_cache
from .security import authenticate, authenticate_oidc_callback, current_user, csrf_token, oidc_authorization_url, owner_id, require_edit_role, verify_csrf
from .storage import append_audit_event, ensure_schema, load_audit_events, load_history, ping_database
from .web import analyze, download_report, export_audit_csv, history_payload, render_dashboard, render_details, render_history, review_analysis

settings.validate_runtime()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
ensure_schema()
load_history()

app = FastAPI(title=f"{settings.app_name} API")
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
templates = Jinja2Templates(directory=str(settings.templates_dir))
_request_times: dict[str, deque[float]] = defaultdict(deque)
_login_times: dict[str, deque[float]] = defaultdict(deque)
_MAX_BUCKETS = 4096
_analysis_slots = asyncio.Semaphore(2)


def _rate_limit_key(request: Request) -> str:
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip() if forwarded else ""
        if first:
            return first
    return request.client.host if request.client else "unknown"


def _login_page_context(request: Request, error: str = "") -> dict[str, object]:
    return {
        "app_name": settings.app_name,
        "csrf_token": csrf_token(request),
        "error": error,
        "oidc_enabled": settings.oidc_enabled,
        "bootstrap_admin_enabled": settings.bootstrap_admin_enabled,
    }


def _consume_bucket(buckets: dict[str, deque[float]], key: str, now: float, window: float, limit: int) -> bool:
    if key not in buckets and len(buckets) >= _MAX_BUCKETS:
        buckets.pop(next(iter(buckets)))
    bucket = buckets.setdefault(key, deque())
    while bucket and bucket[0] < now - window:
        bucket.popleft()
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


@app.middleware("http")
async def protect_health_data_responses(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = None
    if settings.authentication_enabled and request.url.path not in {"/login", "/login/oidc", "/auth/callback", "/api/health", "/api/live", "/api/metrics"} and not request.url.path.startswith("/static/"):
        if current_user(request) is None:
            if request.url.path.startswith("/api/"):
                response = JSONResponse(status_code=401, content={"detail": "Authentication required."})
            else:
                response = RedirectResponse("/login", status_code=303)
    if response is None:
        response = await call_next(request)
    response.headers.setdefault("X-Request-ID", request_id)
    record_http_request(request.method, response.status_code)
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
    database_ready = ping_database()
    return {
        "status": "ready" if datasets_ready and provider_ready and database_ready else "degraded",
        "datasets_ready": datasets_ready,
        "database_ready": database_ready,
        "configured_provider": settings.inference_provider,
        "provider_ready": provider_ready,
        "clinical_use": "human_review_required",
        "authentication_required": settings.authentication_enabled,
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", _login_page_context(request))


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, email: str = Form(...), password: str = Form(...), csrf: str = Form(...)):
    if settings.oidc_enabled and not settings.bootstrap_admin_enabled:
        raise HTTPException(status_code=404, detail="Use organization sign-in.")
    verify_csrf(request, csrf)
    if not _consume_bucket(_login_times, _rate_limit_key(request), time.monotonic(), 60, 20):
        return templates.TemplateResponse(request, "login.html", _login_page_context(request, "Too many login attempts. Try again later."), status_code=429)
    user = authenticate(email, password)
    if user is None:
        return templates.TemplateResponse(request, "login.html", _login_page_context(request, "Invalid email or password."), status_code=401)
    request.session.clear()
    request.session["user"] = user
    csrf_token(request)
    method = "bootstrap" if settings.oidc_enabled else "password"
    append_audit_event(str(uuid.uuid4()), user["id"], "login_succeeded", method=method)
    return RedirectResponse("/", status_code=303)


@app.get("/login/oidc")
async def login_oidc(request: Request):
    return RedirectResponse(await oidc_authorization_url(request), status_code=303)


@app.get("/auth/callback", response_class=HTMLResponse)
async def oidc_callback(request: Request, state: str = "", code: str = "", error: str = ""):
    if error:
        return templates.TemplateResponse(request, "login.html", _login_page_context(request, "Organization sign-in was not completed."), status_code=401)
    try:
        user = await authenticate_oidc_callback(request, state, code)
    except HTTPException as exc:
        return templates.TemplateResponse(request, "login.html", _login_page_context(request, exc.detail), status_code=exc.status_code)
    request.session.clear()
    request.session["user"] = user
    csrf_token(request)
    append_audit_event(str(uuid.uuid4()), user["id"], "login_succeeded", method="oidc")
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


@app.get("/api/audit")
async def get_audit(request: Request) -> dict:
    return {"events": load_audit_events(owner_id(request))}


@app.post("/api/analyze")
async def analyze_prescription(request: Request, file: UploadFile, consent: bool = Form(False), csrf: str | None = Form(None)):
    key = _rate_limit_key(request)
    now = time.monotonic()
    if not _consume_bucket(_request_times, key, now, 60, 10):
        return JSONResponse(status_code=429, content={"detail": "Too many analysis requests. Try again later."})
    async with _analysis_slots:
        return await analyze(request, file, consent, csrf)


@app.get("/api/report/{analysis_id}")
async def get_report(request: Request, analysis_id: str):
    return await download_report(request, analysis_id)


@app.patch("/api/analyses/{analysis_id}/review")
async def review(request: Request, analysis_id: str):
    return await review_analysis(request, analysis_id, await request.json())


@app.get("/api/retrieval/similar")
async def similar_prescriptions(q: str = "", limit: int = 5, min_similarity: float = 0.2) -> dict:
    retriever = get_retriever()
    results = retriever.query_similar(q, top_k=limit, min_similarity=min_similarity)
    return {"query": q, "count": len(results), "results": results}


@app.get("/api/cache/stats")
async def cache_stats() -> dict:
    return get_vector_cache().stats()


@app.post("/api/cache/clear")
async def cache_clear(request: Request) -> dict:
    require_edit_role(request)
    get_vector_cache().clear()
    return {"status": "cleared"}


@app.get("/api/metrics")
async def metrics(request: Request, format: str = ""):
    accept = request.headers.get("accept", "")
    if format == "prometheus" or "text/plain" in accept:
        from fastapi.responses import Response
        return Response(content=generate_prometheus_metrics(), media_type="text/plain; version=0.0.4")
    return get_metrics_snapshot()


@app.get("/api/audit/export")
async def export_audit(request: Request, format: str = "json"):
    from fastapi.responses import Response
    owner = owner_id(request)
    events = load_audit_events(owner, limit=500)
    if format.lower() == "csv":
        csv_data = export_audit_csv(events)
        headers = {
            "Content-Disposition": 'attachment; filename="simpliscribe_audit_events.csv"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        }
        return Response(content=csv_data, media_type="text/csv", headers=headers)
    return {"owner_id": owner, "count": len(events), "events": events}



