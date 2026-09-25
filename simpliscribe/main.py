import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError, OperationalError
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .config import settings
from .metrics import generate_prometheus_metrics, get_metrics_snapshot, record_http_request
from .retrieval import get_retriever, get_vector_cache
from .marketplace import MarketplaceConflict, accept_order, cancel_order, confirmed_prescription, create_order, deactivate_inventory_item, decline_order, list_orders_for, matching_pharmacies, order_detail, pharmacy_inventory, pharmacy_transition, quote_order, save_inventory_item, valid_pin
from .schemas import CacheStatsResponse, HealthResponse, InventoryRequest, LiveResponse, OrderRequest, PatientReviewRequest, QuoteRequest, SimilarPrescriptionsResponse, TransitionRequest
from .security import authenticate, authenticate_oidc_callback, current_user, csrf_token, hash_password, oidc_authorization_url, owner_id, require_edit_role, require_role, verify_csrf
from .storage import append_audit_event, create_user, ensure_schema, get_analysis_record, get_pharmacy_by_user, get_user, get_user_by_email, load_audit_events, load_history, list_pharmacies, ping_database, purge_expired_marketplace, seed_test_pharmacies, set_pharmacy_approval
from .ocr import get_ocr_state, warm_ocr_reader
from .web import analyze, download_report, export_audit_csv, history_payload, patient_review_analysis, prescription_source, render_dashboard, render_details, render_history, retry_processing, review_analysis, save_patient_draft, start_analysis

logger = logging.getLogger(__name__)

settings.validate_runtime()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.prescription_storage_dir.mkdir(parents=True, exist_ok=True)
ensure_schema()
if not settings.production:
    seed_test_pharmacies()
for expired_name in purge_expired_marketplace()[0]:
    try:
        (settings.prescription_storage_dir / expired_name).unlink(missing_ok=True)
    except OSError:
        pass
load_history()

@asynccontextmanager
async def lifespan(application: FastAPI):
    if settings.app_env.strip().lower() == "test":
        yield
        return
    application.state.ocr_warmup_task = asyncio.create_task(asyncio.to_thread(warm_ocr_reader))
    try:
        yield
    finally:
        task = getattr(application.state, "ocr_warmup_task", None)
        if task and not task.done():
            task.cancel()


app = FastAPI(title=f"{settings.app_name} API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
templates = Jinja2Templates(directory=str(settings.templates_dir))


def format_patient_datetime(value: object) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        local = parsed.astimezone()
        hour = local.strftime("%I").lstrip("0") or "12"
        return f"{local.day} {local.strftime('%b %Y')}, {hour}:{local:%M} {local:%p}"
    except (TypeError, ValueError):
        return str(value)


templates.env.filters["patient_datetime"] = format_patient_datetime
_request_times: dict[str, deque[float]] = defaultdict(deque)
_login_times: dict[str, deque[float]] = defaultdict(deque)
_MAX_BUCKETS = 4096
_analysis_slots = asyncio.Semaphore(2)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> RedirectResponse:
    return RedirectResponse("/static/favicon.svg", status_code=307)


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
    public_paths = {"/login", "/login/oidc", "/auth/callback", "/register", "/register/patient", "/register/pharmacy", "/api/health", "/api/live", "/api/metrics"}
    if settings.authentication_enabled and request.url.path not in public_paths and not request.url.path.startswith("/static/"):
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


@app.get("/api/live", response_model=LiveResponse)
async def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/api/health", response_model=HealthResponse)
async def health() -> dict:
    datasets_ready = settings.india_medicine_dataset.exists() and settings.medicine_database_dataset.exists()
    provider_ready = settings.inference_provider == "fallback" or bool(
        settings.hf_token if settings.inference_provider == "huggingface" else settings.model_api_url
    )
    database_ready = ping_database()
    ocr = get_ocr_state()
    return {
        "status": "ready" if datasets_ready and provider_ready and database_ready and ocr["ready"] else "degraded",
        "datasets_ready": datasets_ready,
        "database_ready": database_ready,
        "configured_provider": settings.inference_provider,
        "provider_ready": provider_ready,
        "clinical_use": "human_review_required",
        "authentication_required": settings.authentication_enabled,
        "ocr_state": ocr["state"],
        "ocr_ready": ocr["ready"],
        "ocr_error_code": ocr["error_code"],
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", _login_page_context(request))


def _registration_context(request: Request, role: str, error: str = "", success: str = "") -> dict[str, object]:
    return {"app_name": settings.app_name, "role": role, "csrf_token": csrf_token(request), "error": error, "success": success}


@app.get("/register/{role}", response_class=HTMLResponse)
async def register_page(request: Request, role: str) -> HTMLResponse:
    if role not in {"patient", "pharmacy"}:
        raise HTTPException(status_code=404, detail="Registration type not found.")
    return templates.TemplateResponse(request, "register.html", _registration_context(request, role))


@app.post("/register/{role}", response_class=HTMLResponse)
async def register_account(request: Request, role: str) -> HTMLResponse:
    if role not in {"patient", "pharmacy"}:
        raise HTTPException(status_code=404, detail="Registration type not found.")
    form = await request.form()
    verify_csrf(request, str(form.get("csrf") or ""))
    email = str(form.get("email") or "").strip().lower()[:255]
    full_name = str(form.get("full_name") or "").strip()[:255]
    phone = str(form.get("phone") or "").strip()[:32]
    pin_code = str(form.get("pin_code") or "").strip()
    if "@" not in email or not full_name or not phone or not valid_pin(pin_code):
        return templates.TemplateResponse(request, "register.html", _registration_context(request, role, "Enter a valid name, email, phone number, and six-digit Indian PIN code."), status_code=400)
    try:
        password_hash = hash_password(str(form.get("password") or ""))
        user_id = str(uuid.uuid4())
        user = {"id": user_id, "email": email, "password_hash": password_hash, "role": role,
                "full_name": full_name, "phone": phone, "pin_code": pin_code, "active": True,
                "created_at": datetime.now(timezone.utc)}
        pharmacy = None
        if role == "pharmacy":
            business_name = str(form.get("business_name") or "").strip()[:255]
            license_number = str(form.get("license_number") or "").strip()[:128]
            address = str(form.get("address") or "").strip()[:1000]
            serviceable = [item.strip() for item in str(form.get("serviceable_pins") or "").split(",") if item.strip()]
            if not business_name or not license_number or not address or any(not valid_pin(item) for item in serviceable):
                raise ValueError("Enter complete pharmacy details and valid comma-separated serviceable PIN codes.")
            pharmacy = {"id": str(uuid.uuid4()), "user_id": user_id, "business_name": business_name,
                        "license_number": license_number, "address": address, "pin_code": pin_code,
                        "serviceable_pins_json": json.dumps(sorted(set(serviceable))),
                        "supports_pickup": bool(form.get("supports_pickup")), "supports_delivery": bool(form.get("supports_delivery")),
                        "approval_status": "pending", "approved_at": None}
            if not pharmacy["supports_pickup"] and not pharmacy["supports_delivery"]:
                raise ValueError("Select pickup, local delivery, or both.")
        create_user(user, pharmacy)
    except ValueError as exc:
        return templates.TemplateResponse(request, "register.html", _registration_context(request, role, str(exc)), status_code=400)
    except Exception:
        return templates.TemplateResponse(request, "register.html", _registration_context(request, role, "That email or pharmacy licence is already registered."), status_code=409)
    message = "Account created. You can sign in now." if role == "patient" else "Registration submitted. An administrator must approve the pharmacy before sign-in."
    return templates.TemplateResponse(request, "register.html", _registration_context(request, role, success=message), status_code=201)


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, email: str = Form(...), password: str = Form(...), csrf: str = Form(...)):
    if settings.oidc_enabled and not settings.bootstrap_admin_enabled and get_user_by_email(email.strip().lower()) is None:
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
    user = current_user(request)
    if user and user.get("role") == "pharmacy":
        return RedirectResponse("/pharmacy", status_code=303)
    if user and user.get("role") == "admin":
        return RedirectResponse("/admin/pharmacies", status_code=303)
    return await render_dashboard(request, templates)


@app.get("/history", response_class=HTMLResponse)
async def serve_history(request: Request) -> HTMLResponse:
    return await render_history(request, templates)


@app.get("/register", response_class=HTMLResponse)
async def register_chooser(request: Request) -> RedirectResponse:
    return RedirectResponse("/register/patient", status_code=303)


@app.get("/marketplace", response_class=HTMLResponse)
async def marketplace_page(request: Request) -> HTMLResponse:
    user = current_user(request)
    if user and user.get("role") != "patient":
        raise HTTPException(status_code=403, detail="Patient account required.")
    analyses = load_history(user["id"]) if user else []
    return templates.TemplateResponse(request, "marketplace.html", {
        "analyses": analyses,
        "app_name": settings.app_name,
        "user": user,
        "csrf_token": csrf_token(request),
        "current": "marketplace",
    })


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


@app.post("/api/analyses/start")
async def start_prescription(request: Request, file: UploadFile, consent: bool = Form(False), csrf: str | None = Form(None)):
    key = _rate_limit_key(request)
    if not _consume_bucket(_request_times, key, time.monotonic(), 60, 10):
        return JSONResponse(status_code=429, content={"detail": "Too many analysis requests. Try again later."})
    return await start_analysis(request, file, consent, csrf)


@app.get("/api/analyses/{analysis_id}/status")
async def prescription_status(request: Request, analysis_id: str):
    analysis = get_analysis_record(analysis_id, owner_id(request))
    if analysis is None:
        raise HTTPException(status_code=404, detail="Prescription analysis not found.")
    return {"analysis_id": analysis_id, "prescription_state": analysis.get("prescription_state", "review_required"),
            "processing_stage": analysis.get("processing_stage", ""), "error_code": analysis.get("error_code")}


@app.post("/api/analyses/{analysis_id}/process")
async def process_prescription(request: Request, analysis_id: str):
    async with _analysis_slots:
        return await retry_processing(request, analysis_id, allow_uploaded=True)


@app.get("/api/report/{analysis_id}")
async def get_report(request: Request, analysis_id: str):
    return await download_report(request, analysis_id)


@app.post("/api/analyses/{analysis_id}/retry")
async def retry_prescription(request: Request, analysis_id: str):
    return await retry_processing(request, analysis_id)


@app.patch("/api/analyses/{analysis_id}/review")
async def review(request: Request, analysis_id: str):
    return await review_analysis(request, analysis_id, await request.json())


@app.patch("/api/analyses/{analysis_id}/patient-review")
async def patient_review(request: Request, analysis_id: str, payload: PatientReviewRequest):
    return await patient_review_analysis(request, analysis_id, payload.model_dump())


@app.patch("/api/analyses/{analysis_id}/patient-draft")
async def patient_draft(request: Request, analysis_id: str):
    return await save_patient_draft(request, analysis_id, await request.json())


@app.get("/api/analyses/{analysis_id}/source")
async def get_prescription_source(request: Request, analysis_id: str):
    return await prescription_source(request, analysis_id)


def _marketplace_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, MarketplaceConflict):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, LookupError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (IntegrityError, OperationalError)):
        logger.warning("Marketplace write contention; error_code=MARKETPLACE_CONFLICT")
        return HTTPException(status_code=409, detail="The order changed. Reload and review its current status.")
    if isinstance(exc, json.JSONDecodeError):
        logger.error("Invalid persisted marketplace data; error_code=MARKETPLACE_DATA_INVALID")
        return HTTPException(status_code=500, detail="The marketplace request could not be completed.")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.error(
        "Marketplace operation failed; error_code=MARKETPLACE_INTERNAL_ERROR",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return HTTPException(status_code=500, detail="The marketplace request could not be completed.")


@app.get("/api/analyses/{analysis_id}/pharmacies")
async def find_pharmacies(request: Request, analysis_id: str, pin: str | None = None) -> dict:
    user = current_user(request)
    if settings.authentication_enabled or user is not None:
        user = require_role(request, "patient")
        profile = get_user(user["id"])
        if not profile or profile.get("active") is not True:
            raise HTTPException(status_code=403, detail="An active patient account is required.")
        patient_id = user["id"]
        default_pin = profile.get("pin_code") or ""
    else:
        patient_id = "local"
        default_pin = "560001"
    try:
        analysis = confirmed_prescription(patient_id, analysis_id)
        search_pin = (pin or default_pin).strip()
        if not valid_pin(search_pin):
            raise HTTPException(status_code=400, detail="Enter a valid six-digit PIN code.")
        medications = analysis.get("medications") or []
        return {"pharmacies": matching_pharmacies(
            search_pin,
            [item.get("name", "") for item in medications if isinstance(item, dict)],
        )}
    except HTTPException:
        raise
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.get("/api/inventory")
async def get_inventory(request: Request) -> dict:
    user = require_role(request, "pharmacy")
    try:
        return {"items": pharmacy_inventory(user["id"])}
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.post("/api/inventory", status_code=201)
async def create_inventory(request: Request, payload: InventoryRequest) -> dict:
    user = require_role(request, "pharmacy")
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    try:
        return save_inventory_item(user["id"], payload.model_dump())
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.put("/api/inventory/{item_id}")
async def update_inventory(request: Request, item_id: str, payload: InventoryRequest) -> dict:
    user = require_role(request, "pharmacy")
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    try:
        return save_inventory_item(user["id"], payload.model_dump(), item_id)
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.delete("/api/inventory/{item_id}")
async def delete_inventory(request: Request, item_id: str) -> dict:
    user = require_role(request, "pharmacy")
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    try:
        deactivate_inventory_item(user["id"], item_id)
        return {"id": item_id, "active": False}
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.post("/api/orders")
async def request_order(request: Request, payload: OrderRequest) -> JSONResponse:
    user = require_role(request, "patient")
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    try:
        order_id, created = create_order(user["id"], **payload.model_dump(), return_created=True)
        if created:
            try:
                append_audit_event(
                    str(uuid.uuid4()),
                    user["id"],
                    "order_requested",
                    payload.analysis_id,
                    order_id=order_id,
                )
            except Exception:
                logger.warning("order_id=%s error_code=ORDER_AUDIT_FAILED", order_id)
        status = order_detail(order_id, user)["status"]
        return JSONResponse(
            status_code=201 if created else 200,
            content={"id": order_id, "status": status, "created": created},
        )
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.get("/api/orders/{order_id}")
async def get_order(request: Request, order_id: str) -> dict:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        return order_detail(order_id, user)
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.post("/api/orders/{order_id}/quote")
async def submit_quote(request: Request, order_id: str, payload: QuoteRequest) -> dict:
    user = require_role(request, "pharmacy")
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    try:
        quote_order(user["id"], order_id, [item.model_dump() for item in payload.items])
        return {"id": order_id, "status": "quoted"}
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.post("/api/orders/{order_id}/decline")
async def reject_order(request: Request, order_id: str, payload: TransitionRequest) -> dict:
    user = require_role(request, "pharmacy")
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    try:
        decline_order(user["id"], order_id, payload.note)
        return {"id": order_id, "status": "declined"}
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.post("/api/orders/{order_id}/accept")
async def accept_quote(request: Request, order_id: str) -> dict:
    user = require_role(request, "patient")
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    try:
        accept_order(user["id"], order_id)
        return {"id": order_id, "status": "accepted"}
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.post("/api/orders/{order_id}/cancel")
async def cancel_patient_order(request: Request, order_id: str, payload: TransitionRequest) -> dict:
    user = require_role(request, "patient")
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    try:
        cancel_order(user["id"], order_id, payload.note)
        return {"id": order_id, "status": "cancelled"}
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.post("/api/orders/{order_id}/status")
async def update_order_status(request: Request, order_id: str, payload: TransitionRequest) -> dict:
    user = require_role(request, "pharmacy")
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    try:
        pharmacy_transition(user["id"], order_id, payload.status, payload.note)
        return {"id": order_id, "status": payload.status}
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc


@app.post("/api/admin/pharmacies/{pharmacy_id}/approval")
async def approve_pharmacy(request: Request, pharmacy_id: str, payload: TransitionRequest) -> dict:
    user = require_role(request, "admin")
    verify_csrf(request, request.headers.get("X-CSRF-Token"))
    if payload.status not in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="Approval status must be approved or rejected.")
    if not set_pharmacy_approval(pharmacy_id, payload.status):
        raise HTTPException(status_code=404, detail="Pharmacy not found.")
    append_audit_event(str(uuid.uuid4()), user["id"], "pharmacy_approval_changed", pharmacy_id=pharmacy_id, status=payload.status)
    return {"id": pharmacy_id, "status": payload.status}


@app.get("/orders", response_class=HTMLResponse)
async def orders_page(request: Request) -> HTMLResponse:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        orders_for_user = list_orders_for(user)
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc
    return templates.TemplateResponse(request, "orders.html", {"orders": orders_for_user, "app_name": settings.app_name,
                                                                   "user": user, "csrf_token": csrf_token(request), "current": "orders"})


@app.get("/orders/{order_id}", response_class=HTMLResponse)
async def order_page(request: Request, order_id: str) -> HTMLResponse:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        order = order_detail(order_id, user)
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc
    return templates.TemplateResponse(request, "order.html", {"order": order, "app_name": settings.app_name,
                                                                  "inventory": pharmacy_inventory(user["id"], active_only=True) if user["role"] == "pharmacy" else [],
                                                                  "user": user, "csrf_token": csrf_token(request), "current": "orders"})


@app.get("/pharmacy", response_class=HTMLResponse)
async def pharmacy_page(request: Request) -> HTMLResponse:
    user = require_role(request, "pharmacy")
    try:
        inventory_items = pharmacy_inventory(user["id"])
        pharmacy_orders = list_orders_for(user)
    except Exception as exc:
        raise _marketplace_http_error(exc) from exc
    return templates.TemplateResponse(request, "pharmacy.html", {"profile": get_pharmacy_by_user(user["id"]),
                                                                     "inventory": inventory_items, "orders": pharmacy_orders,
                                                                     "app_name": settings.app_name, "user": user, "csrf_token": csrf_token(request), "current": "pharmacy"})


@app.get("/admin/pharmacies", response_class=HTMLResponse)
async def admin_pharmacies_page(request: Request) -> HTMLResponse:
    user = require_role(request, "admin")
    return templates.TemplateResponse(request, "admin_pharmacies.html", {"pharmacies": list_pharmacies(),
                                                                            "app_name": settings.app_name, "user": user,
                                                                            "csrf_token": csrf_token(request), "current": "admin"})


@app.get("/api/retrieval/similar", response_model=SimilarPrescriptionsResponse)
async def similar_prescriptions(q: str = "", limit: int = 5, min_similarity: float = 0.2) -> dict:
    retriever = get_retriever()
    results = retriever.query_similar(q, top_k=limit, min_similarity=min_similarity)
    return {"query": q, "count": len(results), "results": results}


@app.get("/api/cache/stats", response_model=CacheStatsResponse)
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



