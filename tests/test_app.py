import asyncio
from io import BytesIO
from copy import deepcopy
import re
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import fitz
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from simpliscribe.main import app
from simpliscribe.inference import fallback_extract
from simpliscribe.inference import build_medication_record
from simpliscribe.inference import call_huggingface
from simpliscribe.inference import refine_model_medications
from simpliscribe.ocr import OCRLine, OCRResult
from simpliscribe import ocr
from simpliscribe.security import _oidc_configuration, oidc_user_from_claims
from simpliscribe.storage import append_audit_event, append_history, load_history, save_history, update_analysis_record
from simpliscribe.storage import get_analysis_record
from simpliscribe.config import Settings, settings


client = TestClient(app)


def test_huggingface_client_uses_configured_timeout(monkeypatch):
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def chat_completion(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"medications": []}'))]
            )

    monkeypatch.setattr(
        "simpliscribe.inference.settings",
        SimpleNamespace(hf_token="test-token", hf_model="test-model", request_timeout_seconds=12.5),
    )
    monkeypatch.setattr("simpliscribe.inference.InferenceClient", FakeClient)
    monkeypatch.setattr("simpliscribe.inference.refine_model_medications", lambda _, medications: medications)

    result = call_huggingface("Paracetamol 650 mg")

    assert result["medications"] == []
    assert captured["token"] == "test-token"
    assert captured["timeout"] == 12.5


def test_ocr_disables_mkldnn_for_cpu_compatibility(monkeypatch):
    captured: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ocr, "_ocr_reader", None)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOCR))

    ocr.get_ocr_reader()

    assert captured["device"] == "cpu"
    assert captured["enable_mkldnn"] is False


def test_dashboard_route():
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="/static/favicon.svg"' in response.text
    assert client.get("/static/favicon.svg").status_code == 200


def test_history_api_route():
    response = client.get("/api/history")
    assert response.status_code == 200
    assert "analyses" in response.json()


def test_history_page_exposes_review_triage():
    record = {
        "id": "history-triage-record",
        "created_at": "2026-07-31T00:00:00+00:00",
        "filename": "triage.pdf",
        "review_status": "needs_review",
        "medications": [],
    }
    try:
        append_history(record)
        response = client.get("/history")

        assert response.status_code == 200
        assert 'id="history-search"' in response.text
        assert 'id="review-filter"' in response.text
        assert 'data-review-status="needs_review"' in response.text
    finally:
        save_history([])


def test_audit_api_returns_only_local_owner_events():
    local_event_id = str(uuid4())
    other_event_id = str(uuid4())
    append_audit_event(local_event_id, "local", "analysis_reviewed", "test-analysis", status="confirmed")
    append_audit_event(other_event_id, "other-owner", "analysis_reviewed", "test-analysis", status="confirmed")

    response = client.get("/api/audit")

    assert response.status_code == 200
    event = next(item for item in response.json()["events"] if item["id"] == local_event_id)
    assert event == {
        "id": local_event_id,
        "analysis_id": "test-analysis",
        "event_type": "analysis_reviewed",
        "created_at": event["created_at"],
        "metadata": {"status": "confirmed"},
    }
    assert all(item["id"] != other_event_id for item in response.json()["events"])


def test_health_route_exposes_review_boundary():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["clinical_use"] == "human_review_required"


def test_authentication_redirects_and_creates_secure_session():
    original_required = settings.auth_required
    original_email = settings.admin_email
    original_password = settings.admin_password
    original_role = settings.admin_role
    object.__setattr__(settings, "auth_required", True)
    object.__setattr__(settings, "admin_email", "reviewer@example.test")
    object.__setattr__(settings, "admin_password", "correct horse battery staple")
    auth_client = TestClient(app)
    try:
        response = auth_client.get("/", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-frame-options"] == "DENY"

        login_page = auth_client.get("/login")
        csrf = re.search(r'name="csrf" value="([^"]+)"', login_page.text).group(1)
        login_response = auth_client.post(
            "/login",
            data={"email": "reviewer@example.test", "password": "correct horse battery staple", "csrf": csrf},
            follow_redirects=False,
        )
        assert login_response.status_code == 303
        assert login_response.headers["location"] == "/"
        dashboard = auth_client.get("/")
        assert dashboard.status_code == 200
        logout_csrf = re.search(r'action="/logout"[\s\S]*?name="csrf" value="([^"]+)"', dashboard.text).group(1)
        logout_response = auth_client.post("/logout", data={"csrf": logout_csrf}, follow_redirects=False)
        assert logout_response.status_code == 303
        assert logout_response.headers["location"] == "/login"
        assert auth_client.get("/", follow_redirects=False).status_code == 303
    finally:
        object.__setattr__(settings, "auth_required", original_required)
        object.__setattr__(settings, "admin_email", original_email)
        object.__setattr__(settings, "admin_password", original_password)
        object.__setattr__(settings, "admin_role", original_role)


def test_production_configuration_fails_closed_without_required_secrets():
    with pytest.raises(RuntimeError, match="Unsafe runtime configuration"):
        Settings(
            app_env="production",
            database_url="sqlite:///data/simpliscribe.db",
            session_secret="development-only-change-me",
            admin_password="",
        ).validate_runtime()


def test_production_configuration_requires_an_admin_email():
    with pytest.raises(RuntimeError, match="ADMIN_EMAIL is required"):
        Settings(
            app_env="production",
            database_url="postgresql://example.test/simpliscribe",
            session_secret="a" * 32,
            admin_email=" ",
            admin_password="correct horse battery staple",
        ).validate_runtime()


def test_production_configuration_requires_postgresql():
    with pytest.raises(RuntimeError, match="DATABASE_URL must use PostgreSQL"):
        Settings(
            app_env="production",
            database_url="mysql://example.test/simpliscribe",
            session_secret="a" * 32,
            admin_password="correct horse battery staple",
        ).validate_runtime()


def test_runtime_configuration_rejects_an_invalid_session_lifetime():
    with pytest.raises(RuntimeError, match="SESSION_MAX_AGE_SECONDS must be at least 1"):
        Settings(session_max_age_seconds=0).validate_runtime()


def test_runtime_configuration_rejects_an_invalid_bootstrap_role():
    with pytest.raises(RuntimeError, match="ADMIN_ROLE must be admin, reviewer, or auditor"):
        Settings(admin_role="operator").validate_runtime()


def test_production_configuration_accepts_complete_oidc_without_bootstrap_password():
    Settings(
        app_env="production",
        database_url="postgresql://example.test/simpliscribe",
        session_secret="a" * 32,
        admin_password="",
        oidc_issuer="https://identity.example.test",
        oidc_client_id="simpliscribe",
        oidc_client_secret="secret",
        oidc_redirect_uri="https://app.example.test/auth/callback",
    ).validate_runtime()


def test_runtime_configuration_rejects_partial_oidc_setup():
    with pytest.raises(RuntimeError, match="OIDC_ISSUER, OIDC_CLIENT_ID, OIDC_CLIENT_SECRET, and OIDC_REDIRECT_URI"):
        Settings(oidc_issuer="https://identity.example.test").validate_runtime()


def test_oidc_claims_map_to_least_privilege_roles(monkeypatch):
    oidc_settings = Settings(
        oidc_issuer="https://identity.example.test",
        oidc_client_id="simpliscribe",
        oidc_client_secret="secret",
        oidc_redirect_uri="https://app.example.test/auth/callback",
        oidc_admin_subjects="admin-subject",
        oidc_reviewer_subjects="reviewer-subject",
    )
    monkeypatch.setattr("simpliscribe.security.settings", oidc_settings)

    assert oidc_user_from_claims({"sub": "admin-subject", "email": "admin@example.test"})["role"] == "admin"
    assert oidc_user_from_claims({"sub": "reviewer-subject"})["role"] == "reviewer"
    assert oidc_user_from_claims({"sub": "unmapped-subject"})["role"] == "auditor"


def test_oidc_login_uses_the_provider_redirect(monkeypatch):
    original = (settings.oidc_issuer, settings.oidc_client_id, settings.oidc_client_secret, settings.oidc_redirect_uri)
    object.__setattr__(settings, "oidc_issuer", "https://identity.example.test")
    object.__setattr__(settings, "oidc_client_id", "simpliscribe")
    object.__setattr__(settings, "oidc_client_secret", "secret")
    object.__setattr__(settings, "oidc_redirect_uri", "https://app.example.test/auth/callback")

    async def fake_authorization_url(_):
        return "https://identity.example.test/authorize?state=test"

    monkeypatch.setattr("simpliscribe.main.oidc_authorization_url", fake_authorization_url)
    oidc_client = TestClient(app)
    try:
        login_page = oidc_client.get("/login")
        assert "Continue with organization sign-in" in login_page.text
        response = oidc_client.get("/login/oidc", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "https://identity.example.test/authorize?state=test"
    finally:
        object.__setattr__(settings, "oidc_issuer", original[0])
        object.__setattr__(settings, "oidc_client_id", original[1])
        object.__setattr__(settings, "oidc_client_secret", original[2])
        object.__setattr__(settings, "oidc_redirect_uri", original[3])


def test_oidc_callback_exchanges_pkce_code_and_creates_a_reviewer_session(monkeypatch):
    original = (
        settings.oidc_issuer,
        settings.oidc_client_id,
        settings.oidc_client_secret,
        settings.oidc_redirect_uri,
        settings.oidc_reviewer_subjects,
    )
    object.__setattr__(settings, "oidc_issuer", "https://identity.example.test")
    object.__setattr__(settings, "oidc_client_id", "simpliscribe")
    object.__setattr__(settings, "oidc_client_secret", "secret")
    object.__setattr__(settings, "oidc_redirect_uri", "https://app.example.test/auth/callback")
    object.__setattr__(settings, "oidc_reviewer_subjects", "reviewer-subject")

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        post_data = None

        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url, headers=None):
            if url.endswith("/.well-known/openid-configuration"):
                return FakeResponse({
                    "issuer": "https://identity.example.test",
                    "authorization_endpoint": "https://identity.example.test/authorize",
                    "token_endpoint": "https://identity.example.test/token",
                    "userinfo_endpoint": "https://identity.example.test/userinfo",
                })
            assert url == "https://identity.example.test/userinfo"
            assert headers == {"Authorization": "Bearer access-token"}
            return FakeResponse({"sub": "reviewer-subject", "email": "reviewer@example.test"})

        async def post(self, url, data):
            assert url == "https://identity.example.test/token"
            assert data["code"] == "provider-code"
            assert data["code_verifier"]
            FakeClient.post_data = data
            return FakeResponse({"access_token": "access-token"})

    monkeypatch.setattr("simpliscribe.security.httpx.AsyncClient", FakeClient)
    oidc_client = TestClient(app)
    try:
        start = oidc_client.get("/login/oidc", follow_redirects=False)
        state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
        callback = oidc_client.get(f"/auth/callback?state={state}&code=provider-code", follow_redirects=False)
        assert callback.status_code == 303
        assert FakeClient.post_data["client_id"] == "simpliscribe"
        assert oidc_client.get("/").status_code == 200
    finally:
        object.__setattr__(settings, "oidc_issuer", original[0])
        object.__setattr__(settings, "oidc_client_id", original[1])
        object.__setattr__(settings, "oidc_client_secret", original[2])
        object.__setattr__(settings, "oidc_redirect_uri", original[3])
        object.__setattr__(settings, "oidc_reviewer_subjects", original[4])


def test_oidc_discovery_rejects_insecure_endpoints(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "issuer": "https://identity.example.test",
                "authorization_endpoint": "https://identity.example.test/authorize",
                "token_endpoint": "http://identity.example.test/token",
                "userinfo_endpoint": "https://identity.example.test/userinfo",
            }

    class FakeClient:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, _):
            return FakeResponse()

    monkeypatch.setattr("simpliscribe.security.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(
        "simpliscribe.security.settings",
        SimpleNamespace(oidc_enabled=True, oidc_issuer="https://identity.example.test"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_oidc_configuration())

    assert exc_info.value.status_code == 503


def test_auditor_cannot_create_or_change_analyses():
    original_required = settings.auth_required
    original_email = settings.admin_email
    original_password = settings.admin_password
    original_role = settings.admin_role
    object.__setattr__(settings, "auth_required", True)
    object.__setattr__(settings, "admin_email", "auditor@example.test")
    object.__setattr__(settings, "admin_password", "correct horse battery staple")
    object.__setattr__(settings, "admin_role", "auditor")
    auth_client = TestClient(app)
    try:
        login_page = auth_client.get("/login")
        csrf = re.search(r'name="csrf" value="([^"]+)"', login_page.text).group(1)
        assert auth_client.post(
            "/login",
            data={"email": "auditor@example.test", "password": "correct horse battery staple", "csrf": csrf},
        ).status_code == 200
        assert auth_client.get("/api/audit").status_code == 200
        append_history(
            {"id": "auditor-view-record", "created_at": "2026-07-29T00:00:00+00:00", "medications": []},
            owner_id="auditor@example.test",
        )
        details = auth_client.get("/details/auditor-view-record")
        assert details.status_code == 200
        assert "Auditor access is read-only" in details.text
        assert 'data-review-status="confirmed"' not in details.text
        image_buffer = BytesIO()
        Image.new("RGB", (1, 1), "white").save(image_buffer, format="PNG")
        response = auth_client.post(
            "/api/analyze",
            data={"consent": "true"},
            files={"file": ("rx.png", image_buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 403
        assert response.json()["detail"] == "Reviewer role required."
        assert auth_client.patch("/api/analyses/unknown/review", json={"status": "confirmed"}).status_code == 403
    finally:
        object.__setattr__(settings, "auth_required", original_required)
        object.__setattr__(settings, "admin_email", original_email)
        object.__setattr__(settings, "admin_password", original_password)
        object.__setattr__(settings, "admin_role", original_role)
        save_history([], "auditor@example.test")


def test_analyze_preserves_parsed_contract_and_pipeline_metadata(monkeypatch):
    original_history = load_history()
    threaded_functions = []

    async def run_inline(func, *args, **kwargs):
        threaded_functions.append(func)
        return func(*args, **kwargs)

    def fake_ocr(_):
        return OCRResult(
            "Paracetamol 650 mg tab od 5 days",
            0.94,
            (OCRLine("Paracetamol 650 mg tab od 5 days", 0.94),),
            (),
        )

    def fake_structure(_):
        return {
            "patient_name": "A Patient",
            "doctor_name": "Dr. B",
            "date": "2026-07-10",
            "medications": [{"name": "Paracetamol", "requires_review": True}],
            "pipeline": {"used_provider": "fallback", "warnings": []},
        }

    monkeypatch.setattr("simpliscribe.web.asyncio.to_thread", run_inline)
    monkeypatch.setattr(
        "simpliscribe.web.extract_ocr_result",
        fake_ocr,
    )
    monkeypatch.setattr("simpliscribe.web.structure_medications", fake_structure)

    image_buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(image_buffer, format="PNG")
    image = image_buffer.getvalue()
    try:
        response = client.post("/api/analyze", data={"consent": "true"}, files={"file": ("rx.png", image, "image/png")})
        assert response.status_code == 200
        payload = response.json()
        assert payload["medications"] == [{"name": "Paracetamol", "requires_review": True}]
        assert payload["patient_name"] == "A Patient"
        assert payload["pipeline"]["ocr_confidence"] == 0.94
        assert payload["review_status"] == "needs_review"
        assert response.headers["cache-control"] == "no-store"
        assert threaded_functions == [fake_ocr, fake_structure]
    finally:
        save_history(original_history)


def test_analyze_rejects_spoofed_image_and_removes_upload():
    before = {path.name for path in settings.uploads_dir.iterdir()}

    response = client.post("/api/analyze", data={"consent": "true"}, files={"file": ("fake.png", b"not an image", "image/png")})

    assert response.status_code == 400
    assert "not a valid supported image" in response.json()["detail"]
    assert {path.name for path in settings.uploads_dir.iterdir()} == before


def test_analyze_requires_explicit_consent():
    response = client.post("/api/analyze", files={"file": ("rx.png", b"not an image", "image/png")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Explicit processing consent is required."


def test_storage_isolates_analysis_owners():
    owner_a = "test-owner-a"
    owner_b = "test-owner-b"
    record = {"id": "owner-isolation-record", "created_at": "2026-07-11T00:00:00+00:00", "medications": []}
    try:
        append_history(record, owner_id=owner_a)
        assert get_analysis_record(record["id"], owner_a) == record
        assert get_analysis_record(record["id"], owner_b) is None
    finally:
        save_history([], owner_a)


def test_storage_rejects_stale_analysis_update():
    record = {"id": "stale-update-record", "created_at": "2026-07-11T00:00:00+00:00", "medications": []}
    try:
        append_history(record)
        first_copy = get_analysis_record(record["id"])
        stale_copy = get_analysis_record(record["id"])
        updated = deepcopy(first_copy)
        updated["review_status"] = "confirmed"

        assert update_analysis_record(record["id"], "local", updated, expected_record=first_copy)
        stale_copy["review_status"] = "rejected"
        assert not update_analysis_record(record["id"], "local", stale_copy, expected_record=stale_copy)
    finally:
        save_history([])


def test_review_endpoint_updates_owned_analysis():
    original_history = load_history()
    record = {
        "id": "review-test-id",
        "created_at": "2026-07-11T00:00:00+00:00",
        "medications": [{"name": "Paracetmol", "type": "Tablet", "dosage": "650 mg", "frequency": "once daily", "duration": "5 days"}],
    }
    try:
        append_history(record)
        response = client.patch(
            "/api/analyses/review-test-id/review",
            json={"status": "corrected", "medications": [{"name": "Paracetamol"}]},
        )
        assert response.status_code == 200
        updated = get_analysis_record("review-test-id")
        assert updated["review_status"] == "corrected"
        assert updated["medications"][0]["name"] == "Paracetamol"
        assert updated["reviewed_by"] == "local"
        assert updated["review_versions"] == [
            {
                "version": 1,
                "recorded_at": updated["review_versions"][0]["recorded_at"],
                "status": "needs_review",
                "reviewed_at": None,
                "reviewed_by": None,
                "medications": [{"name": "Paracetmol", "type": "Tablet", "dosage": "650 mg", "frequency": "once daily", "duration": "5 days"}],
            }
        ]
        response = client.patch("/api/analyses/review-test-id/review", json={"status": "confirmed"})
        assert response.status_code == 200
        updated = get_analysis_record("review-test-id")
        assert updated["review_versions"][1]["status"] == "corrected"
        assert updated["review_versions"][1]["medications"][0]["name"] == "Paracetamol"
    finally:
        save_history(original_history)


def test_analyze_removes_upload_when_ocr_is_unusable(monkeypatch):
    before = {path.name for path in settings.uploads_dir.iterdir()}
    monkeypatch.setattr("simpliscribe.web.extract_ocr_result", lambda _: (_ for _ in ()).throw(ValueError("unusable")))
    image_buffer = BytesIO()
    Image.new("RGB", (2, 2), "white").save(image_buffer, format="PNG")

    response = client.post("/api/analyze", data={"consent": "true"}, files={"file": ("rx.png", image_buffer.getvalue(), "image/png")})

    assert response.status_code == 422
    assert response.json()["error_code"] == "UNUSABLE_PRESCRIPTION"
    assert {path.name for path in settings.uploads_dir.iterdir()} == before


def test_fallback_extract_handles_multiline_prescriptions():
    raw_text = "Paracetamol 650 tab od 5 days\nAmoxycillin 500 cap bd 5 days"
    medications = fallback_extract(raw_text)["medications"]

    assert len(medications) >= 2
    assert medications[0]["name"]
    assert medications[0]["frequency"] == "once daily"


def test_build_medication_record_normalizes_model_output_fields():
    record = build_medication_record(
        name="Paracetamol",
        category="General",
        medication_type="Tabular",
        dosage="650 mg daily for 5 days",
        frequency="od",
        duration="for 5 day",
        insight="Follow the prescription exactly as written.",
        entry=None,
    )

    assert record["type"] == "Tablet"
    assert record["dosage"] == "650 mg"
    assert record["frequency"] == "once daily"
    assert record["duration"] == "5 days"


def test_refine_model_medications_uses_ocr_heuristics_for_shorthand_fields():
    raw_text = "Amoxycillin 500 cap bd 5 days\nCetirizine 10 tab hs 3 days"
    model_medications = [
        {
            "name": "Amoxycillin",
            "category": "General",
            "type": "Tablet",
            "dosage": "500 mg",
            "frequency": "twice daily",
            "duration": "5 days",
            "insight": "Follow the prescription exactly as written.",
        },
        {
            "name": "Cetirizine",
            "category": "General",
            "type": "Tablet",
            "dosage": "10 mg",
            "frequency": "three times daily",
            "duration": "3 days",
            "insight": "Follow the prescription exactly as written.",
        },
    ]

    refined = refine_model_medications(raw_text, model_medications)

    assert refined[0]["type"] == "Capsule"
    assert refined[1]["frequency"] == "at bedtime"


def test_report_download_route_returns_pdf():
    original_history = load_history()
    record = {
        "id": "test-report-id",
        "filename": "sample-prescription.pdf",
        "created_at": "2026-03-11T12:00:00+00:00",
        "raw_text": "Paracetamol 650 tab od 5 days",
        "medications": [
            {
                "name": "Paracetamol",
                "category": "General",
                "type": "Tablet",
                "dosage": "650 mg",
                "frequency": "once daily",
                "duration": "5 days",
                "insight": "Use as directed.",
                "source": "OCR only",
                "source_datasets": [],
                "composition": "Paracetamol",
                "manufacturer": "",
                "pack_size": "",
                "therapeutic_class": "",
                "chemical_class": "",
                "action_class": "",
                "substitutes": [],
                "uses": [],
                "side_effects": [],
            }
        ],
    }

    try:
        append_history(record)
        response = client.get("/api/report/test-report-id")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")
    finally:
        save_history(original_history)


def test_report_download_pdf_contains_expected_content():
    original_history = load_history()
    record = {
        "id": "test-report-content-id",
        "filename": "sample prescription.pdf",
        "created_at": "2026-03-11T12:00:00+00:00",
        "patient_name": "Ananya Sharma",
        "doctor_name": "Dr. Meera Rao",
        "date": "2026-03-11",
        "review_status": "needs_review",
        "review_versions": [{"version": 1}],
        "pipeline": {"ocr_confidence": 0.87},
        "raw_text": "Paracetamol 650 tab od 5 days",
        "medications": [
            {
                "name": "Paracetamol",
                "category": "Analgesic",
                "type": "Tablet",
                "dosage": "650 mg",
                "frequency": "once daily",
                "duration": "5 days",
                "insight": "Use as directed.",
                "source": "OCR + dataset match",
                "source_datasets": ["India Medicines Dataset"],
                "composition": "Paracetamol",
                "manufacturer": "ABC Pharma",
                "pack_size": "15 tablets",
                "therapeutic_class": "Pain relief",
                "chemical_class": "Anilide",
                "action_class": "Analgesic",
                "substitutes": ["Dolo 650"],
                "uses": ["Fever"],
                "side_effects": ["Nausea"],
                "requires_review": True,
                "review_reasons": ["Medicine name must be confirmed."],
            }
        ],
    }

    try:
        append_history(record)
        response = client.get("/api/report/test-report-content-id")

        assert response.status_code == 200
        assert 'filename="sample_prescription_report.pdf"' in response.headers["content-disposition"]

        document = fitz.open(stream=response.content, filetype="pdf")
        extracted_text = "\n".join(page.get_text() for page in document)

        assert "Prescription Analysis Report" in extracted_text
        assert "Ananya Sharma" in extracted_text
        assert "Dr. Meera Rao" in extracted_text
        assert "87%" in extracted_text
        assert "Paracetamol" in extracted_text
        assert "ABC Pharma" in extracted_text
        assert "Use as directed." in extracted_text
        assert "Paracetamol 650 tab od 5 days" in extracted_text
        assert "Needs review" in extracted_text
        assert "PRIOR REVIEW STATES" in extracted_text.upper()
        assert "1" in extracted_text
        assert "Medicine name must be confirmed." in extracted_text
    finally:
        save_history(original_history)
