"""Regression tests for security/production hardening fixes."""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from simpliscribe import main as main_module
from simpliscribe.alternatives import _cache, _cache_set
from simpliscribe.config import settings
from simpliscribe.local_model_server import app as model_server_app
from simpliscribe.security import authenticate_oidc_callback


def test_consume_bucket_enforces_limit_and_evicts_stale_keys():
    now = 1000.0
    buckets: dict[str, object] = {}
    assert main_module._consume_bucket(buckets, "a", now, 60, 2)
    assert main_module._consume_bucket(buckets, "a", now, 60, 2)
    assert not main_module._consume_bucket(buckets, "a", now, 60, 2)
    assert main_module._consume_bucket(buckets, "a", now + 61, 60, 2)


def test_consume_bucket_bounds_total_keys():
    buckets: dict[str, object] = {}
    main_module._MAX_BUCKETS = 5
    try:
        for index in range(10):
            assert main_module._consume_bucket(buckets, f"key-{index}", 1000.0, 60, 2)
        assert len(buckets) <= 5
    finally:
        main_module._MAX_BUCKETS = 4096


def test_analyze_requires_csrf_even_without_authentication():
    client = TestClient(main_module.app)
    response = client.post("/api/analyze", data={"consent": "true"}, files={"file": ("rx.png", b"x", "image/png")})
    assert response.status_code == 403


def test_login_rate_limited_after_threshold():
    client = TestClient(main_module.app)
    main_module._login_times.clear()
    main_module._request_times.clear()
    login_page = client.get("/login")
    match = __import__("re").search(r'name="csrf" value="([^"]+)"', login_page.text)
    csrf = match.group(1) if match else ""
    statuses = []
    for _ in range(21):
        response = client.post("/login", data={"email": "a@b.c", "password": "wrong", "csrf": csrf})
        statuses.append(response.status_code)
    assert statuses[-1] == 429
    assert statuses.count(401) >= 19


def test_alternatives_cache_evicts_when_full(monkeypatch):
    monkeypatch.setattr("simpliscribe.alternatives._CACHE_MAX_ENTRIES", 3)
    _cache.clear()
    try:
        for index in range(10):
            _cache_set(f"med-{index}", [{"name": "X", "source": "web", "provider": "ddg", "url": ""}])
        assert len(_cache) <= 3
    finally:
        _cache.clear()


def test_model_server_extract_requires_key_and_caps_input(monkeypatch):
    monkeypatch.setattr(
        "simpliscribe.local_model_server.settings",
        SimpleNamespace(model_server_api_key="secret-key", model_server_max_input_chars=1000),
    )
    client = TestClient(model_server_app)
    assert client.post("/extract", json={"input": "text"}).status_code == 401
    assert client.post("/extract", json={"input": "text"}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.post(
        "/extract",
        json={"input": "x" * 2000},
        headers={"Authorization": "Bearer secret-key"},
    )
    assert response.status_code == 413


def test_oidc_callback_rejects_missing_pkce_verifier():
    request = SimpleNamespace(session={"oidc": {"state": "expected-state"}})
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(authenticate_oidc_callback(request, "expected-state", "provider-code"))
    assert exc_info.value.status_code == 400


def _jwt(payload: dict) -> str:
    def segment(value: dict) -> str:
        raw = json.dumps(value).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{segment({'alg': 'HS256'})}.{segment(payload)}.signature"


def test_oidc_callback_rejects_id_token_with_wrong_audience(monkeypatch):
    original = (
        settings.oidc_issuer,
        settings.oidc_client_id,
        settings.oidc_client_secret,
        settings.oidc_redirect_uri,
    )
    object.__setattr__(settings, "oidc_issuer", "https://identity.example.test")
    object.__setattr__(settings, "oidc_client_id", "simpliscribe")
    object.__setattr__(settings, "oidc_client_secret", "secret")
    object.__setattr__(settings, "oidc_redirect_uri", "https://app.example.test/auth/callback")

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url, headers=None):
            return FakeResponse({
                "issuer": "https://identity.example.test",
                "authorization_endpoint": "https://identity.example.test/authorize",
                "token_endpoint": "https://identity.example.test/token",
                "userinfo_endpoint": "https://identity.example.test/userinfo",
            })

        async def post(self, url, data):
            return FakeResponse({"access_token": "token", "id_token": _jwt({"aud": "another-client"})})

    monkeypatch.setattr("simpliscribe.security.httpx.AsyncClient", FakeClient)
    request = SimpleNamespace(session={"oidc": {"state": "expected-state", "verifier": "verifier-value"}})
    try:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(authenticate_oidc_callback(request, "expected-state", "provider-code"))
        assert exc_info.value.status_code == 401
    finally:
        object.__setattr__(settings, "oidc_issuer", original[0])
        object.__setattr__(settings, "oidc_client_id", original[1])
        object.__setattr__(settings, "oidc_client_secret", original[2])
        object.__setattr__(settings, "oidc_redirect_uri", original[3])
