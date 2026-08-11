import base64
import hashlib
import hmac
import json
import secrets
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import HTTPException, Request

from .config import settings


def current_user(request: Request) -> dict[str, str] | None:
    user = request.session.get("user")
    return user if isinstance(user, dict) and user.get("id") else None


def owner_id(request: Request) -> str:
    user = current_user(request)
    if user:
        return str(user["id"])
    if settings.authentication_enabled:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return "local"


def require_edit_role(request: Request) -> dict[str, str] | None:
    if not settings.authentication_enabled:
        return None
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    if user.get("role") not in {"admin", "reviewer"}:
        raise HTTPException(status_code=403, detail="Reviewer role required.")
    return user


def authenticate(email: str, password: str) -> dict[str, str] | None:
    valid_email = hmac.compare_digest(email.strip().lower(), settings.admin_email.strip().lower())
    valid_password = bool(settings.admin_password) and hmac.compare_digest(password, settings.admin_password)
    if not (valid_email and valid_password):
        return None
    return {"id": settings.admin_email.strip().lower(), "email": settings.admin_email.strip().lower(), "role": settings.admin_role}


async def _oidc_configuration() -> dict[str, str]:
    if not settings.oidc_enabled:
        raise HTTPException(status_code=404, detail="Organization sign-in is not configured.")
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{settings.oidc_issuer}/.well-known/openid-configuration")
        response.raise_for_status()
    payload = response.json()
    required = ("authorization_endpoint", "token_endpoint", "userinfo_endpoint")
    if (
        not isinstance(payload, dict)
        or payload.get("issuer") != settings.oidc_issuer
        or any(not isinstance(payload.get(key), str) or urlparse(payload[key]).scheme != "https" for key in required)
    ):
        raise HTTPException(status_code=503, detail="Organization sign-in configuration is unavailable.")
    return {key: payload[key] for key in required}


async def oidc_authorization_url(request: Request) -> str:
    configuration = await _oidc_configuration()
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    request.session["oidc"] = {"state": state, "verifier": verifier}
    return f"{configuration['authorization_endpoint']}?{urlencode({
        'response_type': 'code',
        'client_id': settings.oidc_client_id,
        'redirect_uri': settings.oidc_redirect_uri,
        'scope': 'openid profile email',
        'state': state,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
    })}"


def oidc_user_from_claims(claims: dict[str, Any]) -> dict[str, str]:
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise HTTPException(status_code=401, detail="Organization sign-in did not provide a subject.")
    admins, reviewers = settings.oidc_subject_roles
    role = "admin" if subject in admins else "reviewer" if subject in reviewers else "auditor"
    email = claims.get("email")
    user_id = hashlib.sha256(f"{settings.oidc_issuer}:{subject}".encode()).hexdigest()
    return {"id": f"oidc:{user_id}", "email": email.strip().lower() if isinstance(email, str) and email.strip() else subject, "role": role}


def _id_token_audiences(token_payload: dict[str, Any]) -> list[str]:
    id_token = token_payload.get("id_token")
    if not isinstance(id_token, str) or "." not in id_token:
        return []
    try:
        segment = id_token.split(".")[1]
        segment += "=" * (-len(segment) % 4)
        claims = json.loads(base64.urlsafe_b64decode(segment))
    except Exception:
        return []
    if not isinstance(claims, dict):
        return []
    aud = claims.get("aud")
    if isinstance(aud, str):
        return [aud]
    if isinstance(aud, list):
        return [str(item) for item in aud if isinstance(item, str)]
    return []


async def authenticate_oidc_callback(request: Request, state: str, code: str) -> dict[str, str]:
    pending = request.session.pop("oidc", None)
    if not isinstance(pending, dict) or not state or not hmac.compare_digest(str(pending.get("state") or ""), state):
        raise HTTPException(status_code=400, detail="Invalid organization sign-in state.")
    if not code:
        raise HTTPException(status_code=400, detail="Missing organization sign-in code.")
    verifier = str(pending.get("verifier") or "")
    if not verifier:
        raise HTTPException(status_code=400, detail="Invalid organization sign-in session.")
    configuration = await _oidc_configuration()
    async with httpx.AsyncClient(timeout=10) as client:
        token_response = await client.post(
            configuration["token_endpoint"],
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.oidc_redirect_uri,
                "client_id": settings.oidc_client_id,
                "client_secret": settings.oidc_client_secret,
                "code_verifier": verifier,
            },
        )
        token_response.raise_for_status()
        token_payload = token_response.json()
        audiences = _id_token_audiences(token_payload)
        if audiences and settings.oidc_client_id not in audiences:
            raise HTTPException(status_code=401, detail="Organization sign-in token audience is invalid.")
        access_token = token_payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise HTTPException(status_code=401, detail="Organization sign-in did not return an access token.")
        userinfo_response = await client.get(configuration["userinfo_endpoint"], headers={"Authorization": f"Bearer {access_token}"})
        userinfo_response.raise_for_status()
    claims = userinfo_response.json()
    if not isinstance(claims, dict):
        raise HTTPException(status_code=401, detail="Organization sign-in claims are invalid.")
    return oidc_user_from_claims(claims)


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return str(token)


def verify_csrf(request: Request, supplied_token: str | None) -> None:
    expected = str(request.session.get("csrf_token") or "")
    if not expected or not supplied_token or not hmac.compare_digest(expected, supplied_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token.")


def public_user_context(request: Request) -> dict[str, Any]:
    user = current_user(request)
    return {"user": user, "csrf_token": csrf_token(request), "can_edit": not user or user.get("role") in {"admin", "reviewer"}}
