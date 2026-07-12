import hmac
import secrets
from typing import Any

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


def authenticate(email: str, password: str) -> dict[str, str] | None:
    valid_email = hmac.compare_digest(email.strip().lower(), settings.admin_email.strip().lower())
    valid_password = bool(settings.admin_password) and hmac.compare_digest(password, settings.admin_password)
    if not (valid_email and valid_password):
        return None
    return {"id": settings.admin_email.strip().lower(), "email": settings.admin_email.strip().lower(), "role": "admin"}


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
    return {"user": current_user(request), "csrf_token": csrf_token(request)}
