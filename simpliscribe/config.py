import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str = os.environ.get("APP_NAME", "SimpliScribe")
    app_env: str = os.environ.get("APP_ENV", "development")
    root_dir: Path = BASE_DIR
    templates_dir: Path = BASE_DIR / "templates"
    static_dir: Path = BASE_DIR / "static"
    uploads_dir: Path = BASE_DIR / "uploads"
    data_dir: Path = BASE_DIR / "data"
    history_file: Path = BASE_DIR / "data" / "analysis_history.json"
    database_url: str = os.environ.get("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'data' / 'simpliscribe.db').as_posix()}")
    session_secret: str = os.environ.get("SESSION_SECRET", "development-only-change-me")
    admin_email: str = os.environ.get("ADMIN_EMAIL", "admin@localhost")
    admin_password: str = os.environ.get("ADMIN_PASSWORD", "")
    admin_role: str = os.environ.get("ADMIN_ROLE", "admin").strip().lower()
    oidc_issuer: str = os.environ.get("OIDC_ISSUER", "").strip().rstrip("/")
    oidc_client_id: str = os.environ.get("OIDC_CLIENT_ID", "").strip()
    oidc_client_secret: str = os.environ.get("OIDC_CLIENT_SECRET", "")
    oidc_redirect_uri: str = os.environ.get("OIDC_REDIRECT_URI", "").strip()
    oidc_admin_subjects: str = os.environ.get("OIDC_ADMIN_SUBJECTS", "")
    oidc_reviewer_subjects: str = os.environ.get("OIDC_REVIEWER_SUBJECTS", "")
    auth_required: bool = os.environ.get("AUTH_REQUIRED", "").strip().lower() in {"1", "true", "yes", "on"}
    retention_days: int = int(os.environ.get("RETENTION_DAYS", "30"))
    session_max_age_seconds: int = int(os.environ.get("SESSION_MAX_AGE_SECONDS", "28800"))
    session_https_only: bool = os.environ.get("SESSION_HTTPS_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
    india_medicine_dataset: Path = BASE_DIR / "A_Z_medicines_dataset_of_India.csv"
    medicine_database_dataset: Path = BASE_DIR / "all_medicine databased.csv"
    max_upload_mb: int = int(os.environ.get("MAX_UPLOAD_MB", "10"))
    max_pdf_pages: int = int(os.environ.get("MAX_PDF_PAGES", "10"))
    max_image_pixels: int = int(os.environ.get("MAX_IMAGE_PIXELS", "40000000"))
    min_ocr_confidence: float = float(os.environ.get("MIN_OCR_CONFIDENCE", "0.80"))
    hf_token: str = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    hf_model: str = os.environ.get("HF_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    inference_provider: str = os.environ.get("INFERENCE_PROVIDER", "huggingface")
    model_api_url: str = os.environ.get("MODEL_API_URL", "")
    model_api_key: str = os.environ.get("MODEL_API_KEY", "")
    request_timeout_seconds: float = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "60"))
    ocr_language: str = os.environ.get("OCR_LANGUAGE", "en")
    ocr_use_gpu: bool = os.environ.get("OCR_USE_GPU", "false").strip().lower() in {"1", "true", "yes", "on"}
    local_model_id: str = os.environ.get("LOCAL_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    local_model_device: str = os.environ.get("LOCAL_MODEL_DEVICE", "auto")
    local_model_temperature: float = float(os.environ.get("LOCAL_MODEL_TEMPERATURE", "0.1"))
    local_model_max_new_tokens: int = int(os.environ.get("LOCAL_MODEL_MAX_NEW_TOKENS", "256"))
    local_model_trust_remote_code: bool = os.environ.get("LOCAL_MODEL_TRUST_REMOTE_CODE", "false").strip().lower() in {"1", "true", "yes", "on"}
    # Alternative medicine reference candidates. Disabled by default (fail-closed):
    # enabling this sends the canonical medicine name to the configured model and/or
    # a DuckDuckGo web search when the local datasets provide no substitutes.
    alternatives_enabled: bool = os.environ.get("ALTERNATIVES_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    alternatives_provider: str = os.environ.get("ALTERNATIVES_PROVIDER", "auto").strip().lower()
    alternatives_timeout_seconds: float = float(os.environ.get("ALTERNATIVES_TIMEOUT_SECONDS", "15"))
    alternatives_cache_ttl_seconds: int = int(os.environ.get("ALTERNATIVES_CACHE_TTL_SECONDS", "86400"))
    alternatives_max_candidates: int = int(os.environ.get("ALTERNATIVES_MAX_CANDIDATES", "5"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def production(self) -> bool:
        return self.app_env.strip().lower() == "production"

    @property
    def authentication_enabled(self) -> bool:
        return self.production or self.auth_required or self.oidc_enabled

    @property
    def oidc_enabled(self) -> bool:
        return bool(self.oidc_issuer and self.oidc_client_id and self.oidc_client_secret and self.oidc_redirect_uri)

    @property
    def oidc_subject_roles(self) -> tuple[set[str], set[str]]:
        parse = lambda value: {item.strip() for item in value.split(",") if item.strip()}
        return parse(self.oidc_admin_subjects), parse(self.oidc_reviewer_subjects)

    @property
    def secure_transport(self) -> bool:
        return self.production or self.session_https_only

    @property
    def alternatives_provider_chain(self) -> list[str]:
        provider = self.alternatives_provider
        if provider in {"model", "web", "duckduckgo"}:
            return ["web" if provider == "duckduckgo" else provider]
        return ["model", "web"]

    def validate_runtime(self) -> None:
        errors: list[str] = []
        if self.production:
            if self.session_secret == "development-only-change-me" or len(self.session_secret) < 32:
                errors.append("SESSION_SECRET must be a unique value of at least 32 characters")
            if not self.oidc_enabled:
                if not self.admin_password:
                    errors.append("ADMIN_PASSWORD is required")
                if not self.admin_email.strip():
                    errors.append("ADMIN_EMAIL is required")
            if not self.database_url.startswith("postgresql"):
                errors.append("DATABASE_URL must use PostgreSQL in production")
        if self.retention_days < 1:
            errors.append("RETENTION_DAYS must be at least 1")
        if self.session_max_age_seconds < 1:
            errors.append("SESSION_MAX_AGE_SECONDS must be at least 1")
        if self.admin_role not in {"admin", "reviewer", "auditor"}:
            errors.append("ADMIN_ROLE must be admin, reviewer, or auditor")
        if self.alternatives_provider not in {"auto", "model", "web", "duckduckgo"}:
            errors.append("ALTERNATIVES_PROVIDER must be auto, model, web, or duckduckgo")
        if self.alternatives_max_candidates < 1 or self.alternatives_max_candidates > 10:
            errors.append("ALTERNATIVES_MAX_CANDIDATES must be between 1 and 10")
        if self.alternatives_timeout_seconds < 1:
            errors.append("ALTERNATIVES_TIMEOUT_SECONDS must be at least 1")
        oidc_values = (self.oidc_issuer, self.oidc_client_id, self.oidc_client_secret, self.oidc_redirect_uri)
        if any(oidc_values) and not all(oidc_values):
            errors.append("OIDC_ISSUER, OIDC_CLIENT_ID, OIDC_CLIENT_SECRET, and OIDC_REDIRECT_URI must be configured together")
        if self.oidc_enabled and self.production:
            if not self.oidc_issuer.startswith("https://"):
                errors.append("OIDC_ISSUER must use HTTPS in production")
            if not self.oidc_redirect_uri.startswith("https://"):
                errors.append("OIDC_REDIRECT_URI must use HTTPS in production")
        if errors:
            raise RuntimeError("Unsafe runtime configuration: " + "; ".join(errors))


settings = Settings()
