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

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def production(self) -> bool:
        return self.app_env.strip().lower() == "production"

    @property
    def authentication_enabled(self) -> bool:
        return self.production or self.auth_required

    @property
    def secure_transport(self) -> bool:
        return self.production or self.session_https_only

    def validate_runtime(self) -> None:
        errors: list[str] = []
        if self.production:
            if self.session_secret == "development-only-change-me" or len(self.session_secret) < 32:
                errors.append("SESSION_SECRET must be a unique value of at least 32 characters")
            if not self.admin_password:
                errors.append("ADMIN_PASSWORD is required")
            if not self.admin_email.strip():
                errors.append("ADMIN_EMAIL is required")
            if self.database_url.startswith("sqlite"):
                errors.append("DATABASE_URL must use a production database such as PostgreSQL")
        if self.retention_days < 1:
            errors.append("RETENTION_DAYS must be at least 1")
        if self.session_max_age_seconds < 1:
            errors.append("SESSION_MAX_AGE_SECONDS must be at least 1")
        if errors:
            raise RuntimeError("Unsafe runtime configuration: " + "; ".join(errors))


settings = Settings()
