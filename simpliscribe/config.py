import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


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
    india_medicine_dataset: Path = BASE_DIR / "A_Z_medicines_dataset_of_India.csv"
    medicine_database_dataset: Path = BASE_DIR / "all_medicine databased.csv"
    max_upload_mb: int = int(os.environ.get("MAX_UPLOAD_MB", "10"))
    hf_token: str = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    hf_model: str = os.environ.get("HF_CHAT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    inference_provider: str = os.environ.get("INFERENCE_PROVIDER", "huggingface")
    model_api_url: str = os.environ.get("MODEL_API_URL", "")
    model_api_key: str = os.environ.get("MODEL_API_KEY", "")
    request_timeout_seconds: float = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "300"))
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


settings = Settings()