import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    app_name: str = os.environ.get("APP_NAME", "SimpliScribe")
    app_env: str = os.environ.get("APP_ENV", "development")
    templates_dir: Path = BASE_DIR / "templates"
    static_dir: Path = BASE_DIR / "static"
    uploads_dir: Path = BASE_DIR / "uploads"
    data_dir: Path = BASE_DIR / "data"
    history_file: Path = BASE_DIR / "data" / "analysis_history.json"
    max_upload_mb: int = int(os.environ.get("MAX_UPLOAD_MB", "10"))
    hf_token: str = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    hf_model: str = os.environ.get("HF_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    inference_provider: str = os.environ.get("INFERENCE_PROVIDER", "huggingface")
    model_api_url: str = os.environ.get("MODEL_API_URL", "")
    model_api_key: str = os.environ.get("MODEL_API_KEY", "")
    request_timeout_seconds: float = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "60"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()