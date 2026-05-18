from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

# Resolve .env relative to this file's location so it is found regardless
# of the working directory uvicorn is launched from.
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    APP_NAME: str = "OCR Pipeline API"
    DEBUG: bool = False
    OCR_TASK_DB_PATH: str = "data/ocr_tasks.sqlite3"

    # Active OCR models loaded at startup.
    ENABLED_MODELS: List[str] = ["qwen_ocr"]

    GPU_DEVICE_ID: int = 0

    # Official Qwen image-text-to-text OCR settings.
    QWEN_OCR_MODEL_ID: str = "Qwen/Qwen3.6-27B"
    QWEN_OCR_DEVICE_MAP: str = "auto"
    QWEN_OCR_TORCH_DTYPE: str = "auto"
    QWEN_OCR_MAX_NEW_TOKENS: int = 4096
    QWEN_OCR_PDF_DPI: int = 200
    QWEN_OCR_MAX_PDF_PAGES: int = 20
    QWEN_OCR_PROMPT: str = (
        "Extract all readable text from this document image. Preserve reading order. "
        "Use Markdown for tables. Return only the extracted text."
    )

    MODEL_TIMEOUT: int = 60
    BENCHMARK_TIMEOUT: int = 300

    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp", "image/tiff", "application/pdf"]

    ENABLE_PROMETHEUS: bool = False

    CORS_ORIGINS: List[str] = ["*"]


settings = Settings()
