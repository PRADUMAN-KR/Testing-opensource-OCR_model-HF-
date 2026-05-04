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

    # Active OCR models loaded at startup.
    ENABLED_MODELS: List[str] = ["qari_ocr_vl_2b", "paddleocr_vl"]

    GPU_DEVICE_ID: int = 0

    # Preset retained for compatibility with /ocr/run/options.
    OCR_WITHOUT_LLM_CAPABILITIES: List[str] = ["paddleocr_vl"]

    # Qari-OCR VL 2B Arabic OCR settings.
    QARI_MODEL_ID: str = "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct"
    QARI_MAX_NEW_TOKENS: int = 2000
    QARI_TORCH_DTYPE: str = "auto"
    QARI_DEVICE_MAP: str = "auto"
    QARI_PROMPT: str = (
        "Below is the image of one page of a document, as well as some raw textual "
        "content that was previously extracted for it. Just return the plain text "
        "representation of this document as if you were reading it naturally. "
        "Do not hallucinate."
    )

    # Official PaddleOCR-VL document parsing settings.
    PADDLEOCR_VL_DEVICE: str = "gpu:0"
    PADDLEOCR_VL_PIPELINE_VERSION: str = "v1"
    PADDLEOCR_VL_USE_LAYOUT_DETECTION: bool = True
    PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY: bool = False
    PADDLEOCR_VL_USE_DOC_UNWARPING: bool = False
    PADDLEOCR_VL_USE_CHART_RECOGNITION: bool = False
    PADDLEOCR_VL_USE_SEAL_RECOGNITION: bool = False
    PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK: bool = False
    PADDLEOCR_VL_FORMAT_BLOCK_CONTENT: bool = True
    PADDLEOCR_VL_MERGE_LAYOUT_BLOCKS: bool = True

    MODEL_TIMEOUT: int = 60
    BENCHMARK_TIMEOUT: int = 300

    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp", "image/tiff", "application/pdf"]

    ENABLE_PROMETHEUS: bool = False

    CORS_ORIGINS: List[str] = ["*"]


settings = Settings()
