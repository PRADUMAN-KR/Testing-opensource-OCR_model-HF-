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

    APP_NAME: str = "PaddleOCR Pipeline API"
    DEBUG: bool = False

    # Single active OCR model backed by PaddleOCR PP-OCRv5.
    ENABLED_MODELS: List[str] = ["paddleocr_v5"]

    # Preset retained for compatibility; resolves to the single active OCR model.
    OCR_WITHOUT_LLM_CAPABILITIES: List[str] = ["paddleocr_v5"]

    PADDLE_USE_GPU: bool = False
    GPU_DEVICE_ID: int = 0
    PADDLE_MEM_FRACTION: float | None = None
    PADDLE_FLAGS_ALLOCATOR_STRATEGY: str | None = None
    PADDLE_FLAGS_FRACTION_OF_GPU_MEMORY_TO_USE: float | None = None
    PADDLE_EMPTY_CACHE_BETWEEN_PAGES: bool = False
    # Cap detection input side length to control VRAM usage.
    # (Used by PaddleOCR det_limit_side_len / limit_side_len if supported.)
    PADDLE_DET_LIMIT_SIDE_LEN: int | None = None
    PADDLE_TEXT_DET_THRESH: float = 0.22
    PADDLE_TEXT_DET_BOX_THRESH: float = 0.35
    PADDLE_TEXT_REC_SCORE_THRESH: float = 0.30

    # F2 pipeline debug visualisations
    PADDLE_DEBUG_OUTPUT_DIR: str = ""
    PADDLE_ARABIC_V3_FALLBACK: bool = False
    PADDLE_ARABIC_RUN_BOTH_ENGINES: bool = False
    PADDLE_MAX_ACCURACY: bool = True
    PADDLE_F2_FALLBACK_MIN_LINES: int = 4
    PADDLE_F2_FALLBACK_MIN_CHARS: int = 80
    PADDLE_F2_FALLBACK_MIN_AVG_CONF: float = 0.65
    PADDLE_F2_FALLBACK_MIN_ARABIC_RATIO: float = 0.60
    PADDLE_F2_FALLBACK_REPLACE_MARGIN: float = 0.03

    PADDLE_INPUT_ROI_WARP: bool = False
    PADDLE_ARABIC_AUTO_PAGE_CROP: bool = True
    PADDLE_ROI_MIN_AREA_RATIO: float = 0.15
    PADDLE_ROI_PAD_RATIO: float = 0.02

    PADDLE_ARABIC_NORMALIZE_ALEF: bool = False
    PADDLE_ARABIC_NORMALIZE_BIDI: bool = False
    PADDLE_ARABIC_FILTER_ISOLATED_LETTERS: bool = True
    PADDLE_ARABIC_MIXED_PAGE_RATIO_THRESHOLD: float = 0.35

    # PaddleOCR mixed precision and acceleration
    # Valid values: fp32, fp16, bf16
    PADDLE_PRECISION: str = "fp32"
    PADDLE_FP16: bool = False
    PADDLE_TENSORRT: bool = False

    MODEL_TIMEOUT: int = 60
    BENCHMARK_TIMEOUT: int = 300

    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp", "image/tiff", "application/pdf"]

    ENABLE_PROMETHEUS: bool = False

    CORS_ORIGINS: List[str] = ["*"]


settings = Settings()
