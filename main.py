"""
OCR pipeline — FastAPI service for document OCR (Arabic, Hindi, Punjabi, English, multilingual).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.routers import ocr
# from app.routers import benchmark, health
from app.core.config import settings
from app.core.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up model registry. Shutdown: release GPU memory."""
    logger.info("Initializing OCR model registry...")
    registry = ModelRegistry()
    await registry.initialize(settings.ENABLED_MODELS)
    app.state.model_registry = registry
    logger.info(f"Loaded models: {list(registry.loaded_models.keys())}")
    yield
    logger.info("Shutting down — releasing model resources...")
    await registry.shutdown()


app = FastAPI(
    title=settings.APP_NAME,
    description="OCR API backed by selectable OCR engines, including PaddleOCR and Qari-OCR VL.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])
# app.include_router(health.router, prefix="/health", tags=["Health"])
# app.include_router(benchmark.router, prefix="/benchmark", tags=["Benchmark"])
