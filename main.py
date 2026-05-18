"""
OCR pipeline — FastAPI service for worker-based document parsing.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging

from app.routers import ocr
from app.core.config import settings
from app.core.model_registry import ModelRegistry
from app.core.ocr_tasks import OCRTaskManager
from app.core.task_store import OCRTaskStore

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
    task_store = OCRTaskStore(settings.OCR_TASK_DB_PATH)
    await asyncio.to_thread(task_store.initialize)
    app.state.model_registry = registry
    app.state.ocr_task_store = task_store
    app.state.ocr_task_manager = OCRTaskManager(registry, task_store)
    app.state.ocr_task_manager.start()
    logger.info(f"Loaded models: {list(registry.loaded_models.keys())}")
    yield
    logger.info("Stopping OCR task worker...")
    await app.state.ocr_task_manager.stop()
    logger.info("Shutting down — releasing model resources...")
    await registry.shutdown()


app = FastAPI(
    title=settings.APP_NAME,
    description="Worker-based OCR API",
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

app.include_router(ocr.router, tags=["OCR"])
