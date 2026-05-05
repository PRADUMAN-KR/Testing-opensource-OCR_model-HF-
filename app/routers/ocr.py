"""
Endpoints for PaddleOCR-VL worker-based processing.
"""

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.core.config import settings
from app.core.model_registry import PADDLEOCR_VL_MODEL_NAME
from app.schemas import (
    HealthResponse,
    OCRTaskStatusResponse,
    OCRTaskSubmitResponse,
)

router = APIRouter()


def get_registry(request: Request):
    return request.app.state.model_registry


def get_task_manager(request: Request):
    return request.app.state.ocr_task_manager


def get_task_store(request: Request):
    return request.app.state.ocr_task_store


def _validate_image(file: UploadFile) -> None:
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {settings.ALLOWED_IMAGE_TYPES}",
        )


async def _read_validated_file(file: UploadFile) -> bytes:
    _validate_image(file)
    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit")
    return image_bytes


def _get_paddleocr_vl_model(registry):
    selected_model = registry.get(PADDLEOCR_VL_MODEL_NAME)
    if selected_model is not None:
        return selected_model

    failure_reason = getattr(registry, "failed_models", {}).get(PADDLEOCR_VL_MODEL_NAME)
    raise HTTPException(
        status_code=503,
        detail=(
            f"Configured OCR model '{PADDLEOCR_VL_MODEL_NAME}' is not loaded at startup."
            + (f" Startup error: {failure_reason}" if failure_reason else "")
        ),
    )


@router.post("/submit", response_model=OCRTaskSubmitResponse, summary="Submit OCR work to the PaddleOCR-VL worker")
async def submit_ocr_task(
    file: UploadFile = File(..., description="Image or PDF file to process"),
    task_manager=Depends(get_task_manager),
    registry=Depends(get_registry),
):
    """
    Upload an image/PDF, enqueue PaddleOCR-VL processing, and return a task id.
    """
    _get_paddleocr_vl_model(registry)
    image_bytes = await _read_validated_file(file)
    task = await task_manager.submit(image_bytes, file.filename)
    return OCRTaskSubmitResponse(**task)


@router.get(
    "/status/{task_id}",
    response_model=OCRTaskStatusResponse,
    summary="Get OCR task status and result",
)
async def get_ocr_task_status(task_id: str, task_manager=Depends(get_task_manager)):
    task = await task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown OCR task_id '{task_id}'.")
    return OCRTaskStatusResponse(**task)


@router.get("/health", response_model=HealthResponse, summary="Check pipeline health")
async def health_check(
    registry=Depends(get_registry),
    task_manager=Depends(get_task_manager),
    task_store=Depends(get_task_store),
):
    database = await asyncio.to_thread(task_store.health)
    model_loaded = registry.get(PADDLEOCR_VL_MODEL_NAME) is not None
    worker_running = task_manager.is_worker_running()
    is_healthy = model_loaded and worker_running and bool(database.get("ok"))

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        api="ok",
        model_loaded=model_loaded,
        loaded_models=sorted(registry.all().keys()),
        failed_models=dict(getattr(registry, "failed_models", {})),
        worker_running=worker_running,
        queue_size=task_manager.queue_size(),
        database=database,
    )
