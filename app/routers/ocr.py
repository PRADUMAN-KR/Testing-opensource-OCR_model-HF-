"""
/ocr endpoints — run loaded OCR models on an uploaded image.
"""

import time
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends

from app.schemas import (
    OCRResponse,
    ModelResult,
    WordDetail,
    Language,
)
from app.models.base import SupportedLanguage
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
PRIMARY_MODEL_NAME = "paddleocr_v5"


def get_registry(request: Request):
    return request.app.state.model_registry


def _validate_image(file: UploadFile):
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {settings.ALLOWED_IMAGE_TYPES}",
        )


@router.post("/run", response_model=OCRResponse, summary="Run OCR on an image")
async def run_ocr(
    request: Request,
    file: UploadFile = File(..., description="Image file to process"),
    language: Language = Form(
        ...,
        description=(
            "Target language. Use 'all' to run best-effort multilingual OCR and extract all detectable text content."
        ),
    ),
    registry=Depends(get_registry),
):
    """
    Upload an image and run the configured PaddleOCR model on it.
    Returns extracted text, word-level details, confidence, and timing.
    """
    _validate_image(file)

    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

    lang_enum = SupportedLanguage(language.value)
    model = registry.get(PRIMARY_MODEL_NAME)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Configured OCR model '{PRIMARY_MODEL_NAME}' is not loaded at startup.",
        )
    if not model.supports_language(lang_enum):
        raise HTTPException(
            status_code=404,
            detail=f"Configured OCR model '{PRIMARY_MODEL_NAME}' does not support language '{language.value}'.",
        )
    
    start = time.perf_counter()
    logger.info(f"[OCR] Running {PRIMARY_MODEL_NAME} | lang={language.value} | file={file.filename}")
    result = await model.run(image_bytes, lang_enum)
    results = [ModelResult(
        model_name=result.model_name,
        language=result.language,
        raw_text=result.raw_text,
        words=[WordDetail(text=w.text, confidence=w.confidence, bbox=w.bbox) for w in result.words],
        inference_time_ms=result.inference_time_ms,
        avg_confidence=result.avg_confidence,
        error=result.error,
        metadata=result.metadata,
    )]

    total_ms = round((time.perf_counter() - start) * 1000, 2)

    return OCRResponse(
        filename=file.filename,
        language=language.value,
        results=results,
        models_run=len(results),
        total_time_ms=total_ms,
    )
