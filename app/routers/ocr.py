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
    PageDetail,
    LineDetail,
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


def _line_from_metadata(item: dict) -> LineDetail:
    return LineDetail(
        page_index=int(item.get("page_index", 0)),
        line_index=int(item.get("line_index", 0)),
        text=str(item.get("text", "")),
        raw_text=str(item.get("raw_text", item.get("text", ""))),
        confidence=float(item.get("confidence", 0.0)),
        bbox=item.get("bbox"),
        bbox_source=item.get("bbox_source", {}) or {},
        bbox_valid=bool(item.get("bbox_valid", True)),
        status=item.get("status"),
        noise_score=item.get("noise_score"),
        filter_reason=item.get("filter_reason"),
        exclude_reason=item.get("exclude_reason"),
    )


def _pages_from_metadata(metadata: dict) -> list[PageDetail]:
    pages: list[PageDetail] = []
    for page in metadata.get("pages", []) or []:
        lines = [_line_from_metadata(line) for line in page.get("lines", []) or []]
        accepted_lines = [
            _line_from_metadata(line)
            for line in page.get("accepted_lines", []) or []
        ]
        review_lines = [
            _line_from_metadata(line)
            for line in page.get("review_lines", []) or []
        ]
        flagged_line_items = page.get("flagged_lines", page.get("filtered_lines", [])) or []
        flagged_lines = [
            _line_from_metadata(line)
            for line in flagged_line_items
        ]
        excluded_noise_lines = [
            _line_from_metadata(line)
            for line in page.get("excluded_noise_lines", []) or []
        ]
        excluded_lines = [
            _line_from_metadata(line)
            for line in page.get("excluded_lines", []) or []
        ]
        pages.append(
            PageDetail(
                page_index=int(page.get("page_index", 0)),
                text=str(page.get("text", "")),
                raw_text=str(page.get("raw_text", page.get("text", ""))),
                lines=lines,
                accepted_lines=accepted_lines,
                review_lines=review_lines,
                flagged_lines=flagged_lines,
                filtered_lines=flagged_lines,
                excluded_noise_lines=excluded_noise_lines,
                excluded_lines=excluded_lines,
                per_line_confidence=page.get("per_line_confidence", []) or [],
                per_line_noise_score=page.get("per_line_noise_score", []) or [],
                layout_mode=str(page.get("layout_mode", "text")),
                tables=page.get("tables", []) or [],
                totals=page.get("totals", {}) or {},
                avg_confidence=float(page.get("avg_confidence", 0.0)),
            )
        )
    return pages


def _flatten_page_lines(pages: list[PageDetail], attr: str) -> list[LineDetail]:
    out: list[LineDetail] = []
    for page in pages:
        out.extend(getattr(page, attr))
    return out


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
    pages = _pages_from_metadata(result.metadata)
    accepted_lines = _flatten_page_lines(pages, "accepted_lines")
    review_lines = _flatten_page_lines(pages, "review_lines")
    flagged_lines = _flatten_page_lines(pages, "flagged_lines")
    excluded_noise_lines = _flatten_page_lines(pages, "excluded_noise_lines")
    excluded_lines = _flatten_page_lines(pages, "excluded_lines")
    quality = result.metadata.get("quality", {})
    response_raw_text = result.metadata.get("raw_text_before_correction", result.raw_text)
    final_text = result.metadata.get("final_text", result.raw_text)
    selected_variant = result.metadata.get("selected_variant")
    confidence_score = float(result.metadata.get("confidence_score", result.avg_confidence))
    debug_lines = result.metadata.get("debug_lines", []) or []
    debug_blocks = result.metadata.get("debug_blocks", []) or []
    layout_mode = str(result.metadata.get("layout_mode", "text"))
    tables = result.metadata.get("tables", []) or []
    totals = result.metadata.get("totals", {}) or {}
    per_line_confidence = result.metadata.get("per_line_confidence", []) or []
    per_line_noise_score = result.metadata.get("per_line_noise_score", []) or []
    rejected_bbox_lines = result.metadata.get("rejected_bbox_lines", []) or []
    corrected_bbox_lines = result.metadata.get("corrected_bbox_lines", []) or []
    results = [ModelResult(
        model_name=result.model_name,
        language=result.language,
        final_text=final_text,
        text=result.raw_text,
        raw_text=response_raw_text,
        selected_variant=selected_variant,
        confidence_score=confidence_score,
        debug_lines=debug_lines,
        debug_blocks=debug_blocks,
        layout_mode=layout_mode,
        tables=tables,
        totals=totals,
        rejected_bbox_lines=rejected_bbox_lines,
        corrected_bbox_lines=corrected_bbox_lines,
        words=[WordDetail(text=w.text, confidence=w.confidence, bbox=w.bbox) for w in result.words],
        pages=pages,
        accepted_lines=accepted_lines,
        review_lines=review_lines,
        flagged_lines=flagged_lines,
        excluded_noise_lines=excluded_noise_lines,
        excluded_lines=excluded_lines,
        per_line_confidence=per_line_confidence,
        per_line_noise_score=per_line_noise_score,
        inference_time_ms=result.inference_time_ms,
        avg_confidence=result.avg_confidence,
        error=result.error,
        quality=quality,
        metadata=result.metadata,
    )]

    total_ms = round((time.perf_counter() - start) * 1000, 2)

    return OCRResponse(
        filename=file.filename,
        language=language.value,
        final_text=final_text,
        text=result.raw_text,
        raw_text=response_raw_text,
        selected_variant=selected_variant,
        confidence_score=confidence_score,
        debug_lines=debug_lines,
        debug_blocks=debug_blocks,
        layout_mode=layout_mode,
        tables=tables,
        totals=totals,
        rejected_bbox_lines=rejected_bbox_lines,
        corrected_bbox_lines=corrected_bbox_lines,
        pages=pages,
        accepted_lines=accepted_lines,
        review_lines=review_lines,
        flagged_lines=flagged_lines,
        excluded_noise_lines=excluded_noise_lines,
        excluded_lines=excluded_lines,
        per_line_confidence=per_line_confidence,
        per_line_noise_score=per_line_noise_score,
        avg_confidence=result.avg_confidence,
        quality=quality,
        results=results,
        models_run=len(results),
        total_time_ms=total_ms,
    )
