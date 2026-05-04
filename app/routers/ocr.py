"""
/ocr endpoints — run loaded OCR models on an uploaded image.
"""

import time
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, Response

from app.schemas import (
    OCRResponse,
    ModelResult,
    WordDetail,
    PageDetail,
    LineDetail,
    Language,
    ModelInfo,
    OCRRunOptionsResponse,
)
from app.models.base import SupportedLanguage
from app.core.config import settings
from app.core.model_registry import AVAILABLE_MODEL_NAMES
from app.core.model_selection import preset_model_groups

logger = logging.getLogger(__name__)
router = APIRouter()
PRIMARY_MODEL_NAME = "paddleocr_vl"
QARI_MODEL_NAME = "qari_ocr_vl_2b"
PADDLEOCR_VL_MODEL_NAME = "paddleocr_vl"
LANGUAGE_OPTIONAL_MODEL_NAMES = {QARI_MODEL_NAME, PADDLEOCR_VL_MODEL_NAME}


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
    )


def _pages_from_metadata(metadata: dict) -> list[PageDetail]:
    pages: list[PageDetail] = []
    for page in metadata.get("pages", []) or []:
        lines = [_line_from_metadata(line) for line in page.get("lines", []) or []]
        accepted_lines = [
            _line_from_metadata(line)
            for line in page.get("accepted_lines", []) or []
        ]
        pages.append(
            PageDetail(
                page_index=int(page.get("page_index", 0)),
                text=str(page.get("text", "")),
                raw_text=str(page.get("raw_text", page.get("text", ""))),
                lines=lines,
                accepted_lines=accepted_lines,
                per_line_confidence=page.get("per_line_confidence", []) or [],
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


def _model_info(name: str, model, loaded: bool) -> ModelInfo:
    return ModelInfo(
        name=name,
        tier=int(getattr(model, "tier", 0)) if model is not None else 0,
        supported_languages=[lang.value for lang in getattr(model, "supported_languages", [])],
        loaded=loaded,
    )


@router.get("/run/options", response_model=OCRRunOptionsResponse, summary="List OCR run options")
async def run_options(registry=Depends(get_registry)):
    loaded_models = [
        _model_info(name, model, True)
        for name, model in sorted(registry.all().items())
    ]
    return OCRRunOptionsResponse(
        loaded_models=loaded_models,
        available_model_names=AVAILABLE_MODEL_NAMES,
        available_languages=list(Language),
        preset_model_groups=preset_model_groups(),
        language_required_by_model={
            QARI_MODEL_NAME: False,
            PADDLEOCR_VL_MODEL_NAME: False,
        },
    )


async def _run_qari_raw_text(
    selected_model,
    image_bytes: bytes,
    filename: str | None,
):
    logger.info(f"[OCR] Running {QARI_MODEL_NAME} | file={filename}")
    try:
        output_text = await selected_model.run_raw(image_bytes)
    except Exception as exc:
        logger.exception("[OCR] Qari raw inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=output_text, media_type="text/plain; charset=utf-8")


async def _run_paddleocr_vl_raw_json(
    selected_model,
    image_bytes: bytes,
    filename: str | None,
):
    logger.info(f"[OCR] Running {PADDLEOCR_VL_MODEL_NAME} raw official response | file={filename}")
    try:
        output_json = await selected_model.run_raw_json(image_bytes)
    except Exception as exc:
        logger.exception("[OCR] PaddleOCR-VL raw inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(content=output_json)


@router.post("/run/qari", summary="Run Qari OCR on an image")
async def run_qari_ocr(
    request: Request,
    file: UploadFile = File(..., description="Image file to process"),
    registry=Depends(get_registry),
):
    """
    Upload an image and run Qari OCR with no language parameter.
    """
    _validate_image(file)

    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

    selected_model = registry.get(QARI_MODEL_NAME)
    if selected_model is None:
        failure_reason = getattr(registry, "failed_models", {}).get(QARI_MODEL_NAME)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Configured OCR model '{QARI_MODEL_NAME}' is not loaded at startup."
                + (f" Startup error: {failure_reason}" if failure_reason else "")
            ),
        )

    return await _run_qari_raw_text(selected_model, image_bytes, file.filename)


@router.post("/run/paddleocr-vl", summary="Run PaddleOCR-VL and return the official PaddleOCR response")
async def run_paddleocr_vl_raw(
    request: Request,
    file: UploadFile = File(..., description="Image or PDF file to process"),
    registry=Depends(get_registry),
):
    """
    Upload an image/PDF and return PaddleOCR-VL's native result.json response.
    """
    _validate_image(file)

    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

    selected_model = registry.get(PADDLEOCR_VL_MODEL_NAME)
    if selected_model is None:
        failure_reason = getattr(registry, "failed_models", {}).get(PADDLEOCR_VL_MODEL_NAME)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Configured OCR model '{PADDLEOCR_VL_MODEL_NAME}' is not loaded at startup."
                + (f" Startup error: {failure_reason}" if failure_reason else "")
            ),
        )

    return await _run_paddleocr_vl_raw_json(selected_model, image_bytes, file.filename)


@router.post("/run", summary="Run OCR on an image")
async def run_ocr(
    request: Request,
    file: UploadFile = File(..., description="Image file to process"),
    language: Language | None = Form(
        default=None,
        description=(
            "Target language. Not required for Qari OCR or PaddleOCR-VL."
        ),
    ),
    model: str = Form(
        default=PRIMARY_MODEL_NAME,
        description="OCR model to run. Options: qari_ocr_vl_2b, paddleocr_vl.",
    ),
    registry=Depends(get_registry),
):
    """
    Upload an image and run the selected OCR model on it.
    Returns extracted text, word-level details, confidence, and timing.
    """
    _validate_image(file)

    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

    model_name = (model or PRIMARY_MODEL_NAME).strip()
    selected_model = registry.get(model_name)
    if selected_model is None:
        if model_name not in AVAILABLE_MODEL_NAMES:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown OCR model '{model_name}'. Available: {AVAILABLE_MODEL_NAMES}",
            )
        failure_reason = getattr(registry, "failed_models", {}).get(model_name)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Configured OCR model '{model_name}' is not loaded at startup."
                + (f" Startup error: {failure_reason}" if failure_reason else "")
            ),
        )
    lang_enum: SupportedLanguage | None = None
    if model_name not in LANGUAGE_OPTIONAL_MODEL_NAMES:
        if language is None:
            raise HTTPException(
                status_code=422,
                detail=f"language is required when model='{model_name}'.",
            )
        lang_enum = SupportedLanguage(language.value)

    if lang_enum is not None and not selected_model.supports_language(lang_enum):
        raise HTTPException(
            status_code=404,
            detail=f"Configured OCR model '{model_name}' does not support language '{lang_enum.value}'.",
        )
    
    start = time.perf_counter()
    log_lang = lang_enum.value if lang_enum is not None else "auto"
    logger.info(f"[OCR] Running {model_name} | lang={log_lang} | file={file.filename}")
    if model_name == QARI_MODEL_NAME:
        return await _run_qari_raw_text(selected_model, image_bytes, file.filename)
    if model_name == PADDLEOCR_VL_MODEL_NAME:
        return await _run_paddleocr_vl_raw_json(selected_model, image_bytes, file.filename)

    result = await selected_model.run(image_bytes, lang_enum)
    pages = _pages_from_metadata(result.metadata)
    accepted_lines = _flatten_page_lines(pages, "accepted_lines")
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
        per_line_confidence=per_line_confidence,
        inference_time_ms=result.inference_time_ms,
        avg_confidence=result.avg_confidence,
        error=result.error,
        quality=quality,
        metadata=result.metadata,
    )]

    total_ms = round((time.perf_counter() - start) * 1000, 2)

    return OCRResponse(
        filename=file.filename,
        language=log_lang,
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
        per_line_confidence=per_line_confidence,
        avg_confidence=result.avg_confidence,
        quality=quality,
        results=results,
        models_run=len(results),
        total_time_ms=total_ms,
    )
