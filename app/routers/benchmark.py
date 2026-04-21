
####################################################
          #uncomment this when needed 

#######################################################




# """
# /benchmark endpoints — run OCR models and evaluate against ground truth text.
# Returns CER, WER, NED, F1 per model so you can compare accuracy side-by-side.
# """

# import time
# import logging
# from typing import Optional

# from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends

# from app.schemas import BenchmarkResponse, BenchmarkModelResult, MetricsDetail, Language
# from app.models.base import SupportedLanguage
# from app.core.metrics import compute_metrics
# from app.core.config import settings
# from app.core.model_selection import (
#     find_unknown_requested_models,
#     resolve_requested_models,
# )

# logger = logging.getLogger(__name__)
# router = APIRouter()


# def get_registry(request: Request):
#     return request.app.state.model_registry


# @router.post(
#     "/evaluate",
#     response_model=BenchmarkResponse,
#     summary="Benchmark OCR models against ground truth",
# )
# async def evaluate(
#     request: Request,
#     file: UploadFile = File(..., description="Image file to OCR"),
#     ground_truth: str = Form(..., description="Expected correct text for accuracy evaluation"),
#     language: Language = Form(
#         ...,
#         description=(
#             "Target language. Use 'all' to run best-effort multilingual OCR and compare extracted full-text output."
#         ),
#     ),
#     models: Optional[str] = Form(
#         default=None,
#         description=(
#             "Comma-separated model names. Leave blank for all loaded models. "
#             "You can also pass OCR_WITHOUT_LLM_CAPABILITIES to run all currently loaded pure OCR models."
#         ),
#     ),
#     registry=Depends(get_registry),
# ):
#     """
#     Upload an image + ground truth text.
#     Runs all (or selected) OCR models and returns accuracy metrics for each:
#     - CER  — Character Error Rate
#     - WER  — Word Error Rate
#     - NED  — Normalized Edit Distance
#     - Char/Word Precision, Recall, F1
#     - Exact Match
#     """
#     if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
#         raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

#     image_bytes = await file.read()
#     if len(image_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
#         raise HTTPException(status_code=413, detail="File too large")

#     if not ground_truth.strip():
#         raise HTTPException(status_code=422, detail="ground_truth cannot be empty")

#     lang_enum = SupportedLanguage(language.value)
#     loaded_model_names = list(registry.all().keys())
#     requested_models = resolve_requested_models(models, loaded_model_names)
#     unknown_requested = find_unknown_requested_models(models, loaded_model_names)
#     if unknown_requested:
#         raise HTTPException(
#             status_code=422,
#             detail={
#                 "message": "Some requested models are not loaded at startup.",
#                 "requested_models": requested_models,
#                 "unknown_models": unknown_requested,
#                 "loaded_models": sorted(loaded_model_names),
#             },
#         )
#     selected = {
#         name: model
#         for name, model in registry.all().items()
#         if name in requested_models and model.supports_language(lang_enum)
#     }

#     if not selected:
#         raise HTTPException(
#             status_code=404,
#             detail=f"No loaded models support language '{language.value}' among: {requested_models}",
#         )

#     start = time.perf_counter()
#     results = []

#     for name, model in selected.items():
#         logger.info(f"[Benchmark] {name} | lang={language.value}")
#         ocr_result = await model.run(image_bytes, lang_enum)

#         if ocr_result.error:
#             metrics = MetricsDetail(
#                 cer=1.0, wer=1.0, ned=1.0,
#                 char_precision=0.0, char_recall=0.0, char_f1=0.0,
#                 word_precision=0.0, word_recall=0.0, word_f1=0.0,
#                 overall_accuracy=0.0,
#                 exact_match=False,
#             )
#         else:
#             m = compute_metrics(ocr_result.raw_text, ground_truth)
#             metrics = MetricsDetail(
#                 cer=m.cer, wer=m.wer, ned=m.ned,
#                 char_precision=m.char_precision,
#                 char_recall=m.char_recall,
#                 char_f1=m.char_f1,
#                 word_precision=m.word_precision,
#                 word_recall=m.word_recall,
#                 word_f1=m.word_f1,
#                 overall_accuracy=m.overall_accuracy,
#                 exact_match=m.exact_match,
#             )

#         results.append(BenchmarkModelResult(
#             model_name=ocr_result.model_name,
#             language=ocr_result.language,
#             raw_text=ocr_result.raw_text,
#             inference_time_ms=ocr_result.inference_time_ms,
#             avg_confidence=ocr_result.avg_confidence,
#             metrics=metrics,
#             error=ocr_result.error,
#         ))

#     total_ms = round((time.perf_counter() - start) * 1000, 2)

#     # Find best model per metric (lowest CER/WER, highest F1/overall accuracy)
#     valid = [r for r in results if not r.error]
#     best_cer = min(valid, key=lambda r: r.metrics.cer).model_name if valid else "N/A"
#     best_wer = min(valid, key=lambda r: r.metrics.wer).model_name if valid else "N/A"
#     best_f1 = max(valid, key=lambda r: r.metrics.word_f1).model_name if valid else "N/A"

#     return BenchmarkResponse(
#         filename=file.filename,
#         language=language.value,
#         ground_truth=ground_truth,
#         results=results,
#         best_model_cer=best_cer,
#         best_model_wer=best_wer,
#         best_model_f1=best_f1,
#         total_time_ms=total_ms,
#     )


# @router.post(
#     "/batch",
#     summary="Batch benchmark multiple images",
# )
# async def batch_evaluate(
#     request: Request,
#     files: list[UploadFile] = File(...),
#     ground_truths: str = Form(..., description="Newline-separated ground truth strings, one per image"),
#     language: Language = Form(..., description="Target language, or 'all' for best-effort multilingual OCR."),
#     models: Optional[str] = Form(default=None),
#     registry=Depends(get_registry),
# ):
#     """
#     Run benchmark on multiple images at once.
#     ground_truths must have the same number of lines as files uploaded.
#     Returns per-image results + aggregate averages per model.
#     """
#     gt_list = [line.strip() for line in ground_truths.strip().split("\n") if line.strip()]

#     if len(gt_list) != len(files):
#         raise HTTPException(
#             status_code=422,
#             detail=f"Mismatch: {len(files)} files but {len(gt_list)} ground truth lines",
#         )

#     lang_enum = SupportedLanguage(language.value)
#     loaded_model_names = list(registry.all().keys())
#     requested_models = resolve_requested_models(models, loaded_model_names)
#     unknown_requested = find_unknown_requested_models(models, loaded_model_names)
#     if unknown_requested:
#         raise HTTPException(
#             status_code=422,
#             detail={
#                 "message": "Some requested models are not loaded at startup.",
#                 "requested_models": requested_models,
#                 "unknown_models": unknown_requested,
#                 "loaded_models": sorted(loaded_model_names),
#             },
#         )
#     selected = {
#         name: model
#         for name, model in registry.all().items()
#         if name in requested_models and model.supports_language(lang_enum)
#     }

#     # Aggregate: model_name -> list of metric dicts
#     from collections import defaultdict
#     aggregates = defaultdict(lambda: {
#         "cer": [],
#         "wer": [],
#         "word_f1": [],
#         "overall_accuracy": [],
#         "inference_time_ms": [],
#     })

#     all_results = []
#     for idx, (file, gt) in enumerate(zip(files, gt_list)):
#         image_bytes = await file.read()
#         image_results = []

#         for name, model in selected.items():
#             ocr_result = await model.run(image_bytes, lang_enum)
#             if not ocr_result.error:
#                 m = compute_metrics(ocr_result.raw_text, gt)
#                 aggregates[name]["cer"].append(m.cer)
#                 aggregates[name]["wer"].append(m.wer)
#                 aggregates[name]["word_f1"].append(m.word_f1)
#                 aggregates[name]["overall_accuracy"].append(m.overall_accuracy)
#                 aggregates[name]["inference_time_ms"].append(ocr_result.inference_time_ms)
#                 image_results.append({
#                     "model": name,
#                     "cer": m.cer,
#                     "wer": m.wer,
#                     "word_f1": m.word_f1,
#                     "overall_accuracy": m.overall_accuracy,
#                     "raw_text": ocr_result.raw_text[:200],
#                 })

#         all_results.append({"image": file.filename, "ground_truth": gt, "results": image_results})

#     # Build summary
#     summary = {}
#     for model_name, agg in aggregates.items():
#         n = len(agg["cer"])
#         summary[model_name] = {
#             "avg_cer": round(sum(agg["cer"]) / n, 4),
#             "avg_wer": round(sum(agg["wer"]) / n, 4),
#             "avg_word_f1": round(sum(agg["word_f1"]) / n, 4),
#             "avg_overall_accuracy": round(sum(agg["overall_accuracy"]) / n, 4),
#             "avg_inference_ms": round(sum(agg["inference_time_ms"]) / n, 2),
#             "images_evaluated": n,
#         }

#     return {"language": language.value, "per_image": all_results, "aggregate_summary": summary}



#-------------------------------------------------------------------
# this should be the not be included in the main projects