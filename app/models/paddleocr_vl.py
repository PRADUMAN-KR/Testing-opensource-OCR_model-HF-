
#******************************************************

# This model is not  implemented in this project  refer the V5 model for better result

#******************************************************



# """
# PaddleOCR-VL — Tier 1 VLM (0.9B)
# Smallest Tier-1 model — good accuracy, lowest VRAM (~4GB)
# Install: pip install paddlepaddle paddleocr>=2.9.0
# """

# import logging
# import tempfile
# import os
# from io import BytesIO
# from PIL import Image

# from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage
# from app.core.document import load_document_as_rgb_images

# logger = logging.getLogger(__name__)

# LANG_CONFIG = {
#     SupportedLanguage.ENGLISH: {"lang": "en", "ocr_version": "PP-OCRv4"},
#     SupportedLanguage.ARABIC: {"lang": "ar", "ocr_version": "PP-OCRv5"},
#     SupportedLanguage.HINDI: {"lang": "hi", "ocr_version": "PP-OCRv5"},
# }


# class PaddleOCRVLModel(BaseOCRModel):
#     name = "paddleocr_vl"
#     supported_languages = [
#         SupportedLanguage.ENGLISH,
#         SupportedLanguage.ARABIC,
#         SupportedLanguage.HINDI,
#     ]
#     tier = 1

#     def __init__(self, use_gpu: bool = True):
#         self.use_gpu = use_gpu
#         self._engines = {}

#     def supports_all_languages(self) -> bool:
#         return True

#     async def load(self) -> None:
#         from paddleocr import PaddleOCR
#         device = "gpu" if self.use_gpu else "cpu"
#         for lang_enum, config in LANG_CONFIG.items():
#             paddle_lang = config["lang"]
#             ocr_version = config["ocr_version"]
#             logger.info(
#                 f"[PaddleOCR-VL] Loading VL engine for lang={paddle_lang}, ocr_version={ocr_version}"
#             )
#             self._engines[lang_enum] = PaddleOCR(
#                 use_textline_orientation=True,
#                 lang=paddle_lang,
#                 device=device,
#                 ocr_version=ocr_version,
#                 use_doc_orientation_classify=True,
#                 use_doc_unwarping=True,
#                 text_det_thresh=0.5,
#                 text_det_box_thresh=0.7,
#                 text_rec_score_thresh=0.5,
#             )
#         logger.info("[PaddleOCR-VL] All language engines loaded.")

#     async def unload(self) -> None:
#         self._engines.clear()

#     @staticmethod
#     def _to_xyxy_bbox(poly) -> list[int]:
#         """Convert polygon/box outputs into [x1, y1, x2, y2]."""
#         if poly is None:
#             return [0, 0, 0, 0]
#         try:
#             if len(poly) == 0:
#                 return [0, 0, 0, 0]
#         except TypeError:
#             return [0, 0, 0, 0]
#         xs = [int(float(p[0])) for p in poly]
#         ys = [int(float(p[1])) for p in poly]
#         return [min(xs), min(ys), max(xs), max(ys)]

#     MIN_BOX_AREA = 150
#     MIN_BOX_HEIGHT = 8

#     def _parse_ocr_page(self, page_result):
#         """
#         Parse one PaddleOCR page result.
#         Supports both legacy tuple format and PaddleOCR 3.x dict format.
#         Filters out tiny/spurious boxes.
#         """
#         raw = []

#         if isinstance(page_result, dict):
#             texts = page_result.get("rec_texts") or []
#             scores = page_result.get("rec_scores") or []
#             polys = page_result.get("dt_polys") or page_result.get("rec_polys") or []
#             for i in range(min(len(texts), len(scores), len(polys))):
#                 raw.append((str(texts[i]), float(scores[i]), self._to_xyxy_bbox(polys[i])))
#         elif isinstance(page_result, list):
#             for line in page_result:
#                 if not (isinstance(line, (list, tuple)) and len(line) >= 2):
#                     continue
#                 bbox = line[0]
#                 text_conf = line[1]
#                 if not (isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2):
#                     continue
#                 raw.append((str(text_conf[0]), float(text_conf[1]), self._to_xyxy_bbox(bbox)))

#         parsed = []
#         for text, conf, (x1, y1, x2, y2) in raw:
#             w, h = x2 - x1, y2 - y1
#             if w * h < self.MIN_BOX_AREA or h < self.MIN_BOX_HEIGHT:
#                 continue
#             if not text or not text.strip():
#                 continue
#             parsed.append((text, conf, [x1, y1, x2, y2]))
#         return parsed

#     @staticmethod
#     def _page_score(parsed_page) -> float:
#         if not parsed_page:
#             return 0.0
#         return sum(conf for _, conf, _ in parsed_page)

#     def _select_all_language_page(self, img_array):
#         best_lang = None
#         best_page = []
#         best_score = -1.0
#         total_elapsed = 0.0

#         for lang_enum, engine in self._engines.items():
#             t0 = self._timer()
#             result = engine.ocr(img_array)
#             total_elapsed += self._elapsed_ms(t0)

#             parsed_page = self._parse_ocr_page(result[0]) if result and result[0] else []
#             score = self._page_score(parsed_page)
#             if score > best_score:
#                 best_score = score
#                 best_lang = lang_enum
#                 best_page = parsed_page

#         return best_lang, best_page, total_elapsed

#     async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
#         if language != SupportedLanguage.ALL:
#             engine = self._engines.get(language)
#         else:
#             engine = None

#         if language != SupportedLanguage.ALL and not engine:
#             return OCRResult.from_error(self.name, language.value, f"Language {language} not loaded")

#         try:
#             import numpy as np
#             words = []
#             page_texts = []
#             total_elapsed = 0.0
#             pages = load_document_as_rgb_images(image_bytes)
#             resolved_page_languages = []
#             for image in pages:
#                 img_array = np.array(image)
#                 if language == SupportedLanguage.ALL:
#                     page_language, parsed_page, page_elapsed = self._select_all_language_page(img_array)
#                     total_elapsed += page_elapsed
#                     resolved_page_languages.append(page_language.value if page_language else None)
#                 else:
#                     t0 = self._timer()
#                     result = engine.ocr(img_array)
#                     total_elapsed += self._elapsed_ms(t0)
#                     parsed_page = self._parse_ocr_page(result[0]) if result and result[0] else []

#                 lines = []
#                 for text, conf, flat_bbox in parsed_page:
#                     words.append(OCRWord(text=text, confidence=conf, bbox=flat_bbox))
#                     lines.append(text)
#                 page_texts.append("\n".join(lines))

#             raw_text = "\n\n".join(text for text in page_texts if text)
#             avg_conf = sum(w.confidence for w in words) / len(words) if words else 0.0
#             metadata = {"doc_unwarping": True, "page_count": len(pages)}
#             if language == SupportedLanguage.ALL:
#                 metadata["best_effort_language_mode"] = "best_single_language_engine_per_page"
#                 metadata["resolved_page_languages"] = resolved_page_languages

#             return OCRResult(
#                 model_name=self.name,
#                 language=language.value,
#                 raw_text=raw_text,
#                 words=words,
#                 inference_time_ms=round(total_elapsed, 2),
#                 avg_confidence=round(avg_conf, 4),
#                 metadata=metadata,
#             )

#         except Exception as e:
#             logger.exception(f"[PaddleOCR-VL] Inference error: {e}")
#             return OCRResult.from_error(self.name, language.value, str(e))
