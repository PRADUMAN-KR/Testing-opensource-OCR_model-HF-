"""
PaddleOCR PP-OCRv5 — Tier 2

Standard pipeline + F2 MAX-ACCURACY pipeline (enabled when max_accuracy=True):

  Stage 1 — Full-page variants (Arabic: original, deskew+CLAHE,
             light sharpen+contrast; non-Arabic keeps original, 2x, CLAHE)
  Stage 2 — Run full engine.ocr() on each variant × each engine
  Stage 3 — Pick best complete pass using deterministic confidence, Arabic word
             count, dictionary coverage, low-confidence, and broken-word scoring.
  Stage 4  — Block/column-aware RTL layout reconstruction.
  Stage 5  — Conservative Arabic cleanup without inventing missing OCR words.

Supports: English, Arabic, Hindi
Install:  pip install paddlepaddle paddleocr opencv-python python-bidi arabic-reshaper
"""

import logging
import math
import os
import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage
from app.core.document import load_document_as_rgb_images
from app.ocr_postprocess import line_noise_score, postprocess_ocr_result, score_ocr_words
from app.ocr_postprocess.utils import split_line_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language → PaddleOCR engine config
# ---------------------------------------------------------------------------

LANG_CONFIG: dict[SupportedLanguage, dict] = {
    SupportedLanguage.ENGLISH: {"lang": "en", "ocr_version": "PP-OCRv5"},
    SupportedLanguage.ARABIC:  {"lang": "ar", "ocr_version": "PP-OCRv5"},
    SupportedLanguage.HINDI:   {"lang": "hi", "ocr_version": "PP-OCRv5"},
}

# Alternate ocr_version engines for F2 multi-engine diversity.
# Only versions that have models for the given language are listed.
_ALT_ENGINES: dict[SupportedLanguage, list[str]] = {
    SupportedLanguage.ARABIC:  ["PP-OCRv3"],   # PP-OCRv5 primary + PP-OCRv3 secondary
    SupportedLanguage.HINDI:   [],
    SupportedLanguage.ENGLISH: [],
}

# ---------------------------------------------------------------------------
# Arabic normalisation tables and constants (applied in Stage 5)
# ---------------------------------------------------------------------------

# Invisible / zero-width characters that carry no readable value.
_ARABIC_CLEANUP_MAP = str.maketrans({
    "\u200C": "",   # ZWNJ
    "\u200D": "",   # ZWJ
    "\u200B": "",   # zero-width space
    "\u200E": "",   # LRM
    "\u200F": "",   # RLM
    "\uFEFF": "",   # BOM / ZWNBSP
})

# Arabic punctuation → canonical Unicode equivalents frequently misread by OCR.
_ARABIC_PUNCT_MAP = str.maketrans({
    ",":  "\u060C",  # Western comma  → Arabic comma  ،
    ";":  "\u061B",  # Western semi   → Arabic semi   ؛
    "?":  "\u061F",  # Western ?      → Arabic ?      ؟
})

# Alef variants → bare Alef (U+0627).  Applied only when normalize_alef=True.
_ALEF_VARIANTS = str.maketrans({
    "\u0622": "\u0627",  # Alef with Madda  آ → ا
    "\u0623": "\u0627",  # Alef with Hamza Above  أ → ا
    "\u0625": "\u0627",  # Alef with Hamza Below  إ → ا
    "\u0671": "\u0627",  # Alef Wasla  ٱ → ا
})

# Diacritic Unicode range U+064B–U+0652 (Fathatan … Sukun).
_RE_DIACRITICS         = re.compile(r"[\u064B-\u0652]")
# Kashida (tatweel) runs.
_RE_KASHIDA            = re.compile(r"\u0640+")
# Repeated diacritics (two or more identical diacritic codepoints in a row).
_RE_REPEAT_DIACRITIC   = re.compile(r"([\u064B-\u0652])\1+")
# Generic repeated character run of 5+ (noise detection).
_RE_REPEATED           = re.compile(r"(.)\1{4,}")
# Isolated single Arabic letter token (likely detection artefact).
_RE_ISOLATED_AR_LETTER = re.compile(r"^[\u0600-\u06FF]$")

# Arabic base-letter codepoint ranges used to count genuine word tokens.
_ARABIC_BASE_RANGES = (
    ("\u0600", "\u06FF"),  # Arabic block
    ("\u0750", "\u077F"),  # Arabic Supplement
    ("\uFB50", "\uFDFF"),  # Arabic Presentation Forms-A
    ("\uFE70", "\uFEFF"),  # Arabic Presentation Forms-B
)

# Fallback band tolerance when bbox statistics are unavailable.
_ARABIC_LINE_BAND_TOL_PX = 15

# High-DPI threshold: if max(h, w) exceeds this, add a 0.75× downscale variant.
_HIGHDPI_THRESHOLD_PX = 3000


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class PaddleOCRv5Model(BaseOCRModel):
    """
    PaddleOCR wrapper with an optional F2 MAX-ACCURACY pipeline.

    The F2 pipeline runs the full PaddleOCR det+rec engine on multiple
    preprocessed versions of the entire page (not isolated crops), then
    merges/scores results at the text-line level via bbox IoU alignment.
    """

    name = "paddleocr_v5"
    supported_languages = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.ARABIC,
        SupportedLanguage.HINDI,
    ]
    tier = 2

    def __init__(
        self,
        use_gpu: bool = False,
        max_accuracy: bool = True,
        debug_output_dir: str | None = None,
        enable_arabic_v3_fallback: bool = False,
        always_run_both_arabic_engines: bool = False,
        fallback_min_lines: int = 4,
        fallback_min_chars: int = 80,
        fallback_min_avg_conf: float = 0.72,
        fallback_min_ar_ratio: float = 0.60,
        fallback_replace_margin: float = 0.03,
        input_roi_warp: bool = False,
        arabic_auto_page_crop: bool = True,
        roi_min_area_ratio: float = 0.15,
        roi_pad_ratio: float = 0.02,
        paddle_mem_fraction: float | None = None,
        paddle_allocator_strategy: str | None = None,
        paddle_gpu_memory_fraction: float | None = None,
        empty_cache_between_pages: bool = False,
        det_limit_side_len: int | None = None,
        text_det_thresh: float = 0.3,
        text_det_box_thresh: float = 0.45,
        text_rec_score_thresh: float = 0.3,
        precision: str = "fp32",
        enable_fp16: bool = False,
        use_tensorrt: bool = False,
        # ---- Arabic-specific options ----
        # Normalize all Alef variants (أ إ آ ٱ) to bare Alef (ا).
        # Improves downstream search/NLP matching but loses orthographic detail.
        # Leave False for archival / faithful transcription use-cases.
        arabic_normalize_alef: bool = False,
        # Legacy flag: previously applied python-bidi get_display() to “fix” order; that
        # API maps logical→visual and scrambled Paddle output (already logical). Kept
        # for config compatibility; has no effect on corrected text (see _to_logical_order).
        arabic_normalize_bidi: bool = False,
        # Filter isolated single-letter tokens that are almost always det artefacts.
        arabic_filter_isolated_letters: bool = True,
        # Mixed Arabic-English ratio threshold: pages above this fraction of
        # non-Arabic chars are treated as mixed and get a relaxed scoring gate.
        arabic_mixed_page_ratio_threshold: float = 0.35,
    ):
        self.use_gpu = use_gpu
        self.max_accuracy = max_accuracy
        self.debug_output_dir = debug_output_dir
        # Optional Arabic v3 fallback (disabled by default to save VRAM).
        self.enable_arabic_v3_fallback = enable_arabic_v3_fallback
        # If enabled, always run both Arabic engines (v5 + v3) for comparison.
        self.always_run_both_arabic_engines = always_run_both_arabic_engines
        self.fallback_min_lines = fallback_min_lines
        self.fallback_min_chars = fallback_min_chars
        self.fallback_min_avg_conf = fallback_min_avg_conf
        self.fallback_min_ar_ratio = fallback_min_ar_ratio
        self.fallback_replace_margin = fallback_replace_margin
        # Optional: largest-quad perspective warp before OCR (napkin / document on clutter).
        self.input_roi_warp = input_roi_warp
        self.arabic_auto_page_crop = arabic_auto_page_crop
        self.roi_min_area_ratio = roi_min_area_ratio
        self.roi_pad_ratio = roi_pad_ratio
        self.paddle_mem_fraction = paddle_mem_fraction
        self.paddle_allocator_strategy = paddle_allocator_strategy
        self.paddle_gpu_memory_fraction = paddle_gpu_memory_fraction
        self.empty_cache_between_pages = empty_cache_between_pages
        self.det_limit_side_len = det_limit_side_len
        self.text_det_thresh = text_det_thresh
        self.text_det_box_thresh = text_det_box_thresh
        self.text_rec_score_thresh = text_rec_score_thresh
        requested_precision = (precision or "fp32").strip().lower()
        if requested_precision not in {"fp32", "fp16", "bf16"}:
            logger.warning(
                "[PaddleOCR] Unsupported precision=%s requested; falling back to fp32",
                precision,
            )
            requested_precision = "fp32"
        if enable_fp16 and requested_precision == "fp32":
            requested_precision = "fp16"
        self.precision = requested_precision
        self.use_tensorrt = bool(use_tensorrt)
        # Arabic-specific options
        self.arabic_normalize_alef = arabic_normalize_alef
        self.arabic_normalize_bidi = arabic_normalize_bidi
        self.arabic_filter_isolated_letters = arabic_filter_isolated_letters
        self.arabic_mixed_page_ratio_threshold = arabic_mixed_page_ratio_threshold
        # Primary full-pipeline engines (one per language)
        self._engines: dict[SupportedLanguage, object] = {}
        # F2: alternate-version full-pipeline engines per language
        self._alt_engines: dict[SupportedLanguage, list[tuple[str, object]]] = {}
        self._paddleocr_cls = None
        self._device = "cpu"

    def supports_all_languages(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Load / Unload
    # ------------------------------------------------------------------

    async def load(self) -> None:
        if self.paddle_allocator_strategy:
            os.environ["FLAGS_allocator_strategy"] = self.paddle_allocator_strategy
        if self.paddle_gpu_memory_fraction is not None:
            os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = str(self.paddle_gpu_memory_fraction)

        from paddleocr import PaddleOCR

        self._paddleocr_cls = PaddleOCR
        self._device = "gpu" if self.use_gpu else "cpu"
        logger.info(
            "[PaddleOCR] Device: %s | max_accuracy(F2): %s | precision=%s | TensorRT=%s | mem_fraction=%s | flags_allocator=%s | flags_gpu_fraction=%s | load_mode=lazy",
            self._device.upper(),
            self.max_accuracy,
            self.precision,
            self.use_tensorrt,
            self.paddle_mem_fraction if self.paddle_mem_fraction is not None else "default",
            self.paddle_allocator_strategy if self.paddle_allocator_strategy else "default",
            self.paddle_gpu_memory_fraction if self.paddle_gpu_memory_fraction is not None else "default",
        )

    async def unload(self) -> None:
        self._engines.clear()
        self._alt_engines.clear()
        self._paddleocr_cls = None

    def _build_engine(self, paddle_lang: str, ocr_version: str) -> object:
        if self._paddleocr_cls is None:
            raise RuntimeError("PaddleOCR class is not initialized. Call load() first.")

        requested_precision = self.precision if self.use_gpu else "fp32"
        requested_tensorrt = bool(self.use_gpu and self.use_tensorrt)
        kwargs = {
            "use_textline_orientation": True,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "lang": paddle_lang,
            "device": self._device,
            "ocr_version": ocr_version,
            "text_det_thresh": self.text_det_thresh,
            "text_det_box_thresh": self.text_det_box_thresh,
            "text_rec_score_thresh": self.text_rec_score_thresh,
        }
        if self.use_gpu:
            if self.precision != "fp32":
                kwargs["precision"] = self.precision
            if self.use_tensorrt:
                kwargs["use_tensorrt"] = True
        if self.paddle_mem_fraction is not None:
            kwargs["paddle_mem_fraction"] = self.paddle_mem_fraction
        if self.det_limit_side_len is not None:
            kwargs["det_limit_side_len"] = self.det_limit_side_len
            # Paddle's resize logging ("max_side_limit") is typically driven by
            # `limit_side_len`, so set it as well to ensure the cap applies.
            kwargs["limit_side_len"] = self.det_limit_side_len

        try:
            logger.info(
                "[PaddleOCR] Engine kwargs: lang=%s ocr_version=%s precision=%s use_tensorrt=%s det_limit_side_len=%s limit_side_len=%s det_thresh=%s box_thresh=%s rec_thresh=%s",
                paddle_lang,
                ocr_version,
                kwargs.get("precision", "fp32"),
                kwargs.get("use_tensorrt", False),
                kwargs.get("det_limit_side_len"),
                kwargs.get("limit_side_len"),
                kwargs.get("text_det_thresh"),
                kwargs.get("text_det_box_thresh"),
                kwargs.get("text_rec_score_thresh"),
            )
            engine = self._paddleocr_cls(**kwargs)
            logger.info(
                "[PaddleOCR] Engine active: lang=%s ocr_version=%s device=%s requested_precision=%s applied_precision=%s requested_tensorrt=%s applied_tensorrt=%s",
                paddle_lang,
                ocr_version,
                self._device,
                requested_precision,
                kwargs.get("precision", "fp32"),
                requested_tensorrt,
                kwargs.get("use_tensorrt", False),
            )
            return engine
        except (TypeError, ValueError, ModuleNotFoundError) as exc:
            msg = str(exc)
            retried = False

            if "paddle_mem_fraction" in kwargs and "paddle_mem_fraction" in msg:
                logger.warning(
                    "[PaddleOCR] paddle_mem_fraction unsupported by current PaddleOCR build; falling back: %s",
                    exc,
                )
                kwargs.pop("paddle_mem_fraction", None)
                retried = True

            if "precision" in kwargs and "precision" in msg:
                logger.warning(
                    "[PaddleOCR] precision=%s unsupported by current PaddleOCR build; falling back to fp32: %s",
                    kwargs.get("precision"),
                    exc,
                )
                kwargs.pop("precision", None)
                retried = True

            if "use_tensorrt" in kwargs and "use_tensorrt" in msg:
                logger.warning(
                    "[PaddleOCR] use_tensorrt unsupported by current PaddleOCR build; disabling TensorRT: %s",
                    exc,
                )
                kwargs.pop("use_tensorrt", None)
                retried = True

            if "use_tensorrt" in kwargs and isinstance(exc, ModuleNotFoundError) and "tensorrt" in msg.lower():
                logger.warning(
                    "[PaddleOCR] TensorRT requested but Python package 'tensorrt' is not installed; disabling TensorRT and continuing: %s",
                    exc,
                )
                kwargs.pop("use_tensorrt", None)
                retried = True

            if (
                "det_limit_side_len" in kwargs
                and "det_limit_side_len" in msg
                and self.det_limit_side_len is not None
            ):
                # Some PaddleOCR builds only support `limit_side_len` (applies globally).
                kwargs.pop("det_limit_side_len", None)
                if "limit_side_len" not in kwargs:
                    kwargs["limit_side_len"] = self.det_limit_side_len
                retried = True

            if "limit_side_len" in kwargs and "limit_side_len" in msg:
                kwargs.pop("limit_side_len", None)
                retried = True

            for threshold_key in (
                "text_det_thresh",
                "text_det_box_thresh",
                "text_rec_score_thresh",
            ):
                if threshold_key in kwargs and threshold_key in msg:
                    logger.warning(
                        "[PaddleOCR] %s unsupported by current PaddleOCR build; falling back: %s",
                        threshold_key,
                        exc,
                    )
                    kwargs.pop(threshold_key, None)
                    retried = True

            if retried:
                engine = self._paddleocr_cls(**kwargs)
                logger.info(
                    "[PaddleOCR] Engine active after fallback: lang=%s ocr_version=%s device=%s requested_precision=%s applied_precision=%s requested_tensorrt=%s applied_tensorrt=%s",
                    paddle_lang,
                    ocr_version,
                    self._device,
                    requested_precision,
                    kwargs.get("precision", "fp32"),
                    requested_tensorrt,
                    kwargs.get("use_tensorrt", False),
                )
                return engine
            raise

    def _get_or_load_primary_engine(self, lang_enum: SupportedLanguage):
        engine = self._engines.get(lang_enum)
        if engine is not None:
            return engine

        config = LANG_CONFIG.get(lang_enum)
        if not config:
            return None

        paddle_lang = config["lang"]
        ocr_version = config["ocr_version"]
        logger.info("[PaddleOCR] Lazy-loading engine: lang=%s version=%s", paddle_lang, ocr_version)
        engine = self._build_engine(paddle_lang, ocr_version)
        self._engines[lang_enum] = engine
        return engine

    def _get_or_load_alt_engines(self, lang_enum: SupportedLanguage) -> list[tuple[str, object]]:
        loaded = self._alt_engines.get(lang_enum)
        if loaded is not None:
            return loaded

        config = LANG_CONFIG.get(lang_enum)
        if not config:
            self._alt_engines[lang_enum] = []
            return []

        paddle_lang = config["lang"]
        alt_list: list[tuple[str, object]] = []
        for alt_ver in _ALT_ENGINES.get(lang_enum, []):
            try:
                logger.info("[PaddleOCR F2] Lazy-loading alt engine: lang=%s version=%s", paddle_lang, alt_ver)
                alt_engine = self._build_engine(paddle_lang, alt_ver)
                alt_list.append((alt_ver, alt_engine))
            except Exception as exc:
                logger.warning("[PaddleOCR F2] Alt engine %s unavailable for %s: %s", alt_ver, paddle_lang, exc)

        self._alt_engines[lang_enum] = alt_list
        if alt_list:
            logger.info(
                "[PaddleOCR F2] %d alt engine(s) for %s: %s",
                len(alt_list), paddle_lang, [v for v, _ in alt_list],
            )
        return alt_list

    def _maybe_empty_gpu_cache(self, page_idx: int) -> None:
        if not (self.use_gpu and self.empty_cache_between_pages):
            return
        try:
            import paddle

            paddle.device.cuda.empty_cache()
        except Exception as exc:
            logger.debug("[PaddleOCR] GPU cache clear skipped on page %d: %s", page_idx, exc)

    # ------------------------------------------------------------------
    # Input pipeline — document ROI (before detection)
    # ------------------------------------------------------------------

    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        """Order 4 points as tl, tr, br, bl (OpenCV doc-scanner convention)."""
        pts = pts.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _find_bright_page_rect(self, gray: np.ndarray) -> tuple[int, int, int, int] | None:
        """
        Fallback page detector for camera photos where page edges are faint and
        Canny/approxPoly misses the quad. It looks for the largest bright,
        page-shaped connected component, which works well for book pages on
        darker desks and excludes adjacent side pages.
        """
        h, w = gray.shape[:2]
        img_area = float(h * w)
        min_area = max(self.roi_min_area_ratio * img_area, 0.08 * img_area)
        max_area = 0.92 * img_area
        best: tuple[float, int, int, int, int] | None = None

        for thresh in (205, 195, 185, 175, 165):
            mask = cv2.inRange(gray, thresh, 255)
            close_k = max(9, int(min(h, w) * 0.025))
            open_k = max(5, int(min(h, w) * 0.010))
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                np.ones((close_k, close_k), np.uint8),
                iterations=1,
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                np.ones((open_k, open_k), np.uint8),
                iterations=1,
            )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw <= 0 or bh <= 0:
                    continue
                aspect = bw / float(bh)
                extent = area / float(bw * bh)
                touches_left_and_right = x <= 2 and x + bw >= w - 2
                touches_top_and_bottom = y <= 2 and y + bh >= h - 2
                if touches_left_and_right and touches_top_and_bottom:
                    continue
                if not (0.42 <= aspect <= 0.95):
                    continue
                if extent < 0.55:
                    continue
                score = area * (1.0 - min(0.5, abs(aspect - 0.66)))
                if best is None or score > best[0]:
                    best = (score, x, y, bw, bh)
            if best is not None:
                break

        if best is None:
            return None
        _, x, y, bw, bh = best
        pad = max(4, int(self.roi_pad_ratio * max(bw, bh)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        return x1, y1, x2, y2

    def _apply_document_roi_warp(
        self,
        img_rgb: np.ndarray,
        language: SupportedLanguage | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Find largest plausible document quad (Canny → contours → approxPoly)
        and apply perspective warp. Falls back to original image on failure.
        """
        meta: dict = {
            "roi_warp_applied": False,
            "roi_warp_reason": "disabled",
            "page_crop_applied": False,
            "page_crop_reason": "disabled",
        }
        if not self.input_roi_warp:
            return img_rgb, meta

        h, w = img_rgb.shape[:2]
        img_area = float(h * w)
        if img_area < 5000:
            meta["roi_warp_reason"] = "image_too_small"
            meta["page_crop_reason"] = "image_too_small"
            return img_rgb, meta

        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            meta["roi_warp_reason"] = "no_contours"
            return self._apply_bright_page_crop_fallback(img_rgb, gray, language, meta)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        min_area = self.roi_min_area_ratio * img_area
        quad = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                break
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2)
                break

        if quad is None:
            meta["roi_warp_reason"] = "no_suitable_quad"
            return self._apply_bright_page_crop_fallback(img_rgb, gray, language, meta)

        ordered = self._order_quad_points(quad)
        (tl, tr, br, bl) = ordered
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_w = int(max(width_a, width_b))
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_h = int(max(height_a, height_b))
        max_w = max(max_w, 32)
        max_h = max(max_h, 32)

        dst = np.array(
            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
            dtype=np.float32,
        )
        m = cv2.getPerspectiveTransform(ordered, dst)
        warped_bgr = cv2.warpPerspective(bgr, m, (max_w, max_h))
        pad = max(2, int(self.roi_pad_ratio * max(max_w, max_h)))
        warped_bgr = cv2.copyMakeBorder(
            warped_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        out_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
        meta["roi_warp_applied"] = True
        meta["roi_warp_reason"] = "ok"
        meta["page_crop_reason"] = "roi_warp_applied"
        meta["roi_warp_size"] = [out_rgb.shape[1], out_rgb.shape[0]]
        logger.info(
            "[PaddleOCR] ROI warp applied: %dx%d → %dx%d (pad=%d)",
            w, h, out_rgb.shape[1], out_rgb.shape[0], pad,
        )
        return out_rgb, meta

    def _apply_bright_page_crop_fallback(
        self,
        img_rgb: np.ndarray,
        gray: np.ndarray,
        language: SupportedLanguage | None,
        meta: dict,
    ) -> tuple[np.ndarray, dict]:
        if language not in (None, SupportedLanguage.ARABIC):
            meta["page_crop_reason"] = "not_arabic"
            return img_rgb, meta
        if not self.arabic_auto_page_crop:
            meta["page_crop_reason"] = "disabled"
            return img_rgb, meta
        rect = self._find_bright_page_rect(gray)
        if rect is None:
            meta["page_crop_reason"] = "no_bright_page_rect"
            return img_rgb, meta
        x1, y1, x2, y2 = rect
        cropped = img_rgb[y1:y2, x1:x2]
        if cropped.size == 0:
            meta["page_crop_reason"] = "empty_crop"
            return img_rgb, meta
        meta["page_crop_applied"] = True
        meta["page_crop_reason"] = "roi_fallback_bright_page_rect"
        meta["page_crop_bbox"] = [int(x1), int(y1), int(x2), int(y2)]
        meta["page_crop_size"] = [int(cropped.shape[1]), int(cropped.shape[0])]
        logger.info(
            "[PaddleOCR] Bright-page crop applied: %dx%d → %dx%d bbox=%s",
            img_rgb.shape[1],
            img_rgb.shape[0],
            cropped.shape[1],
            cropped.shape[0],
            meta["page_crop_bbox"],
        )
        return cropped, meta

    # ------------------------------------------------------------------
    # Stage 1 — Full-page preprocessing variants
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_skew_angle(gray: np.ndarray) -> float:
        """
        Estimate dominant text-line skew angle (degrees) via Hough on
        a binarized image.  Returns 0.0 on failure so callers can safely
        apply np.rot90(..., 0) i.e. no rotation.
        Clamped to ±10° to avoid rotating legitimate rotated stamps/logos.
        """
        try:
            _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
            dilated = cv2.dilate(binarized, kernel, iterations=1)
            lines = cv2.HoughLinesP(
                dilated, 1, np.pi / 180, threshold=80,
                minLineLength=gray.shape[1] // 5, maxLineGap=20,
            )
            if lines is None or len(lines) == 0:
                return 0.0
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    if abs(angle) <= 10:
                        angles.append(angle)
            if not angles:
                return 0.0
            return float(np.median(angles))
        except Exception:
            return 0.0

    @staticmethod
    def _rotate_image(img_rgb: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate image by angle_deg around centre; fill with white."""
        if abs(angle_deg) < 0.2:
            return img_rgb
        h, w = img_rgb.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w / 2.0) - cx
        M[1, 2] += (new_h / 2.0) - cy
        rotated = cv2.warpAffine(
            img_rgb, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return rotated

    def _generate_page_variants(
        self,
        img_rgb: np.ndarray,
        language: SupportedLanguage = SupportedLanguage.ENGLISH,
    ) -> list[tuple[str, float, np.ndarray]]:
        """
        Generate full-page preprocessing variants.

        Arabic variants are intentionally limited and ordered:
          - original
          - deskew_clahe
          - sharpen_contrast

        Returns list of (name, scale_factor, image_rgb).
        scale_factor is relative to the original.
        """
        variants: list[tuple[str, float, np.ndarray]] = []

        variants.append(("original", 1.0, img_rgb))

        h, w = img_rgb.shape[:2]

        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ------------------------------------------------------------------
        # Arabic-specific variants (only added when language == ARABIC)
        # ------------------------------------------------------------------
        if language == SupportedLanguage.ARABIC:
            # 1) deskew + CLAHE: rotate first so contrast enhancement operates
            # on the corrected text baselines.
            skew = self._estimate_skew_angle(gray)
            deskewed_rgb = self._rotate_image(img_rgb, -skew) if abs(skew) >= 0.2 else img_rgb
            if abs(skew) >= 0.2:
                logger.debug("[PaddleOCR] Deskew applied: %.2f°", skew)
            deskewed_gray = cv2.cvtColor(
                cv2.cvtColor(deskewed_rgb, cv2.COLOR_RGB2BGR),
                cv2.COLOR_BGR2GRAY,
            )
            deskew_clahe = clahe.apply(deskewed_gray)
            variants.append((
                "deskew_clahe",
                1.0,
                cv2.cvtColor(cv2.cvtColor(deskew_clahe, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB),
            ))

            # 2) light sharpen + contrast: conservative unsharp masking; no
            # binarization, so faint dots and Arabic strokes are not deleted.
            contrast = cv2.convertScaleAbs(gray, alpha=1.12, beta=3)
            blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
            sharpened = cv2.addWeighted(contrast, 1.25, blur, -0.25, 0)
            variants.append((
                "sharpen_contrast",
                1.0,
                cv2.cvtColor(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB),
            ))
            return variants

        # Non-Arabic max-accuracy behavior keeps the previous simple diversity.
        up = cv2.resize(img_rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        variants.append(("upscaled_2x", 2.0, up))
        clahe_gray = clahe.apply(gray)
        variants.append((
            "clahe", 1.0,
            cv2.cvtColor(cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB),
        ))

        return variants

    # ------------------------------------------------------------------
    # Stage 2 — Run full engine.ocr() on each variant × engine
    # ------------------------------------------------------------------

    def _run_all_passes(
        self,
        img_rgb: np.ndarray,
        lang_enum: SupportedLanguage,
        include_primary: bool = True,
        include_alt: bool = False,
    ) -> list[tuple[str, str, list[tuple[str, float, list[int]]]]]:
        """
        Run every (variant × engine) combination.
        All bboxes are normalized back to the original image scale so that
        IoU alignment works across upscaled variants.
        Returns list of (variant_name, engine_name, parsed_page).
        """
        primary_engine = self._get_or_load_primary_engine(lang_enum)
        if not primary_engine:
            return []

        variants = self._generate_page_variants(img_rgb, language=lang_enum)
        engines: list[tuple[str, object]] = []
        if include_primary:
            primary_ver = LANG_CONFIG[lang_enum]["ocr_version"]
            engines.append((primary_ver, primary_engine))
        if include_alt:
            engines.extend(self._get_or_load_alt_engines(lang_enum))
        if not engines:
            return []

        all_passes: list[tuple[str, str, list[tuple[str, float, list[int]]]]] = []
        bbox_sanity = getattr(self, "_bbox_sanity_current", None)

        for variant_name, scale, variant_img in variants:
            # Hard-cap very large inputs to control VRAM.
            # PaddleOCR has internal side-length logic, but relying on it can be
            # brittle across versions; this keeps bbox normalization correct.
            effective_scale = float(scale)
            variant_for_engine = variant_img
            if self.det_limit_side_len is not None:
                vh, vw = variant_for_engine.shape[:2]
                max_side = max(vh, vw)
                if max_side > self.det_limit_side_len:
                    down_scale = self.det_limit_side_len / float(max_side)
                    new_w = max(1, int(vw * down_scale))
                    new_h = max(1, int(vh * down_scale))
                    variant_for_engine = cv2.resize(
                        variant_for_engine,
                        (new_w, new_h),
                        interpolation=cv2.INTER_AREA,
                    )
                    effective_scale = float(scale) * down_scale
                    logger.debug(
                        "[PaddleOCR F2] Downscaled variant=%s from %dx%d to %dx%d (effective_scale=%.4f)",
                        variant_name,
                        vw,
                        vh,
                        new_w,
                        new_h,
                        effective_scale,
                    )
            for engine_name, engine in engines:
                try:
                    result = engine.ocr(variant_for_engine)
                    parsed = self._parse_ocr_page(result[0]) if result and result[0] else []
                    normalized = []
                    inv = 1.0 / effective_scale if effective_scale else 1.0
                    for text, conf, bbox in parsed:
                        original_bbox = [
                            bbox[0] * inv,
                            bbox[1] * inv,
                            bbox[2] * inv,
                            bbox[3] * inv,
                        ]
                        normalized.append(self._line_with_bbox_source(
                            text,
                            conf,
                            original_bbox,
                            variant_name=variant_name,
                            scale=scale,
                            effective_scale=effective_scale,
                            original_bbox_before_rescale=bbox,
                            img_shape=img_rgb.shape,
                            engine_name=engine_name,
                        ))
                    parsed = normalized
                    parsed, rejected, corrected, median_h = self._validate_bbox_lines(
                        parsed,
                        img_rgb.shape,
                    )
                    if isinstance(bbox_sanity, dict):
                        bbox_sanity.setdefault("rejected_bbox_lines", []).extend(rejected)
                        bbox_sanity.setdefault("corrected_bbox_lines", []).extend(corrected)
                        bbox_sanity["median_line_height"] = round(median_h, 4)
                    if rejected or corrected:
                        logger.debug(
                            "[PaddleOCR F2] Bbox validation variant=%s engine=%s rejected=%d corrected=%d median_h=%.1f",
                            variant_name,
                            engine_name,
                            len(rejected),
                            len(corrected),
                            median_h,
                        )
                    all_passes.append((variant_name, engine_name, parsed))
                except Exception as exc:
                    logger.warning(
                        "[PaddleOCR F2] Pass failed (%s × %s): %s",
                        variant_name, engine_name, exc,
                    )

        logger.info(
            "[PaddleOCR F2] Stage 2: %d passes completed (%d variants × %d engines)",
            len(all_passes), len(variants), len(engines),
        )
        return all_passes

    # ------------------------------------------------------------------
    # Stage 3 — Pick the best complete pass (not line-level merging)
    # ------------------------------------------------------------------

    @staticmethod
    def _arabic_char_ratio(text: str) -> float:
        """Fraction of characters in Arabic Unicode ranges."""
        if not text:
            return 0.0
        arabic = sum(
            1 for c in text
            if "\u0600" <= c <= "\u06FF"
            or "\u0750" <= c <= "\u077F"
            or "\uFB50" <= c <= "\uFDFF"
            or "\uFE70" <= c <= "\uFEFF"
        )
        return arabic / len(text)

    @staticmethod
    def _count_arabic_word_tokens(text: str) -> int:
        """
        Count genuine Arabic word tokens: whitespace-delimited tokens that
        contain at least 2 base Arabic letters (U+0600–U+06FF, supplements,
        presentation forms).  Single-codepoint tokens and pure-diacritic
        clusters are excluded.

        This avoids inflating scores on passes that consist mostly of isolated
        diacritics or hallucinated single-character detections.
        """
        count = 0
        for token in text.split():
            base_letters = sum(
                1 for c in token
                if (
                    ("\u0600" <= c <= "\u06FF" and c not in "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0640")
                    or "\u0750" <= c <= "\u077F"
                    or "\uFB50" <= c <= "\uFDFF"
                    or "\uFE70" <= c <= "\uFEFF"
                )
            )
            if base_letters >= 2:
                count += 1
        return count

    @staticmethod
    def _diacritic_density(text: str) -> float:
        """
        Fraction of characters that are Arabic diacritics (U+064B–U+0652).
        > 0.15 indicates heavy tashkeel (Quranic / classical text).
        """
        if not text:
            return 0.0
        diacritics = sum(1 for c in text if "\u064B" <= c <= "\u0652")
        return diacritics / len(text)

    @staticmethod
    def _arabic_base_letters(text: str) -> list[str]:
        return [
            c for c in text
            if (
                ("\u0600" <= c <= "\u06FF" and not ("\u064B" <= c <= "\u0652") and c != "\u0640")
                or "\u0750" <= c <= "\u077F"
                or "\uFB50" <= c <= "\uFDFF"
                or "\uFE70" <= c <= "\uFEFF"
            )
        ]

    @classmethod
    def _arabic_noise_line_reason(cls, text: str, confidence: float) -> str | None:
        """
        Flag only lines that are both low-confidence and independently noisy.
        Low-confidence meaningful Arabic is kept for postprocess review.
        """
        if not (text or "").strip() or confidence >= 0.60:
            return None
        noise = line_noise_score(text)
        if float(noise["score"]) >= 0.65:
            return "low_confidence_high_noise_score"
        return None

    @staticmethod
    def _median_box_height(parsed: list[tuple[str, float, list[int]]]) -> float:
        """
        Compute median bounding-box height from a parsed page.
        Used to derive a dynamic band tolerance for RTL line sorting.
        Returns _ARABIC_LINE_BAND_TOL_PX if fewer than 3 boxes present.
        """
        if len(parsed) < 3:
            return float(_ARABIC_LINE_BAND_TOL_PX)
        heights = [float(item[2][3] - item[2][1]) for item in parsed if item[2][3] > item[2][1]]
        if not heights:
            return float(_ARABIC_LINE_BAND_TOL_PX)
        return statistics.median(heights)

    @staticmethod
    def _detect_columns(
        items: list,
        bbox_extractor,
        img_width: int = 0,
        bimodal_gap_ratio: float = 0.10,
    ) -> list[list]:
        """
        Detect whether bounding boxes are bimodally distributed along the
        x-axis (two-column layout).  If so, partition items into right-column
        and left-column lists.

        ``bbox_extractor(item) -> [x1, y1, x2, y2]``

        Returns a list of column groups ordered right → left (Arabic reading
        order).  Returns [items] (single column) when no clear gap is found.
        """
        if len(items) < 6:
            return [items]

        centers_x = sorted(bbox_extractor(it)[0] + (bbox_extractor(it)[2] - bbox_extractor(it)[0]) / 2
                            for it in items)
        # Find the largest gap between consecutive x-centers.
        gaps = [(centers_x[i + 1] - centers_x[i], i) for i in range(len(centers_x) - 1)]
        max_gap, split_idx = max(gaps, key=lambda g: g[0])
        page_width = img_width or max(bbox_extractor(it)[2] for it in items)
        if page_width == 0 or max_gap / page_width < bimodal_gap_ratio:
            return [items]

        split_x = (centers_x[split_idx] + centers_x[split_idx + 1]) / 2
        left_col  = [it for it in items if (bbox_extractor(it)[0] + bbox_extractor(it)[2]) / 2 <  split_x]
        right_col = [it for it in items if (bbox_extractor(it)[0] + bbox_extractor(it)[2]) / 2 >= split_x]
        # Arabic: right column first
        return [right_col, left_col]

    @classmethod
    def _sort_lines_reading_order(
        cls,
        items: list[tuple[str, float, list[int]]],
        language: SupportedLanguage,
        band_tol: int | None = None,
    ) -> list[tuple[str, float, list[int]]]:
        """
        Reading order: English/Hindi — top-to-bottom, then left-to-right.
        Arabic — column-aware RTL with dynamic band tolerance.

        band_tol is computed dynamically as 40% of median box height when not
        provided, falling back to _ARABIC_LINE_BAND_TOL_PX.
        """
        if not items:
            return []
        if language != SupportedLanguage.ARABIC:
            return sorted(items, key=lambda r: ((r[2][1] + r[2][3]) / 2, r[2][0]))

        tol = band_tol if band_tol is not None else max(
            _ARABIC_LINE_BAND_TOL_PX,
            int(cls._median_box_height(items) * 0.4),
        )

        def _bbox(it):
            return it[2]

        columns = cls._detect_columns(items, _bbox)
        out: list[tuple[str, float, list[int]]] = []
        for col_items in columns:
            by_y = sorted(col_items, key=lambda r: (r[2][1] + r[2][3]) / 2)
            bands: list[list[tuple[str, float, list[int]]]] = []
            cur: list[tuple[str, float, list[int]]] = [by_y[0]]
            ref_cy = (cur[0][2][1] + cur[0][2][3]) / 2
            for it in by_y[1:]:
                cy = (it[2][1] + it[2][3]) / 2
                if abs(cy - ref_cy) < tol:
                    cur.append(it)
                else:
                    bands.append(cur)
                    cur = [it]
                    ref_cy = cy
            bands.append(cur)
            for band in bands:
                out.extend(sorted(band, key=lambda r: r[2][0], reverse=True))
        return out

    @classmethod
    def _sort_f2_results_reading_order(
        cls,
        results: list[tuple[str, str, float, list[int]]],
        language: SupportedLanguage,
        band_tol: int | None = None,
    ) -> list[tuple[str, str, float, list[int]]]:
        """Same as _sort_lines_reading_order but for F2 (text_raw, text_corr, conf, bbox)."""
        if not results:
            return []
        if language != SupportedLanguage.ARABIC:
            return sorted(
                results,
                key=lambda r: ((r[3][1] + r[3][3]) / 2, r[3][0]),
            )

        # Derive items compatible with _sort_lines_reading_order helper.
        items_std = [(r[0], r[2], r[3]) for r in results]
        tol = band_tol if band_tol is not None else max(
            _ARABIC_LINE_BAND_TOL_PX,
            int(cls._median_box_height(items_std) * 0.4),
        )

        def _bbox(it):
            return it[3]

        columns = cls._detect_columns(results, _bbox)
        out: list[tuple[str, str, float, list[int]]] = []
        for col_items in columns:
            by_y = sorted(col_items, key=lambda r: (r[3][1] + r[3][3]) / 2)
            bands: list[list[tuple[str, str, float, list[int]]]] = []
            cur = [by_y[0]]
            ref_cy = (cur[0][3][1] + cur[0][3][3]) / 2
            for it in by_y[1:]:
                cy = (it[3][1] + it[3][3]) / 2
                if abs(cy - ref_cy) < tol:
                    cur.append(it)
                else:
                    bands.append(cur)
                    cur = [it]
                    ref_cy = cy
            bands.append(cur)
            for band in bands:
                out.extend(sorted(band, key=lambda r: r[3][0], reverse=True))
        return out

    def _score_pass(
        self,
        parsed: list[tuple[str, float, list[int]]],
        language: SupportedLanguage,
        variant_name: str | None = None,
    ) -> float:
        """
        Score an entire pass result.  Higher = better.

        Arabic-aware improvements over the naive sqrt(len) × mean_conf formula:

        1. Token count — uses genuine Arabic word tokens (≥2 base letters) rather
           than raw character count, so isolated-diacritic passes do not win.
        2. Ligature compensation — Arabic ligatures (ﷲ، لا، etc.) are single
           codepoints representing 2-4 chars.  We multiply token count × 4.5
           (avg chars/word) as a proxy for actual content length.
        3. Mixed-page awareness — invoices and forms legitimately mix Arabic with
           Latin/numerics.  When non-Arabic fraction exceeds the configured
           threshold we use the looser ratio gate (0.10) instead of 0.30/0.50.
        4. Diacritic-density gate — if a pass is >40% diacritics it is almost
           certainly a detection fragmentation artefact; penalise heavily.
        5. Bbox density check — if median box height < MIN_BOX_HEIGHT × 2 the
           detector fired on noise rows rather than text lines.
        """
        if not parsed:
            return 0.0

        total_text = " ".join(str(item[0]).strip() for item in parsed if str(item[0]).strip())
        if not total_text:
            return 0.0

        mean_conf = sum(float(item[1]) for item in parsed) / len(parsed)

        if language == SupportedLanguage.ARABIC:
            score_details = score_ocr_words([
                {
                    "text": text,
                    "confidence": conf,
                    "bbox": bbox,
                    "line_level": True,
                }
                for text, conf, bbox, *_meta in parsed
            ])
            score = float(score_details["score"])
            logger.debug(
                "[PaddleOCR F2] Arabic score details for %s: %s",
                variant_name,
                score_details,
            )
        else:
            total_len = len(total_text)
            score = math.sqrt(max(1.0, float(total_len))) * mean_conf

        if language != SupportedLanguage.ARABIC:
            # Upscale penalty — applies to the legacy non-Arabic variants.
            if variant_name == "upscaled_2x":
                score *= 0.88

            # Noise guard — repeated characters.
            if _RE_REPEATED.search(total_text):
                score *= 0.5

        return score

    def _select_best_pass(
        self,
        all_passes: list[tuple[str, str, list[tuple[str, float, list[int]]]]],
        language: SupportedLanguage,
    ) -> tuple[str, str, list[tuple[str, float, list[int]]], float]:
        """Pick the single pass whose full-page result is best."""
        best_pass = ("", "", [])
        best_score = -1.0
        for variant_name, engine_name, parsed in all_passes:
            s = self._score_pass(parsed, language, variant_name)
            logger.debug(
                "[PaddleOCR F2] Pass score: %s × %s → %.1f (%d lines)",
                variant_name, engine_name, s, len(parsed),
            )
            if s > best_score:
                best_score = s
                best_pass = (variant_name, engine_name, parsed)

        vname, ename, _ = best_pass
        logger.info(
            "[PaddleOCR F2] Stage 3: Best pass = %s × %s (score %.1f)",
            vname, ename, best_score,
        )
        return best_pass[0], best_pass[1], best_pass[2], best_score

    def _should_trigger_arabic_fallback(
        self,
        parsed: list[tuple[str, float, list[int]]],
    ) -> tuple[bool, list[str]]:
        """
        Decide whether to run the optional PP-OCRv3 fallback.

        Combines five signal categories:
          1. Coverage  — line count, total character count
          2. Quality   — average confidence, Arabic ratio
          3. Noise     — repeated character patterns
          4. Density   — median bbox height (sub-pixel rows are noise, not text)
          5. Structure — vertical IoU between adjacent boxes (diacritic
             fragmentation where OCR splits a single line into many bbox pairs
             that heavily overlap vertically)
        """
        reasons: list[str] = []
        if not parsed:
            return True, ["empty_primary_result"]

        line_count  = len(parsed)
        total_text  = " ".join(str(item[0]).strip() for item in parsed if str(item[0]).strip())
        total_chars = len(total_text)
        avg_conf    = sum(float(item[1]) for item in parsed) / line_count if line_count else 0.0
        ar_ratio    = self._arabic_char_ratio(total_text) if total_text else 0.0

        # --- 1. Coverage checks ---
        if line_count < self.fallback_min_lines:
            reasons.append(f"low_line_count<{self.fallback_min_lines}")
        if total_chars < self.fallback_min_chars:
            reasons.append(f"low_char_count<{self.fallback_min_chars}")

        # --- 2. Quality checks ---
        if avg_conf < self.fallback_min_avg_conf:
            reasons.append(f"low_avg_conf<{self.fallback_min_avg_conf:.2f}")
        if ar_ratio < self.fallback_min_ar_ratio:
            reasons.append(f"low_arabic_ratio<{self.fallback_min_ar_ratio:.2f}")

        # --- 3. Noise guard ---
        if _RE_REPEATED.search(total_text):
            reasons.append("repeated_char_pattern")

        # --- 4. Bbox height density ---
        med_h = self._median_box_height(parsed)
        if med_h < self.MIN_BOX_HEIGHT * 2:
            reasons.append(f"low_median_box_height<{self.MIN_BOX_HEIGHT * 2}px(got {med_h:.1f}px)")

        # --- 5. Vertical IoU fragmentation ---
        # Sort boxes by top-y and check consecutive pairs for heavy vertical overlap.
        # >20% of adjacent pairs overlapping by >30% of the shorter box height
        # indicates the detector fragmented lines rather than detecting them whole.
        if line_count >= 4:
            sorted_boxes = sorted(parsed, key=lambda r: r[2][1])
            overlap_pairs = 0
            for i in range(len(sorted_boxes) - 1):
                _, _, b1 = sorted_boxes[i]
                _, _, b2 = sorted_boxes[i + 1]
                # Vertical intersection
                inter_top    = max(b1[1], b2[1])
                inter_bottom = min(b1[3], b2[3])
                if inter_bottom <= inter_top:
                    continue
                inter_h = inter_bottom - inter_top
                min_h   = min(b1[3] - b1[1], b2[3] - b2[1])
                if min_h > 0 and inter_h / min_h > 0.30:
                    overlap_pairs += 1
            overlap_ratio = overlap_pairs / (line_count - 1)
            if overlap_ratio > 0.20:
                reasons.append(
                    f"bbox_vertical_overlap_fragmentation({overlap_pairs}/{line_count - 1}pairs)"
                )

        return (len(reasons) > 0), reasons

    # ------------------------------------------------------------------
    # Stage 5 — Arabic text correction pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _to_logical_order(text: str) -> str:
        """
        Return Arabic text unchanged.

        PaddleOCR Arabic recognition already emits Unicode *logical* order (typing
        order).  ``bidi.algorithm.get_display()`` does the opposite — it builds a
        *visual* string for LTR display — and was incorrectly used here, which
        reversed or mangled ``primary_response_corrected_text`` relative to raw OCR.

        True visual→logical repair would need a dedicated algorithm; until then,
        API consumers should use the corrected string as-is.  For *drawing* on an
        LTR canvas, use reshape + get_display in the visualization path only
        (see ``_pil_draw_arabic``), not in stored OCR results.
        """
        return text

    def _arabic_correct(self, text: str) -> str:
        """
        Full Arabic post-OCR correction pipeline applied in Stage 5.

        Steps (in order):
          1. NFC normalization  — canonical decomposition + canonical composition.
          2. Invisible char removal — ZWNJ, ZWJ, zero-width space, LRM, RLM, BOM.
          3. NFKC normalization  — resolves Arabic Presentation Forms (ﻛ → ك).
          4. Kashida (tatweel) removal — OCR hallucinates U+0640 runs; strip them.
          5. Repeated diacritic deduplication — e.g. ً ً → ً.
          6. Punctuation normalization — Western , ; ? → Arabic ، ؛ ؟.
          7. Alef normalization (opt-in) — أ إ آ ٱ → ا; useful for search/NLP
             but loses orthographic fidelity; off by default.
          8. Isolated single-letter token filtering (opt-in) — standalone Arabic
             letters surrounded by whitespace are almost always detector artefacts.
          9. BiDi hook (legacy, no-op) — ``arabic_normalize_bidi`` is retained for
             compatibility; see ``_to_logical_order``.
        """
        if not text or not text.strip():
            return text

        # 1. NFC
        text = unicodedata.normalize("NFC", text)

        # 2. Invisible chars
        text = text.translate(_ARABIC_CLEANUP_MAP)

        # 3. NFKC — resolves presentation forms (ﻛ → ك, ﷲ stays as is)
        text = unicodedata.normalize("NFKC", text)

        # 4. Kashida removal — strip all tatweel runs.
        text = _RE_KASHIDA.sub("", text)

        # 5. Repeated diacritic deduplication.
        text = _RE_REPEAT_DIACRITIC.sub(r"\1", text)

        # 6. Punctuation normalization: Western → Arabic.
        #    Only replace when the character is adjacent to Arabic context to
        #    avoid corrupting embedded URLs or code snippets.
        _PUNCT_REPL = {",": "\u060C", ";": "\u061B", "?": "\u061F"}
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch in _PUNCT_REPL:
                prev_ar = idx > 0 and any(
                    lo <= chars[idx - 1] <= hi for lo, hi in _ARABIC_BASE_RANGES
                )
                next_ar = idx < len(chars) - 1 and any(
                    lo <= chars[idx + 1] <= hi for lo, hi in _ARABIC_BASE_RANGES
                )
                if prev_ar or next_ar:
                    chars[idx] = _PUNCT_REPL[ch]
        text = "".join(chars)

        # 7. Alef normalization (opt-in).
        if self.arabic_normalize_alef:
            text = text.translate(_ALEF_VARIANTS)

        # 8. Isolated single-letter filtering (opt-in).
        if self.arabic_filter_isolated_letters:
            tokens = text.split()
            filtered = [
                tok for tok in tokens
                if not _RE_ISOLATED_AR_LETTER.match(tok)
            ]
            # Only apply if filtering removed ≤20% of tokens to avoid destroying
            # legitimate single-letter tokens (e.g. Arabic conjunctions ف، و، ب).
            if tokens and len(filtered) / len(tokens) >= 0.80:
                text = " ".join(filtered)

        # 9. BiDi logical-order normalization (opt-in).
        if self.arabic_normalize_bidi:
            text = self._to_logical_order(text)

        return text.strip()

    @staticmethod
    def _bbox_iou(a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = float((ix2 - ix1) * (iy2 - iy1))
        area_a = max(1.0, float((ax2 - ax1) * (ay2 - ay1)))
        area_b = max(1.0, float((bx2 - bx1) * (by2 - by1)))
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _bbox_y_overlap_ratio(a: list[int], b: list[int]) -> float:
        ay1, ay2 = float(a[1]), float(a[3])
        by1, by2 = float(b[1]), float(b[3])
        inter = max(0.0, min(ay2, by2) - max(ay1, by1))
        denom = max(1.0, min(ay2 - ay1, by2 - by1))
        return inter / denom

    @staticmethod
    def _line_text_similarity(a: str, b: str) -> float:
        def _norm(text: str) -> str:
            return re.sub(r"\s+", "", text or "")

        return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

    @staticmethod
    def _clamp_bbox_to_image(bbox: list[int], img_shape: tuple[int, ...]) -> list[int]:
        h, w = img_shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(x1, max(0, w - 1)))
        y1 = max(0, min(y1, max(0, h - 1)))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        return [x1, y1, x2, y2]

    def _line_with_bbox_source(
        self,
        text: str,
        conf: float,
        bbox: list[int],
        *,
        variant_name: str,
        scale: float,
        effective_scale: float,
        original_bbox_before_rescale: list[int],
        img_shape: tuple[int, ...],
        engine_name: str | None = None,
    ) -> tuple[str, float, list[int], dict]:
        normalized = self._clamp_bbox_to_image(bbox, img_shape)
        meta = {
            "variant_name": variant_name,
            "scale": float(scale),
            "effective_scale": float(effective_scale),
            "original_bbox_before_rescale": [
                int(round(float(v))) for v in original_bbox_before_rescale
            ],
            "bbox_after_rescale": normalized,
            "engine_name": engine_name,
        }
        return (text, conf, normalized, meta)

    def _validate_bbox_lines(
        self,
        parsed: list[tuple],
        img_shape: tuple[int, ...],
    ) -> tuple[list[tuple], list[dict], list[dict], float]:
        if not parsed:
            return [], [], [], 0.0

        h, w = img_shape[:2]
        clamped: list[tuple] = []
        corrected: list[dict] = []
        for item in parsed:
            text, conf, bbox = item[:3]
            meta = dict(item[3]) if len(item) > 3 and isinstance(item[3], dict) else {}
            new_bbox = self._clamp_bbox_to_image(bbox, img_shape)
            if list(bbox) != new_bbox:
                corrected.append({
                    "text": text,
                    "confidence": round(float(conf), 4),
                    "old_bbox": list(bbox),
                    "new_bbox": new_bbox,
                    "reason": "clamped_to_image_bounds",
                    "bbox_source": meta,
                })
            meta["bbox_after_validation_clamp"] = new_bbox
            clamped.append((text, conf, new_bbox, meta))

        heights = [
            float(item[2][3] - item[2][1])
            for item in clamped
            if item[2][2] > item[2][0] and item[2][3] > item[2][1]
        ]
        median_h = statistics.median(heights) if heights else float(self.MIN_BOX_HEIGHT)

        valid: list[tuple] = []
        rejected: list[dict] = []
        for item in clamped:
            text, conf, bbox, meta = item
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            reasons: list[str] = []
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
                reasons.append("outside_image_bounds")
            if bw <= 10:
                reasons.append("width<=10")
            if bh <= 8:
                reasons.append("height<=8")
            if median_h > 0 and bh > 3.0 * median_h:
                reasons.append("height>3x_median")
            if bw > 1.2 * w:
                reasons.append("width>1.2x_page_width")

            if reasons:
                rejected.append({
                    "text": text,
                    "confidence": round(float(conf), 4),
                    "bbox": bbox,
                    "reasons": reasons,
                    "bbox_source": meta,
                })
                if float(conf) >= 0.85:
                    meta["bbox_validated"] = False
                    meta["bbox_reject_reasons"] = reasons
                    valid.append((text, conf, bbox, meta))
                continue
            meta["bbox_validated"] = True
            valid.append((text, conf, bbox, meta))

        return valid, rejected, corrected, float(median_h)

    def _resolve_bbox_disagreements(
        self,
        selected: list[tuple],
        all_passes: list[tuple[str, str, list[tuple]]],
        img_shape: tuple[int, ...],
        confidence_close_margin: float = 0.05,
    ) -> tuple[list[tuple], list[dict]]:
        if not selected:
            return selected, []

        all_lines = [
            line
            for _variant_name, _engine_name, parsed in all_passes
            for line in parsed
            if len(line) >= 4 and line[2]
        ]
        corrected: list[dict] = []
        resolved: list[tuple] = []

        for line in selected:
            text, conf, bbox = line[:3]
            meta = dict(line[3]) if len(line) > 3 and isinstance(line[3], dict) else {}
            matches = [
                other for other in all_lines
                if self._bbox_y_overlap_ratio(bbox, other[2]) >= 0.35
                and self._line_text_similarity(text, other[0]) >= 0.72
            ]
            if len(matches) <= 1:
                resolved.append((text, conf, bbox, meta))
                continue

            original_candidates = [
                other for other in matches
                if isinstance(other[3], dict)
                and other[3].get("variant_name") == "original"
                and float(other[1]) >= float(conf) - confidence_close_margin
            ]
            if original_candidates:
                chosen = max(original_candidates, key=lambda item: float(item[1]))
                reason = "original_variant_confidence_close"
            else:
                ordered = sorted(matches, key=lambda item: (item[2][1] + item[2][3]) / 2.0)
                chosen = ordered[len(ordered) // 2]
                reason = "median_y_center_across_variants"

            chosen_bbox = self._clamp_bbox_to_image(chosen[2], img_shape)
            chosen_meta = dict(chosen[3]) if len(chosen) > 3 and isinstance(chosen[3], dict) else {}
            if chosen_bbox != bbox:
                corrected.append({
                    "text": text,
                    "confidence": round(float(conf), 4),
                    "old_bbox": bbox,
                    "new_bbox": chosen_bbox,
                    "reason": reason,
                    "matched_variant_count": len(matches),
                    "bbox_source": chosen_meta,
                })
            merged_meta = dict(meta)
            merged_meta["bbox_resolution"] = {
                "reason": reason,
                "matched_variant_count": len(matches),
                "chosen_source": chosen_meta,
            }
            resolved.append((text, conf, chosen_bbox, merged_meta))

        return resolved, corrected

    @staticmethod
    def _padded_bbox(
        bbox: list[int],
        img_shape: tuple[int, ...],
        pad_ratio: float = 0.45,
    ) -> list[int]:
        h, w = img_shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = max(8, int(bw * 0.10))
        pad_y = max(8, int(bh * pad_ratio))
        return [
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w, x2 + pad_x),
            min(h, y2 + pad_y),
        ]

    def _ocr_region(
        self,
        engine: object,
        img_rgb: np.ndarray,
        crop_bbox: list[int],
        upscale: float = 1.8,
    ) -> list[tuple[str, float, list[int]]]:
        x1, y1, x2, y2 = crop_bbox
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        scaled = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * upscale)), max(1, int(crop.shape[0] * upscale))),
            interpolation=cv2.INTER_CUBIC,
        )
        result = engine.ocr(scaled)  # type: ignore[attr-defined]
        parsed = self._parse_ocr_page(result[0]) if result and result[0] else []
        if not parsed:
            return []
        inv = 1.0 / upscale
        return [
            (
                text,
                conf,
                [
                    int(bbox[0] * inv) + x1,
                    int(bbox[1] * inv) + y1,
                    int(bbox[2] * inv) + x1,
                    int(bbox[3] * inv) + y1,
                ],
            )
            for text, conf, bbox in parsed
        ]

    def _recover_top_band_lines(
        self,
        img_rgb: np.ndarray,
        parsed: list[tuple[str, float, list[int]]],
        engine: object,
        trigger_ratio: float = 0.15,
        band_ratio: float = 0.22,
    ) -> tuple[list[tuple[str, float, list[int]]], dict]:
        stats = {"attempted": False, "added": 0, "reason": ""}
        if not parsed:
            stats["reason"] = "no_existing_lines"
            return parsed, stats
        h, w = img_rgb.shape[:2]
        first_top = min(item[2][1] for item in parsed)
        if first_top <= h * trigger_ratio:
            stats["reason"] = "first_line_near_top"
            return parsed, stats

        stats["attempted"] = True
        band_h = max(32, int(h * band_ratio))
        try:
            recovered = self._ocr_region(engine, img_rgb, [0, 0, w, band_h], upscale=1.6)
        except Exception as exc:
            stats["reason"] = f"ocr_failed:{exc}"
            return parsed, stats

        additions: list[tuple[str, float, list[int]]] = []
        for text, conf, bbox in recovered:
            if not text.strip() or conf < 0.30:
                continue
            if bbox[1] >= first_top:
                continue
            if any(self._bbox_iou(bbox, old_item[2]) > 0.25 for old_item in parsed):
                continue
            additions.append(self._line_with_bbox_source(
                text,
                conf,
                bbox,
                variant_name="top_band_recovery",
                scale=1.0,
                effective_scale=1.0,
                original_bbox_before_rescale=bbox,
                img_shape=img_rgb.shape,
                engine_name="region_ocr",
            ))

        stats["added"] = len(additions)
        stats["reason"] = "ok" if additions else "no_new_top_lines"
        return additions + parsed, stats

    def _refine_low_confidence_regions(
        self,
        img_rgb: np.ndarray,
        parsed: list[tuple[str, float, list[int]]],
        engine: object,
        confidence_threshold: float = 0.80,
        replace_margin: float = 0.03,
        max_regions: int = 12,
    ) -> tuple[list[tuple[str, float, list[int]]], dict]:
        stats = {"attempted": 0, "replaced": 0, "failed": 0}
        if not parsed:
            return parsed, stats

        refined = list(parsed)
        candidates = [
            (idx, item)
            for idx, item in enumerate(refined)
            if item[1] < confidence_threshold and item[2][2] > item[2][0] and item[2][3] > item[2][1]
        ][:max_regions]

        for idx, old_item in candidates:
            old_text, old_conf, old_bbox = old_item[:3]
            crop_bbox = self._padded_bbox(old_bbox, img_rgb.shape)
            stats["attempted"] += 1
            try:
                crop_lines = self._ocr_region(engine, img_rgb, crop_bbox, upscale=1.8)
            except Exception:
                stats["failed"] += 1
                continue
            if not crop_lines:
                continue
            best = max(
                crop_lines,
                key=lambda item: (self._bbox_iou(item[2], old_bbox), item[1]),
            )
            new_text, new_conf, new_bbox = best[:3]
            if not new_text.strip():
                continue
            if new_conf <= old_conf + replace_margin:
                continue
            if self._arabic_char_ratio(new_text) < max(0.10, self._arabic_char_ratio(old_text) * 0.70):
                continue
            old_meta = dict(old_item[3]) if len(old_item) > 3 and isinstance(old_item[3], dict) else {}
            old_meta["bbox_refinement"] = {
                "old_text": old_text,
                "old_confidence": round(float(old_conf), 4),
                "old_bbox": old_bbox,
            }
            refined[idx] = (new_text, new_conf, new_bbox, old_meta)
            stats["replaced"] += 1

        return refined, stats

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_draw_arabic(
        img_bgr: np.ndarray, text: str, xy: tuple[int, int],
        font_size: int = 18, color: tuple[int, int, int] = (255, 255, 0),
    ) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        draw_text = text
        try:
            import arabic_reshaper  # type: ignore[import-untyped]
            from bidi.algorithm import get_display  # type: ignore[import-untyped]

            if any("\u0600" <= c <= "\u06FF" for c in text):
                draw_text = get_display(arabic_reshaper.reshape(text))
        except Exception:
            pass

        font = ImageFont.load_default()
        for fp in (
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
            "/usr/share/fonts/truetype/arabic/NotoNaskhArabic-Regular.ttf",
            "/usr/share/fonts/truetype/fonts-arabeyes/ae_AlArabiya.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue

        draw.text(xy, draw_text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _get_debug_dir(self) -> Path | None:
        if not self.debug_output_dir:
            return None
        raw_dir = Path(self.debug_output_dir)
        if not raw_dir.is_absolute():
            raw_dir = Path.cwd() / raw_dir
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            return raw_dir
        except Exception:
            fallback = Path.cwd() / "debug_output"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _visualize_f2(
        self,
        img_rgb: np.ndarray,
        variants: list[tuple[str, float, np.ndarray]],
        results: list[tuple[str, str, float, list[int]]],
        run_id: str,
    ) -> dict[str, str]:
        out_dir = self._get_debug_dir()
        if not out_dir:
            return {}
        saved: dict[str, str] = {}

        # Stage 1: variant thumbnails grid
        THUMB_W, THUMB_H = 320, 200
        cells: list[np.ndarray] = []
        for vname, _scale, vimg in variants:
            thumb = cv2.resize(
                cv2.cvtColor(vimg, cv2.COLOR_RGB2BGR),
                (THUMB_W, THUMB_H), interpolation=cv2.INTER_LINEAR,
            )
            label_bar = np.zeros((25, THUMB_W, 3), dtype=np.uint8)
            cv2.putText(label_bar, vname, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cells.append(np.vstack([label_bar, thumb]))

        while len(cells) % 3:
            cells.append(np.zeros_like(cells[0]))
        rows = [np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]
        grid = np.vstack(rows)
        p1 = str(out_dir / f"{run_id}_stage1_variants.jpg")
        cv2.imwrite(p1, grid)
        saved["stage1_variants"] = p1

        # Stage 4: final result overlay
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        for text_raw, text_corrected, conf, (x1, y1, x2, y2), *_rest in results:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"[{conf:.2f}] {text_corrected[:40]}"
            vis = self._pil_draw_arabic(vis, label, (x1, max(y1 - 22, 2)), font_size=15)
        p4 = str(out_dir / f"{run_id}_stage4_results.jpg")
        cv2.imwrite(p4, vis)
        saved["stage4_results"] = p4

        logger.info("[PaddleOCR F2 VIS] Saved → %s", out_dir)
        return saved

    # ------------------------------------------------------------------
    # F2 pipeline orchestrator
    # ------------------------------------------------------------------

    def _run_f2_pipeline(
        self,
        img_rgb: np.ndarray,
        lang_enum: SupportedLanguage,
        debug_run_id: str | None = None,
    ) -> tuple[list[tuple[str, str, float, list[int]]], dict[str, str], dict]:
        """
        Full-page multi-pass F2 pipeline.

        Runs OCR on multiple preprocessed page variants × engines, then
        picks the single best *complete pass* (not a line-level merge)
        so that the output preserves full context.

        Returns:
          (results, debug_paths)
          results     — list of (text_raw, text_corrected, confidence, bbox)
          debug_paths — dict of saved visualisation file paths
        """
        diagnostics: dict = {
            "selected_variant": None,
            "selected_engine": None,
            "primary_best_score": None,
            "alt_best_score": None,
            "comparison_mode": "primary_only",
            "fallback_triggered": False,
            "fallback_reasons": [],
            "primary_response_raw_text": "",
            "primary_response_corrected_text": "",
            "alt_response_raw_text": "",
            "alt_response_corrected_text": "",
            "variant_scores": [],
            "confidence_score": 0.0,
            "rejected_bbox_lines": [],
            "corrected_bbox_lines": [],
        }
        self._bbox_sanity_current = {
            "rejected_bbox_lines": diagnostics["rejected_bbox_lines"],
            "corrected_bbox_lines": diagnostics["corrected_bbox_lines"],
            "median_line_height": 0.0,
        }

        # Stage 1+2 (primary): generate variants, run primary engine only
        try:
            primary_passes = self._run_all_passes(
                img_rgb, lang_enum, include_primary=True, include_alt=False
            )
            if not primary_passes:
                return [], {}, diagnostics
        finally:
            bbox_sanity = getattr(self, "_bbox_sanity_current", None)
            if isinstance(bbox_sanity, dict):
                diagnostics["bbox_median_line_height"] = bbox_sanity.get("median_line_height", 0.0)
        diagnostics["variant_scores"].extend([
            {
                "variant": variant_name,
                "engine": engine_name,
                "score": round(self._score_pass(parsed, lang_enum, variant_name), 4),
                "item_count": len(parsed),
            }
            for variant_name, engine_name, parsed in primary_passes
        ])

        # Stage 3: pick the best complete primary pass
        best_variant, best_engine, best_parsed, best_score = self._select_best_pass(
            primary_passes, lang_enum
        )
        diagnostics["primary_best_score"] = round(best_score, 3)

        if lang_enum == SupportedLanguage.ARABIC:
            primary_engine = self._get_or_load_primary_engine(lang_enum)
            if primary_engine is not None:
                best_parsed, top_recovery_stats = self._recover_top_band_lines(
                    img_rgb, best_parsed, primary_engine
                )
                best_parsed, roi_refine_stats = self._refine_low_confidence_regions(
                    img_rgb, best_parsed, primary_engine
                )
                diagnostics["top_band_recovery"] = top_recovery_stats
                diagnostics["roi_upscale_refinement"] = roi_refine_stats

        primary_sorted = self._sort_lines_reading_order(best_parsed, lang_enum)
        diagnostics["primary_response_raw_text"] = "\n".join(
            str(item[0]) for item in primary_sorted if item[0] and str(item[0]).strip()
        )
        diagnostics["primary_response_corrected_text"] = "\n".join(
            (
                self._arabic_correct(str(item[0]))
                if lang_enum == SupportedLanguage.ARABIC
                else str(item[0])
            )
            for item in primary_sorted
            if item[0] and str(item[0]).strip()
        )

        all_pass_count = len(primary_passes)

        # Optional Arabic v3 execution modes:
        # 1) always_run_both_arabic_engines=True  -> always compare v5 vs v3
        # 2) enable_arabic_v3_fallback=True       -> run v3 only on weak v5 signal
        alt_engines_available = (
            bool(_ALT_ENGINES.get(lang_enum))
            and (self.enable_arabic_v3_fallback or self.always_run_both_arabic_engines)
        )
        if lang_enum == SupportedLanguage.ARABIC and alt_engines_available:
            should_run_alt = False
            reasons: list[str] = []
            if self.always_run_both_arabic_engines:
                should_run_alt = True
                diagnostics["comparison_mode"] = "always_compare_v5_v3"
                reasons = ["always_compare_enabled"]
            elif self.enable_arabic_v3_fallback:
                # When fallback mode is enabled, always run v3 too so user can
                # benchmark both outputs manually from metadata.
                diagnostics["comparison_mode"] = "fallback_enabled_with_dual_output"
                should_run_alt = True
                trigger, reasons = self._should_trigger_arabic_fallback(best_parsed)
                diagnostics["fallback_triggered"] = trigger
                diagnostics["fallback_reasons"] = reasons

            if should_run_alt:
                logger.info(
                    "[PaddleOCR F2] Running alt engine passes (%s).",
                    ", ".join(reasons),
                )
                alt_passes = self._run_all_passes(
                    img_rgb, lang_enum, include_primary=False, include_alt=True
                )
                all_pass_count += len(alt_passes)
                if alt_passes:
                    diagnostics["variant_scores"].extend([
                        {
                            "variant": variant_name,
                            "engine": engine_name,
                            "score": round(self._score_pass(parsed, lang_enum, variant_name), 4),
                            "item_count": len(parsed),
                        }
                        for variant_name, engine_name, parsed in alt_passes
                    ])
                    alt_variant, alt_engine, alt_parsed, alt_score = self._select_best_pass(
                        alt_passes, lang_enum
                    )
                    diagnostics["alt_best_score"] = round(alt_score, 3)
                    alt_sorted = self._sort_lines_reading_order(alt_parsed, lang_enum)
                    diagnostics["alt_response_raw_text"] = "\n".join(
                        str(item[0]) for item in alt_sorted if item[0] and str(item[0]).strip()
                    )
                    diagnostics["alt_response_corrected_text"] = "\n".join(
                        self._arabic_correct(str(item[0]))
                        for item in alt_sorted
                        if item[0] and str(item[0]).strip()
                    )
                    if alt_score > best_score * (1.0 + self.fallback_replace_margin):
                        logger.info(
                            "[PaddleOCR F2] Winner selected from alt engine: %s × %s (%.1f > %.1f)",
                            alt_variant, alt_engine, alt_score, best_score,
                        )
                        best_variant, best_engine, best_parsed, best_score = (
                            alt_variant,
                            alt_engine,
                            alt_parsed,
                            alt_score,
                        )
                    else:
                        logger.info(
                            "[PaddleOCR F2] Keeping primary winner (%s × %s). Alt score %.1f not above margin.",
                            best_variant, best_engine, alt_score
                        )

        all_candidate_passes = list(primary_passes)
        if lang_enum == SupportedLanguage.ARABIC and diagnostics.get("alt_response_raw_text"):
            # alt_passes is only defined in the branch above; collect it when present.
            try:
                all_candidate_passes.extend(alt_passes)  # type: ignore[name-defined]
            except NameError:
                pass

        if lang_enum == SupportedLanguage.ARABIC:
            best_parsed, bbox_resolution_corrections = self._resolve_bbox_disagreements(
                best_parsed,
                all_candidate_passes,
                img_rgb.shape,
            )
            diagnostics["corrected_bbox_lines"].extend(bbox_resolution_corrections)

        # Stage 4+5: apply Arabic correction to the winning pass
        results: list[tuple[str, str, float, list[int]]] = []
        for text_raw, conf, bbox, *rest in best_parsed:
            if not text_raw or not text_raw.strip():
                continue
            bbox_meta = rest[0] if rest and isinstance(rest[0], dict) else {}
            text_corrected = (
                self._arabic_correct(text_raw)
                if lang_enum == SupportedLanguage.ARABIC
                else text_raw
            )
            if text_corrected:
                results.append((text_raw, text_corrected, conf, bbox, bbox_meta))

        results = self._sort_f2_results_reading_order(results, lang_enum)

        logger.info(
            "[PaddleOCR F2] Pipeline done: %d lines (best pass: %s × %s, from %d total passes)",
            len(results), best_variant, best_engine, all_pass_count,
        )
        diagnostics["selected_variant"] = best_variant
        diagnostics["selected_engine"] = best_engine
        diagnostics["confidence_score"] = round(best_score, 4)
        logger.info(
            "[PaddleOCR F2] Selected response source: engine=%s variant=%s mode=%s primary_score=%s alt_score=%s",
            diagnostics["selected_engine"],
            diagnostics["selected_variant"],
            diagnostics["comparison_mode"],
            diagnostics["primary_best_score"],
            diagnostics["alt_best_score"],
        )

        # Debug visualisation
        debug_paths: dict[str, str] = {}
        if self.debug_output_dir and debug_run_id:
            try:
                variants = self._generate_page_variants(img_rgb, language=lang_enum)
                debug_paths = self._visualize_f2(img_rgb, variants, results, debug_run_id)
            except Exception as exc:
                logger.warning("[PaddleOCR F2 VIS] Visualisation failed: %s", exc)

        if hasattr(self, "_bbox_sanity_current"):
            delattr(self, "_bbox_sanity_current")

        return results, debug_paths, diagnostics

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _cap_image_side_len(
        self, img_rgb: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Pre-resize *img_rgb* so that its longest side does not exceed
        ``self.det_limit_side_len``.

        This is necessary because PP-OCRv5 uses PaddleX internally, which
        applies its own ``max_side_limit`` cap (default 4000 px) **before**
        the ``det_limit_side_len`` resize step.  By resizing here we ensure
        the image never reaches that internal cap and the log message
        "Resized image size … exceeds max_side_limit of 4000" is suppressed.

        Returns ``(resized_image, scale_factor)`` where ``scale_factor < 1``
        when a resize was applied, or ``(img_rgb, 1.0)`` when the image
        already fits within the limit.
        """
        if self.det_limit_side_len is None:
            return img_rgb, 1.0
        h, w = img_rgb.shape[:2]
        max_side = max(h, w)
        if max_side <= self.det_limit_side_len:
            return img_rgb, 1.0
        scale = self.det_limit_side_len / float(max_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(
            "[PaddleOCR] Pre-resized image %dx%d → %dx%d (det_limit_side_len=%d)",
            w, h, new_w, new_h, self.det_limit_side_len,
        )
        return resized, scale

    @staticmethod
    def _to_xyxy_bbox(poly) -> list[int]:
        if poly is None:
            return [0, 0, 0, 0]
        try:
            if len(poly) == 0:
                return [0, 0, 0, 0]
        except TypeError:
            return [0, 0, 0, 0]
        xs = [int(float(p[0])) for p in poly]
        ys = [int(float(p[1])) for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]

    MIN_BOX_AREA = 150      # px² — smaller boxes are almost always noise
    MIN_BOX_HEIGHT = 8      # px  — shorter than this is grid lines / artifacts

    def _parse_ocr_page(self, page_result) -> list[tuple[str, float, list[int]]]:
        raw: list[tuple[str, float, list[int]]] = []
        if isinstance(page_result, dict):
            texts  = page_result.get("rec_texts") or []
            scores = page_result.get("rec_scores") or []
            polys  = page_result.get("dt_polys") or page_result.get("rec_polys") or []
            for i in range(min(len(texts), len(scores), len(polys))):
                raw.append((str(texts[i]), float(scores[i]), self._to_xyxy_bbox(polys[i])))
        elif isinstance(page_result, list):
            for line in page_result:
                if not (isinstance(line, (list, tuple)) and len(line) >= 2):
                    continue
                bbox, text_conf = line[0], line[1]
                if not (isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2):
                    continue
                raw.append((str(text_conf[0]), float(text_conf[1]), self._to_xyxy_bbox(bbox)))

        parsed: list[tuple[str, float, list[int]]] = []
        for text, conf, (x1, y1, x2, y2) in raw:
            w, h = x2 - x1, y2 - y1
            if w * h < self.MIN_BOX_AREA or h < self.MIN_BOX_HEIGHT:
                continue
            if not text or not text.strip():
                continue
            parsed.append((text, conf, [x1, y1, x2, y2]))
        return parsed

    @staticmethod
    def _page_score(parsed_page: list[tuple[str, float, list[int]]]) -> float:
        return sum(float(item[1]) for item in parsed_page) if parsed_page else 0.0

    def _select_all_language_page(
        self, img_array: np.ndarray
    ) -> tuple[SupportedLanguage | None, list[tuple[str, float, list[int]]], float]:
        # Pre-resize before passing to any engine so PaddleX's internal
        # max_side_limit (4000 px) is never triggered.
        capped_img, cap_scale = self._cap_image_side_len(img_array)
        best_lang: SupportedLanguage | None = None
        best_page: list[tuple[str, float, list[int]]] = []
        best_score = -1.0
        total_elapsed = 0.0
        for lang_enum in LANG_CONFIG:
            engine = self._get_or_load_primary_engine(lang_enum)
            if not engine:
                continue
            t0 = self._timer()
            result = engine.ocr(capped_img)
            total_elapsed += self._elapsed_ms(t0)
            parsed = self._parse_ocr_page(result[0]) if result and result[0] else []
            inv = 1.0 / cap_scale if cap_scale else 1.0
            parsed = [
                self._line_with_bbox_source(
                    text,
                    conf,
                    [bbox[0] * inv, bbox[1] * inv, bbox[2] * inv, bbox[3] * inv],
                    variant_name="pre_resized" if cap_scale != 1.0 else "original",
                    scale=cap_scale,
                    effective_scale=cap_scale,
                    original_bbox_before_rescale=bbox,
                    img_shape=img_array.shape,
                    engine_name=LANG_CONFIG[lang_enum]["ocr_version"],
                )
                for text, conf, bbox in parsed
            ]
            parsed, _rejected, _corrected, _median_h = self._validate_bbox_lines(
                parsed,
                img_array.shape,
            )
            score = self._page_score(parsed)
            if score > best_score:
                best_score, best_lang, best_page = score, lang_enum, parsed
        return best_lang, best_page, total_elapsed

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, image_bytes: bytes, language: SupportedLanguage) -> OCRResult:
        if language != SupportedLanguage.ALL:
            engine = self._get_or_load_primary_engine(language)
            if not engine:
                return OCRResult.from_error(self.name, language.value, f"Language {language} not loaded")
        else:
            engine = None

        try:
            words:          list[OCRWord] = []
            page_texts:     list[str]     = []
            page_texts_raw: list[str]     = []
            total_elapsed   = 0.0
            pages           = load_document_as_rgb_images(image_bytes)
            resolved_page_languages: list[str | None] = []

            f2_active = (
                self.max_accuracy
                and language != SupportedLanguage.ALL
            )

            run_id    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            all_debug: dict[str, str] = {}
            f2_page_diagnostics: list[dict] = []
            input_pipeline_pages: list[dict] = []
            structured_pages: list[dict] = []
            rejected_bbox_lines: list[dict] = []
            corrected_bbox_lines: list[dict] = []

            for page_idx, image in enumerate(pages):
                img_array = np.array(image)
                img_array, prep_meta = self._apply_document_roi_warp(img_array, language)
                input_pipeline_pages.append(prep_meta)
                page_lines: list[dict] = []
                page_flagged_lines: list[dict] = []
                page_raw_lines: list[str] = []

                def append_page_line(
                    text: str,
                    raw_text: str,
                    conf: float,
                    flat_bbox: list[int],
                    line_language: SupportedLanguage | None,
                    bbox_source: dict | None = None,
                    bbox_valid: bool = True,
                ) -> None:
                    page_raw_lines.append(raw_text)
                    line_index = len(page_raw_lines) - 1
                    line_meta = {
                        "page_index": page_idx,
                        "line_index": line_index,
                        "text": text,
                        "raw_text": raw_text,
                        "confidence": round(float(conf), 4),
                        "bbox": flat_bbox,
                        "line_level": True,
                        "bbox_source": bbox_source or {},
                        "bbox_valid": bbox_valid,
                    }
                    if not bbox_valid:
                        line_meta["status"] = "review_line"
                        line_meta["review_reason"] = "invalid_bbox"
                    noise_reason = (
                        self._arabic_noise_line_reason(text, conf)
                        if line_language == SupportedLanguage.ARABIC
                        else None
                    )
                    if noise_reason:
                        line_meta["filter_reason"] = noise_reason
                        page_flagged_lines.append(line_meta.copy())
                    token_items = split_line_tokens(
                        text,
                        conf,
                        flat_bbox,
                        synthetic_line_tokens=True,
                    )
                    line_meta["words"] = [
                        {
                            "text": token.get("text", ""),
                            "confidence": round(float(token.get("confidence", 0.0)), 4),
                            "bbox": token.get("bbox"),
                        }
                        for token in token_items
                    ]
                    for token in token_items or [{
                        "text": text,
                        "confidence": conf,
                        "bbox": flat_bbox,
                    }]:
                        words.append(OCRWord(
                            text=str(token.get("text", "")),
                            confidence=float(token.get("confidence", conf)),
                            bbox=token.get("bbox"),
                        ))
                    lines.append(text)
                    lines_raw.append(raw_text)
                    page_lines.append(line_meta)

                if language == SupportedLanguage.ALL:
                    page_lang, parsed_std, page_elapsed = self._select_all_language_page(img_array)
                    total_elapsed += page_elapsed
                    resolved_page_languages.append(page_lang.value if page_lang else None)
                    lines, lines_raw = [], []
                    for text, conf, flat_bbox, *rest in parsed_std:
                        bbox_source = rest[0] if rest and isinstance(rest[0], dict) else {}
                        bbox_valid = bbox_source.get("bbox_validated", True) is not False
                        append_page_line(text, text, conf, flat_bbox, page_lang, bbox_source, bbox_valid)

                elif f2_active:
                    t0 = self._timer()
                    page_debug_id = f"{run_id}_p{page_idx}"
                    f2_page, debug_paths, diag = self._run_f2_pipeline(
                        img_array, language, debug_run_id=page_debug_id
                    )
                    total_elapsed += self._elapsed_ms(t0)
                    all_debug.update({f"p{page_idx}_{k}": v for k, v in debug_paths.items()})
                    f2_page_diagnostics.append(diag)
                    for item in diag.get("rejected_bbox_lines", []) or []:
                        rejected_item = dict(item)
                        rejected_item["page_index"] = page_idx
                        rejected_bbox_lines.append(rejected_item)
                    for item in diag.get("corrected_bbox_lines", []) or []:
                        corrected_item = dict(item)
                        corrected_item["page_index"] = page_idx
                        corrected_bbox_lines.append(corrected_item)
                    lines, lines_raw = [], []
                    for text_raw, text_corrected, conf, flat_bbox, *rest in f2_page:
                        bbox_source = rest[0] if rest and isinstance(rest[0], dict) else {}
                        bbox_valid = bbox_source.get("bbox_validated", True) is not False
                        append_page_line(text_corrected, text_raw, conf, flat_bbox, language, bbox_source, bbox_valid)

                else:
                    # Pre-resize so PaddleX's internal max_side_limit (4000 px)
                    # is never triggered when det_limit_side_len is set.
                    capped_img, cap_scale = self._cap_image_side_len(img_array)
                    t0 = self._timer()
                    result = engine.ocr(capped_img)  # type: ignore[union-attr]
                    total_elapsed += self._elapsed_ms(t0)
                    parsed_std = self._parse_ocr_page(result[0]) if result and result[0] else []
                    inv = 1.0 / cap_scale if cap_scale else 1.0
                    parsed_std = [
                        self._line_with_bbox_source(
                            text,
                            conf,
                            [bbox[0] * inv, bbox[1] * inv, bbox[2] * inv, bbox[3] * inv],
                            variant_name="pre_resized" if cap_scale != 1.0 else "original",
                            scale=cap_scale,
                            effective_scale=cap_scale,
                            original_bbox_before_rescale=bbox,
                            img_shape=img_array.shape,
                            engine_name=LANG_CONFIG[language]["ocr_version"],
                        )
                        for text, conf, bbox in parsed_std
                    ]
                    parsed_std, rejected_bbox, corrected_bbox, median_h = self._validate_bbox_lines(
                        parsed_std,
                        img_array.shape,
                    )
                    for item in rejected_bbox:
                        rejected_item = dict(item)
                        rejected_item["page_index"] = page_idx
                        rejected_bbox_lines.append(rejected_item)
                    for item in corrected_bbox:
                        corrected_item = dict(item)
                        corrected_item["page_index"] = page_idx
                        corrected_bbox_lines.append(corrected_item)
                    lines, lines_raw = [], []
                    for text, conf, flat_bbox, *rest in parsed_std:
                        bbox_source = rest[0] if rest and isinstance(rest[0], dict) else {}
                        bbox_valid = bbox_source.get("bbox_validated", True) is not False
                        append_page_line(text, text, conf, flat_bbox, language, bbox_source, bbox_valid)

                page_text = "\n".join(lines)
                page_text_raw = "\n".join(page_raw_lines)
                page_avg_conf = (
                    sum(line["confidence"] for line in page_lines) / len(page_lines)
                    if page_lines else 0.0
                )
                structured_pages.append({
                    "page_index": page_idx,
                    "text": page_text,
                    "raw_text": page_text_raw,
                    "lines": page_lines,
                    "flagged_lines": page_flagged_lines,
                    "filtered_lines": page_flagged_lines,
                    "avg_confidence": round(page_avg_conf, 4),
                })
                page_texts.append(page_text)
                page_texts_raw.append(page_text_raw)
                self._maybe_empty_gpu_cache(page_idx)

            postprocess_applied = False
            postprocess_page_results: list[dict] = []
            debug_lines: list[dict] = []
            debug_blocks: list[dict] = []
            review_lines: list[dict] = []
            excluded_noise_lines: list[dict] = []
            per_line_confidence: list[dict] = []
            per_line_noise_score: list[dict] = []
            confidence_scores: list[float] = []
            selected_variants: list[str] = []
            if language == SupportedLanguage.ARABIC:
                page_texts = []
                for page in structured_pages:
                    original_page_text = page["text"]
                    page_idx = int(page.get("page_index", 0))
                    page_diag = (
                        f2_page_diagnostics[page_idx]
                        if f2_active and page_idx < len(f2_page_diagnostics)
                        else {}
                    )
                    post_result = postprocess_ocr_result(
                        page["lines"],
                        raw_text=page.get("raw_text", original_page_text),
                        selected_variant=page_diag.get("selected_variant") or "original",
                        variant_scores=page_diag.get("variant_scores", []),
                    )
                    processed_page_text = post_result["text"]
                    page["text_before_postprocess"] = original_page_text
                    page["postprocessed_text"] = processed_page_text
                    page["accepted_lines"] = post_result["accepted_lines"]
                    page["review_lines"] = post_result["review_lines"]
                    page["excluded_noise_lines"] = post_result["excluded_noise_lines"]
                    page["excluded_lines"] = post_result["excluded_lines"]
                    page["flagged_lines"] = post_result["flagged_lines"]
                    page["filtered_lines"] = post_result["flagged_lines"]
                    page["per_line_confidence"] = post_result["per_line_confidence"]
                    page["per_line_noise_score"] = post_result["per_line_noise_score"]
                    page["debug_lines"] = post_result["debug_lines"]
                    page["debug_blocks"] = post_result["debug_blocks"]
                    if processed_page_text:
                        page["text"] = processed_page_text
                        postprocess_applied = True
                    for line in post_result.get("review_lines", []):
                        review_line = dict(line)
                        review_line["page_index"] = page_idx
                        review_lines.append(review_line)
                    for line in post_result.get("excluded_noise_lines", []):
                        noise_line = dict(line)
                        noise_line["page_index"] = page_idx
                        excluded_noise_lines.append(noise_line)
                    for item in post_result.get("per_line_confidence", []):
                        conf_item = dict(item)
                        conf_item["page_index"] = page_idx
                        per_line_confidence.append(conf_item)
                    for item in post_result.get("per_line_noise_score", []):
                        noise_item = dict(item)
                        noise_item["page_index"] = page_idx
                        per_line_noise_score.append(noise_item)
                    for line in post_result.get("debug_lines", []):
                        debug_line = dict(line)
                        debug_line["page_index"] = page_idx
                        debug_lines.append(debug_line)
                    for block in post_result.get("debug_blocks", []):
                        debug_block = dict(block)
                        debug_block["page_index"] = page_idx
                        debug_blocks.append(debug_block)
                    confidence_scores.append(float(post_result.get("confidence_score", 0.0)))
                    if post_result.get("selected_variant"):
                        selected_variants.append(str(post_result["selected_variant"]))
                    postprocess_page_results.append(post_result)
                    page_texts.append(page["text"])

            raw_text     = "\n\n".join(t for t in page_texts     if t)
            raw_text_pre = "\n\n".join(t for t in page_texts_raw if t)
            avg_conf     = sum(w.confidence for w in words) / len(words) if words else 0.0
            line_count   = sum(len(page["lines"]) for page in structured_pages)
            char_count   = len(raw_text.strip())
            flagged_line_count = sum(
                len(page.get("flagged_lines", [])) for page in structured_pages
            )
            excluded_line_count = sum(
                len(page.get("excluded_lines", [])) for page in structured_pages
            )
            review_line_count = sum(
                len(page.get("review_lines", [])) for page in structured_pages
            )
            excluded_noise_line_count = sum(
                len(page.get("excluded_noise_lines", [])) for page in structured_pages
            )
            low_confidence_lines = sum(
                1
                for page in structured_pages
                for line in page["lines"]
                if line["confidence"] < 0.70
            )
            warnings: list[str] = []
            if char_count == 0:
                warnings.append("no_text_detected")
            if words and avg_conf < 0.65:
                warnings.append("low_average_confidence")
            if flagged_line_count > 0:
                warnings.append("arabic_noise_lines_flagged")
            if excluded_line_count > 0:
                warnings.append("excluded_high_noise_lines")
            if review_line_count > 0:
                warnings.append("low_confidence_review_lines_present")
            if line_count <= 2 and char_count < 40:
                warnings.append("low_text_coverage")
            quality_status = (
                "empty" if char_count == 0
                else "review" if warnings
                else "usable"
            )

            metadata: dict = {
                "page_count":                len(pages),
                "max_accuracy_mode":         f2_active,
                "final_text":                raw_text,
                "raw_text_before_correction": raw_text_pre,
                "selected_variant": (
                    selected_variants[0]
                    if selected_variants and len(set(selected_variants)) == 1
                    else "multi_page" if selected_variants else None
                ),
                "selected_variants": selected_variants,
                "confidence_score": round(
                    sum(confidence_scores) / len(confidence_scores),
                    4,
                ) if confidence_scores else round(avg_conf, 4),
                "debug_lines": debug_lines,
                "debug_blocks": debug_blocks,
                "rejected_bbox_lines": rejected_bbox_lines,
                "corrected_bbox_lines": corrected_bbox_lines,
                "review_lines": review_lines,
                "excluded_noise_lines": excluded_noise_lines,
                "per_line_confidence": per_line_confidence,
                "per_line_noise_score": per_line_noise_score,
                "pages":                     structured_pages,
                "quality": {
                    "status": quality_status,
                    "warnings": warnings,
                    "line_count": line_count,
                    "char_count": char_count,
                    "flagged_line_count": flagged_line_count,
                    "filtered_line_count": flagged_line_count,
                    "review_line_count": review_line_count,
                    "excluded_noise_line_count": excluded_noise_line_count,
                    "excluded_line_count": excluded_line_count,
                    "low_confidence_line_count": low_confidence_lines,
                    "avg_confidence": round(avg_conf, 4),
                    "line_handling": "exclude_only_when_low_confidence_and_high_noise_score",
                },
                "postprocess": {
                    "applied": postprocess_applied,
                    "mode": "deterministic_arabic_bbox_dictionary",
                    "preserves_raw_text": True,
                    "pages": postprocess_page_results,
                },
            }
            if all_debug:
                metadata["debug_visualizations"] = all_debug
            if f2_page_diagnostics:
                metadata["f2_page_diagnostics"] = f2_page_diagnostics
            if input_pipeline_pages:
                metadata["input_pipeline"] = input_pipeline_pages
            if language == SupportedLanguage.ALL:
                metadata["best_effort_language_mode"] = "best_single_language_engine_per_page"
                metadata["resolved_page_languages"]   = resolved_page_languages
            if language == SupportedLanguage.ARABIC:
                # Expose Arabic pipeline configuration for downstream consumers.
                metadata["arabic_pipeline"] = {
                    "normalize_alef":            self.arabic_normalize_alef,
                    "normalize_bidi":            self.arabic_normalize_bidi,
                    "filter_isolated_letters":   self.arabic_filter_isolated_letters,
                    "mixed_page_ratio_threshold": self.arabic_mixed_page_ratio_threshold,
                }
                # Lightweight document-type signal: diacritic density on the
                # full output text.  Informs downstream whether confidence
                # estimates may be systematically lower than normal.
                if raw_text:
                    ddensity = self._diacritic_density(raw_text)
                    if ddensity > 0.15:
                        doc_type = "quranic_or_classical"
                    elif ddensity > 0.05:
                        doc_type = "partially_diacritized"
                    else:
                        doc_type = "modern_printed"
                    metadata["arabic_doc_type"]         = doc_type
                    metadata["arabic_diacritic_density"] = round(ddensity, 4)
                    if doc_type == "quranic_or_classical":
                        logger.info(
                            "[PaddleOCR] Arabic doc type: %s (diacritic density=%.3f) "
                            "— confidence scores may be suppressed; consider lowering "
                            "text_rec_score_thresh for this document class.",
                            doc_type, ddensity,
                        )

            return OCRResult(
                model_name=self.name,
                language=language.value,
                raw_text=raw_text,
                words=words,
                inference_time_ms=round(total_elapsed, 2),
                avg_confidence=round(avg_conf, 4),
                metadata=metadata,
            )

        except Exception as exc:
            logger.exception("[PaddleOCR] Inference error: %s", exc)
            return OCRResult.from_error(self.name, language.value, str(exc))
