"""
PaddleOCR PP-OCRv5 — Tier 2

Standard pipeline + F2 MAX-ACCURACY pipeline (enabled when max_accuracy=True):

  Stage 1 — Full-page variants (original, 2x, CLAHE,
             + Arabic-specific: adaptive_thresh, ink_bleed, deskewed,
               downscaled_0.75x [for high-DPI inputs])
  Stage 2 — Run full engine.ocr() on each variant × each engine
  Stage 3 — Pick best complete pass (scored; upscaled variant lightly penalised)
             Score function is Arabic-aware: token-based counting, ligature
             compensation, mixed-page detection, diacritic-density gating.
  Stage 4  — Column-aware RTL line ordering with dynamic band tolerance derived
             from median box height.
  Stage 5  — Full Arabic text correction: kashida removal, Alef normalization
             (opt-in), diacritic deduplication, punctuation normalization,
             isolated-letter noise filtering, ZWNJ/ZWJ/BOM stripping.

Supports: English, Arabic, Hindi
Install:  pip install paddlepaddle paddleocr opencv-python python-bidi arabic-reshaper
"""

import logging
import math
import os
import re
import statistics
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models.base import BaseOCRModel, OCRResult, OCRWord, SupportedLanguage
from app.core.document import load_document_as_rgb_images

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
        roi_min_area_ratio: float = 0.15,
        roi_pad_ratio: float = 0.02,
        paddle_mem_fraction: float | None = None,
        paddle_allocator_strategy: str | None = None,
        paddle_gpu_memory_fraction: float | None = None,
        empty_cache_between_pages: bool = False,
        det_limit_side_len: int | None = None,
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
        self.roi_min_area_ratio = roi_min_area_ratio
        self.roi_pad_ratio = roi_pad_ratio
        self.paddle_mem_fraction = paddle_mem_fraction
        self.paddle_allocator_strategy = paddle_allocator_strategy
        self.paddle_gpu_memory_fraction = paddle_gpu_memory_fraction
        self.empty_cache_between_pages = empty_cache_between_pages
        self.det_limit_side_len = det_limit_side_len
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
            "text_det_thresh": 0.5,
            "text_det_box_thresh": 0.7,
            "text_rec_score_thresh": 0.5,
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
                "[PaddleOCR] Engine kwargs: lang=%s ocr_version=%s precision=%s use_tensorrt=%s det_limit_side_len=%s limit_side_len=%s",
                paddle_lang,
                ocr_version,
                kwargs.get("precision", "fp32"),
                kwargs.get("use_tensorrt", False),
                kwargs.get("det_limit_side_len"),
                kwargs.get("limit_side_len"),
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

    def _apply_document_roi_warp(self, img_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Find largest plausible document quad (Canny → contours → approxPoly)
        and apply perspective warp. Falls back to original image on failure.
        """
        meta: dict = {"roi_warp_applied": False, "roi_warp_reason": "disabled"}
        if not self.input_roi_warp:
            return img_rgb, meta

        h, w = img_rgb.shape[:2]
        img_area = float(h * w)
        if img_area < 5000:
            meta["roi_warp_reason"] = "image_too_small"
            return img_rgb, meta

        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            meta["roi_warp_reason"] = "no_contours"
            return img_rgb, meta

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
            return img_rgb, meta

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
        meta["roi_warp_size"] = [out_rgb.shape[1], out_rgb.shape[0]]
        logger.info(
            "[PaddleOCR] ROI warp applied: %dx%d → %dx%d (pad=%d)",
            w, h, out_rgb.shape[1], out_rgb.shape[0], pad,
        )
        return out_rgb, meta

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

        For Arabic, additional variants are produced:
          - adaptive_thresh : Sauvola-style adaptive binarization — preserves
            tashkeel thin strokes that CLAHE can smear.
          - ink_bleed_clean  : morphological opening to suppress show-through
            between lines (common in scanned/photocopied Arabic documents).
          - deskewed         : Hough-based rotation correction (±10° clamp).
          - downscaled_0.75x : Only for high-DPI inputs (max dim > 3000 px);
            counteracts the 2× upscale benefit reversal on already-dense text.

        Returns list of (name, scale_factor, image_rgb).
        scale_factor is relative to the original (1.0 = same size, 2.0 = 2x).
        """
        variants: list[tuple[str, float, np.ndarray]] = []

        variants.append(("original", 1.0, img_rgb))

        h, w = img_rgb.shape[:2]

        # Upscale — always generated; lightly penalised in scorer.
        up = cv2.resize(img_rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        variants.append(("upscaled_2x", 2.0, up))

        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # CLAHE — contrast enhancement, good for faded / low-contrast scans.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)
        variants.append((
            "clahe", 1.0,
            cv2.cvtColor(cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB),
        ))

        # ------------------------------------------------------------------
        # Arabic-specific variants (only added when language == ARABIC)
        # ------------------------------------------------------------------
        if language == SupportedLanguage.ARABIC:

            # 1) Adaptive threshold — preserves diacritics (tashkeel) thin strokes
            #    that CLAHE can smear.  Block size 11 px works well for 150–300 DPI.
            adaptive = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11, C=6,
            )
            adaptive_rgb = cv2.cvtColor(
                cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB
            )
            variants.append(("adaptive_thresh", 1.0, adaptive_rgb))

            # 2) Ink-bleed removal — morphological opening with a 1-row horizontal
            #    structuring element removes horizontal smear between lines while
            #    preserving connected Arabic strokes.
            ink_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
            ink_clean = cv2.morphologyEx(gray, cv2.MORPH_OPEN, ink_kernel, iterations=1)
            ink_rgb = cv2.cvtColor(
                cv2.cvtColor(ink_clean, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB
            )
            variants.append(("ink_bleed_clean", 1.0, ink_rgb))

            # 3) Deskewed — correct small rotations from scanner/camera misalignment.
            skew = self._estimate_skew_angle(gray)
            if abs(skew) >= 0.2:
                deskewed_rgb = self._rotate_image(img_rgb, -skew)
                variants.append(("deskewed", 1.0, deskewed_rgb))
                logger.debug("[PaddleOCR] Deskew applied: %.2f°", skew)

            # 4) High-DPI downscale — large images cause the 2× upscale to be
            #    counter-productive; a 0.75× pass gives a different receptive field.
            if max(h, w) > _HIGHDPI_THRESHOLD_PX:
                dw = int(w * 0.75)
                dh = int(h * 0.75)
                downscaled = cv2.resize(img_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
                variants.append(("downscaled_0.75x", 0.75, downscaled))
                logger.debug("[PaddleOCR] High-DPI downscale variant added (%dx%d → %dx%d)", w, h, dw, dh)

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
                    if effective_scale != 1.0 and parsed:
                        inv = 1.0 / effective_scale
                        parsed = [
                            (text, conf, [
                                int(bbox[0] * inv), int(bbox[1] * inv),
                                int(bbox[2] * inv), int(bbox[3] * inv),
                            ])
                            for text, conf, bbox in parsed
                        ]
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
    def _median_box_height(parsed: list[tuple[str, float, list[int]]]) -> float:
        """
        Compute median bounding-box height from a parsed page.
        Used to derive a dynamic band tolerance for RTL line sorting.
        Returns _ARABIC_LINE_BAND_TOL_PX if fewer than 3 boxes present.
        """
        if len(parsed) < 3:
            return float(_ARABIC_LINE_BAND_TOL_PX)
        heights = [float(bbox[3] - bbox[1]) for _, _, bbox in parsed if bbox[3] > bbox[1]]
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

        total_text = " ".join(t.strip() for t, _, _ in parsed if t.strip())
        if not total_text:
            return 0.0

        mean_conf = sum(c for _, c, _ in parsed) / len(parsed)

        if language == SupportedLanguage.ARABIC:
            # Use word-token count × proxy chars/token instead of raw char len.
            token_count = self._count_arabic_word_tokens(total_text)
            content_len = max(1.0, float(token_count) * 4.5)
            score = math.sqrt(content_len) * mean_conf

            # Diacritic density guard.
            ddensity = self._diacritic_density(total_text)
            if ddensity > 0.40:
                score *= 0.3
                logger.debug("[PaddleOCR F2] Diacritic-dense pass penalised (density=%.2f): %s", ddensity, variant_name)

            # Bbox height density check.
            med_h = self._median_box_height(parsed)
            if med_h < self.MIN_BOX_HEIGHT * 2:
                score *= 0.4
                logger.debug("[PaddleOCR F2] Low median box height (%.1f px) penalised: %s", med_h, variant_name)

            # Arabic ratio gating — mixed-page aware.
            ar_ratio = self._arabic_char_ratio(total_text)
            non_ar_ratio = 1.0 - ar_ratio
            is_mixed_page = non_ar_ratio > self.arabic_mixed_page_ratio_threshold
            if is_mixed_page:
                # Looser gate for legitimate Arabic + Latin/numeric mixed content.
                if ar_ratio < 0.10:
                    score *= 0.2
            else:
                if ar_ratio < 0.30:
                    score *= 0.2
                elif ar_ratio < 0.50:
                    score *= 0.5
        else:
            total_len = len(total_text)
            score = math.sqrt(max(1.0, float(total_len))) * mean_conf

        # Upscale penalty — applies to all languages.
        if variant_name == "upscaled_2x":
            score *= 0.88

        # Downscale variant is also slightly penalised vs the original.
        if variant_name == "downscaled_0.75x":
            score *= 0.93

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
        total_text  = " ".join(t.strip() for t, _, _ in parsed if t.strip())
        total_chars = len(total_text)
        avg_conf    = sum(c for _, c, _ in parsed) / line_count if line_count else 0.0
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
        for text_raw, text_corrected, conf, (x1, y1, x2, y2) in results:
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
        }

        # Stage 1+2 (primary): generate variants, run primary engine only
        primary_passes = self._run_all_passes(
            img_rgb, lang_enum, include_primary=True, include_alt=False
        )
        if not primary_passes:
            return [], {}, diagnostics

        # Stage 3: pick the best complete primary pass
        best_variant, best_engine, best_parsed, best_score = self._select_best_pass(
            primary_passes, lang_enum
        )
        diagnostics["primary_best_score"] = round(best_score, 3)

        primary_sorted = self._sort_lines_reading_order(best_parsed, lang_enum)
        diagnostics["primary_response_raw_text"] = "\n".join(
            t for t, _, _ in primary_sorted if t and t.strip()
        )
        diagnostics["primary_response_corrected_text"] = "\n".join(
            (
                self._arabic_correct(t)
                if lang_enum == SupportedLanguage.ARABIC
                else t
            )
            for t, _, _ in primary_sorted
            if t and t.strip()
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
                    alt_variant, alt_engine, alt_parsed, alt_score = self._select_best_pass(
                        alt_passes, lang_enum
                    )
                    diagnostics["alt_best_score"] = round(alt_score, 3)
                    alt_sorted = self._sort_lines_reading_order(alt_parsed, lang_enum)
                    diagnostics["alt_response_raw_text"] = "\n".join(
                        t for t, _, _ in alt_sorted if t and t.strip()
                    )
                    diagnostics["alt_response_corrected_text"] = "\n".join(
                        self._arabic_correct(t) for t, _, _ in alt_sorted if t and t.strip()
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

        # Stage 4+5: apply Arabic correction to the winning pass
        results: list[tuple[str, str, float, list[int]]] = []
        for text_raw, conf, bbox in best_parsed:
            if not text_raw or not text_raw.strip():
                continue
            text_corrected = (
                self._arabic_correct(text_raw)
                if lang_enum == SupportedLanguage.ARABIC
                else text_raw
            )
            if text_corrected:
                results.append((text_raw, text_corrected, conf, bbox))

        results = self._sort_f2_results_reading_order(results, lang_enum)

        logger.info(
            "[PaddleOCR F2] Pipeline done: %d lines (best pass: %s × %s, from %d total passes)",
            len(results), best_variant, best_engine, all_pass_count,
        )
        diagnostics["selected_variant"] = best_variant
        diagnostics["selected_engine"] = best_engine
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
        return sum(conf for _, conf, _ in parsed_page) if parsed_page else 0.0

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
            # Scale bboxes back to original image coordinates.
            if cap_scale != 1.0 and parsed:
                inv = 1.0 / cap_scale
                parsed = [
                    (text, conf, [
                        int(bbox[0] * inv), int(bbox[1] * inv),
                        int(bbox[2] * inv), int(bbox[3] * inv),
                    ])
                    for text, conf, bbox in parsed
                ]
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

            for page_idx, image in enumerate(pages):
                img_array = np.array(image)
                img_array, prep_meta = self._apply_document_roi_warp(img_array)
                input_pipeline_pages.append(prep_meta)

                if language == SupportedLanguage.ALL:
                    page_lang, parsed_std, page_elapsed = self._select_all_language_page(img_array)
                    total_elapsed += page_elapsed
                    resolved_page_languages.append(page_lang.value if page_lang else None)
                    lines, lines_raw = [], []
                    for text, conf, flat_bbox in parsed_std:
                        words.append(OCRWord(text=text, confidence=conf, bbox=flat_bbox))
                        lines.append(text)
                        lines_raw.append(text)

                elif f2_active:
                    t0 = self._timer()
                    page_debug_id = f"{run_id}_p{page_idx}"
                    f2_page, debug_paths, diag = self._run_f2_pipeline(
                        img_array, language, debug_run_id=page_debug_id
                    )
                    total_elapsed += self._elapsed_ms(t0)
                    all_debug.update({f"p{page_idx}_{k}": v for k, v in debug_paths.items()})
                    f2_page_diagnostics.append(diag)
                    lines, lines_raw = [], []
                    for text_raw, text_corrected, conf, flat_bbox in f2_page:
                        words.append(OCRWord(text=text_corrected, confidence=conf, bbox=flat_bbox))
                        lines.append(text_corrected)
                        lines_raw.append(text_raw)

                else:
                    # Pre-resize so PaddleX's internal max_side_limit (4000 px)
                    # is never triggered when det_limit_side_len is set.
                    capped_img, cap_scale = self._cap_image_side_len(img_array)
                    t0 = self._timer()
                    result = engine.ocr(capped_img)  # type: ignore[union-attr]
                    total_elapsed += self._elapsed_ms(t0)
                    parsed_std = self._parse_ocr_page(result[0]) if result and result[0] else []
                    # Scale bboxes back to original image coordinates.
                    if cap_scale != 1.0 and parsed_std:
                        inv = 1.0 / cap_scale
                        parsed_std = [
                            (text, conf, [
                                int(bbox[0] * inv), int(bbox[1] * inv),
                                int(bbox[2] * inv), int(bbox[3] * inv),
                            ])
                            for text, conf, bbox in parsed_std
                        ]
                    lines, lines_raw = [], []
                    for text, conf, flat_bbox in parsed_std:
                        words.append(OCRWord(text=text, confidence=conf, bbox=flat_bbox))
                        lines.append(text)
                        lines_raw.append(text)

                page_texts.append("\n".join(lines))
                page_texts_raw.append("\n".join(lines_raw))
                self._maybe_empty_gpu_cache(page_idx)

            raw_text     = "\n\n".join(t for t in page_texts     if t)
            raw_text_pre = "\n\n".join(t for t in page_texts_raw if t)
            avg_conf     = sum(w.confidence for w in words) / len(words) if words else 0.0

            metadata: dict = {
                "page_count":                len(pages),
                "max_accuracy_mode":         f2_active,
                "raw_text_before_correction": raw_text_pre,
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
