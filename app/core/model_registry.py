"""
Model Registry — loads PaddleOCR at startup.
"""

import logging
from typing import Dict, List, Optional

from app.models.base import BaseOCRModel
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.loaded_models: Dict[str, BaseOCRModel] = {}
        self.failed_models: Dict[str, str] = {}

    async def initialize(self, model_names: List[str]):
        """Load all enabled models at startup."""
        self.loaded_models.clear()
        self.failed_models.clear()
        for name in model_names:
            try:
                model = self._build_model(name)
                if not model:
                    self.failed_models[name] = "Unknown model name or optional dependency missing"
                    logger.error(f"[Registry] Failed to load {name}: {self.failed_models[name]}")
                    continue

                await model.load()
                self.loaded_models[name] = model
                logger.info(f"[Registry] Loaded: {name}")
            except Exception as e:
                self.failed_models[name] = str(e)
                logger.error(f"[Registry] Failed to load {name}: {e}")

        loaded_names = sorted(self.loaded_models.keys())
        failed_names = sorted(self.failed_models.keys())
        requested_count = len(model_names)
        loaded_count = len(loaded_names)
        failed_count = len(failed_names)

        if failed_count == 0:
            logger.info(
                f"[Registry] Startup complete: all {loaded_count}/{requested_count} models loaded successfully: "
                f"{', '.join(loaded_names)}"
            )
            return

        logger.warning(
            f"[Registry] Startup complete with partial success: loaded {loaded_count}/{requested_count}, failed {failed_count}"
        )
        logger.warning(
            f"[Registry] Loaded models: {', '.join(loaded_names) if loaded_names else 'none'}"
        )
        for name in failed_names:
            logger.error(f"[Registry] Startup failure detail | model={name} | reason={self.failed_models[name]}")

    def _build_model(self, name: str) -> Optional["BaseOCRModel"]:
        """Factory: maps model name → implementation class."""
        if name == "paddleocr_v5":
            from app.models.paddleocr_v5 import PaddleOCRv5Model

            return PaddleOCRv5Model(
                use_gpu=settings.PADDLE_USE_GPU,
                max_accuracy=settings.PADDLE_MAX_ACCURACY,
                paddle_mem_fraction=settings.PADDLE_MEM_FRACTION,
                paddle_allocator_strategy=settings.PADDLE_FLAGS_ALLOCATOR_STRATEGY,
                paddle_gpu_memory_fraction=settings.PADDLE_FLAGS_FRACTION_OF_GPU_MEMORY_TO_USE,
                empty_cache_between_pages=settings.PADDLE_EMPTY_CACHE_BETWEEN_PAGES,
                det_limit_side_len=settings.PADDLE_DET_LIMIT_SIDE_LEN,
                text_det_thresh=settings.PADDLE_TEXT_DET_THRESH,
                text_det_box_thresh=settings.PADDLE_TEXT_DET_BOX_THRESH,
                text_rec_score_thresh=settings.PADDLE_TEXT_REC_SCORE_THRESH,
                precision=settings.PADDLE_PRECISION,
                enable_fp16=settings.PADDLE_FP16,
                use_tensorrt=settings.PADDLE_TENSORRT,
                debug_output_dir=settings.PADDLE_DEBUG_OUTPUT_DIR or None,
                enable_arabic_v3_fallback=settings.PADDLE_ARABIC_V3_FALLBACK,
                always_run_both_arabic_engines=settings.PADDLE_ARABIC_RUN_BOTH_ENGINES,
                fallback_min_lines=settings.PADDLE_F2_FALLBACK_MIN_LINES,
                fallback_min_chars=settings.PADDLE_F2_FALLBACK_MIN_CHARS,
                fallback_min_avg_conf=settings.PADDLE_F2_FALLBACK_MIN_AVG_CONF,
                fallback_min_ar_ratio=settings.PADDLE_F2_FALLBACK_MIN_ARABIC_RATIO,
                fallback_replace_margin=settings.PADDLE_F2_FALLBACK_REPLACE_MARGIN,
                input_roi_warp=settings.PADDLE_INPUT_ROI_WARP,
                arabic_auto_page_crop=settings.PADDLE_ARABIC_AUTO_PAGE_CROP,
                roi_min_area_ratio=settings.PADDLE_ROI_MIN_AREA_RATIO,
                roi_pad_ratio=settings.PADDLE_ROI_PAD_RATIO,
                arabic_normalize_alef=settings.PADDLE_ARABIC_NORMALIZE_ALEF,
                arabic_normalize_bidi=settings.PADDLE_ARABIC_NORMALIZE_BIDI,
                arabic_filter_isolated_letters=settings.PADDLE_ARABIC_FILTER_ISOLATED_LETTERS,
                arabic_mixed_page_ratio_threshold=settings.PADDLE_ARABIC_MIXED_PAGE_RATIO_THRESHOLD,
            )

        logger.warning(f"[Registry] Unknown model: {name}")
        return None

    def get(self, name: str) -> Optional["BaseOCRModel"]:
        return self.loaded_models.get(name)

    def all(self) -> Dict[str, "BaseOCRModel"]:
        return self.loaded_models

    async def shutdown(self):
        for name, model in self.loaded_models.items():
            try:
                await model.unload()
                logger.info(f"[Registry] Unloaded: {name}")
            except Exception as e:
                logger.warning(f"[Registry] Error unloading {name}: {e}")
