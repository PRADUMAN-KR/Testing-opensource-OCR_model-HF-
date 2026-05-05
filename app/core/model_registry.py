"""
Model Registry — loads configured OCR models at startup.
"""

import logging
from typing import Dict, List, Optional

from app.core.config import settings
from app.models.base import BaseOCRModel

logger = logging.getLogger(__name__)

PADDLEOCR_VL_MODEL_NAME = "paddleocr_vl"
AVAILABLE_MODEL_NAMES = [PADDLEOCR_VL_MODEL_NAME]


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
        if name == PADDLEOCR_VL_MODEL_NAME:
            from app.models.paddleocr_vl import PaddleOCRVLModel

            return PaddleOCRVLModel(
                device=settings.PADDLEOCR_VL_DEVICE,
                pipeline_version=settings.PADDLEOCR_VL_PIPELINE_VERSION,
                use_layout_detection=settings.PADDLEOCR_VL_USE_LAYOUT_DETECTION,
                use_doc_orientation_classify=settings.PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY,
                use_doc_unwarping=settings.PADDLEOCR_VL_USE_DOC_UNWARPING,
                use_chart_recognition=settings.PADDLEOCR_VL_USE_CHART_RECOGNITION,
                use_seal_recognition=settings.PADDLEOCR_VL_USE_SEAL_RECOGNITION,
                use_ocr_for_image_block=settings.PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK,
                format_block_content=settings.PADDLEOCR_VL_FORMAT_BLOCK_CONTENT,
                merge_layout_blocks=settings.PADDLEOCR_VL_MERGE_LAYOUT_BLOCKS,
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
