"""
Model Registry — loads configured OCR models at startup.
"""

import logging
from typing import Dict, List, Optional

from app.core.config import settings
from app.models.base import BaseOCRModel

logger = logging.getLogger(__name__)

QWEN_OCR_MODEL_NAME = "qwen_ocr"
AVAILABLE_MODEL_NAMES = [QWEN_OCR_MODEL_NAME]


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
        if name == QWEN_OCR_MODEL_NAME:
            from app.models.qwen_ocr import QwenOCRModel

            return QwenOCRModel(
                provider=settings.QWEN_OCR_PROVIDER,
                model_id=settings.QWEN_OCR_MODEL_ID,
                vllm_base_url=settings.QWEN_OCR_VLLM_BASE_URL,
                vllm_api_key=settings.QWEN_OCR_VLLM_API_KEY,
                vllm_timeout=settings.QWEN_OCR_VLLM_TIMEOUT,
                device_map=settings.QWEN_OCR_DEVICE_MAP,
                torch_dtype=settings.QWEN_OCR_TORCH_DTYPE,
                max_new_tokens=settings.QWEN_OCR_MAX_NEW_TOKENS,
                prompt=settings.QWEN_OCR_PROMPT,
                pdf_dpi=settings.QWEN_OCR_PDF_DPI,
                max_pdf_pages=settings.QWEN_OCR_MAX_PDF_PAGES,
                low_cpu_mem_usage=settings.QWEN_OCR_LOW_CPU_MEM_USAGE,
                offload_buffers=settings.QWEN_OCR_OFFLOAD_BUFFERS,
                offload_folder=settings.QWEN_OCR_OFFLOAD_FOLDER,
                verbose=settings.QWEN_OCR_VERBOSE,
            )

        logger.warning(f"[Registry] Unknown model: {name}")
        return None

    def get_active_model(self) -> Optional["BaseOCRModel"]:
        for name in settings.ENABLED_MODELS:
            model = self.loaded_models.get(name)
            if model is not None:
                return model
        return None

    def active_model_name(self) -> Optional[str]:
        model = self.get_active_model()
        return getattr(model, "name", None) if model is not None else None

    def active_failure_reason(self) -> Optional[str]:
        for name in settings.ENABLED_MODELS:
            failure_reason = self.failed_models.get(name)
            if failure_reason:
                return f"{name}: {failure_reason}"
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
