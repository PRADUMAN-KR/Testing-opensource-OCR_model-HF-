"""
Qari-OCR 0.2.2.1 VL 2B - Tier 1 Arabic VLM OCR.

Model card: https://huggingface.co/NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct
Install: pip install transformers qwen-vl-utils accelerate peft bitsandbytes
"""

import logging
import os
import threading
from typing import Any

from PIL import Image

from app.core.document import load_document_as_rgb_images
from app.models.base import BaseOCRModel, OCRResult, SupportedLanguage

logger = logging.getLogger(__name__)
_IMAGE_FILE_LOCK = threading.Lock()


DEFAULT_MODEL_ID = "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct"
DEFAULT_PROMPT = (
    """You are a high-precision document understanding system. Analyze the image and extract all visible content.

Tasks:
1. Extract plain text
2. Extract tables
3. Extract mathematical formulas

Rules:
- Do not explain anything.
- Do not summarize.
- Do not translate.
- Do not hallucinate missing content.
- Preserve original language (including Arabic) and formatting.
- Preserve reading order (for Arabic: right-to-left).
- Ignore noise (stains, shadows, borders, artifacts).
- If content is unreadable, use [unclear].

Output format (STRICT JSON):

{
  "text": "<all non-table, non-formula text with line breaks preserved>",
  "tables": [
    {
      "table_id": 1,
      "content": "<table in Markdown format>"
    }
  ],
  "formulas": [
    {
      "formula_id": 1,
      "latex": "<LaTeX representation>"
    }
  ]
}

Instructions:
- Put normal paragraph text only in "text"
- Put each detected table separately in "tables"
- Put each formula separately in "formulas"
- If no tables or formulas exist, return empty arrays: []
- Do not include anything outside this JSON

""")

class QariOCRVL2BModel(BaseOCRModel):
    """
    Hugging Face Qwen2-VL/PEFT OCR wrapper.

    Unlike PaddleOCR this model generates page-level text and does not return
    detector boxes or calibrated OCR confidence scores.
    """

    name = "qari_ocr_vl_2b"
    supported_languages = [
        SupportedLanguage.ARABIC,
        SupportedLanguage.ENGLISH,
        SupportedLanguage.HINDI,
        SupportedLanguage.PUNJABI,
    ]
    tier = 1

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 2000,
        torch_dtype: str = "auto",
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self._model: Any | None = None
        self._processor: Any | None = None
        self._process_vision_info: Any | None = None
        self._torch: Any | None = None

    async def load(self) -> None:
        from qwen_vl_utils import process_vision_info
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        logger.info("[QariOCR] Loading model=%s device_map=%s", self.model_id, self.device_map)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._process_vision_info = process_vision_info
        self._torch = torch
        logger.info("[QariOCR] Loaded model=%s", self.model_id)

    async def unload(self) -> None:
        model = self._model
        self._model = None
        self._processor = None
        self._process_vision_info = None
        torch = self._torch
        self._torch = None
        del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _input_device(self):
        if self._model is None:
            return "cpu"
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return getattr(self._model, "device", "cpu")

    def supports_all_languages(self) -> bool:
        return True

    def _generate_page_text(self, image: Image.Image) -> str:
        if self._model is None or self._processor is None or self._process_vision_info is None:
            raise RuntimeError("Qari OCR model is not initialized. Call load() first.")
        if self._torch is None or not self._torch.cuda.is_available():
            raise RuntimeError("Qari official generation path requires CUDA because it calls inputs.to('cuda').")

        src = "image.png"
        with _IMAGE_FILE_LOCK:
            try:
                image.save(src)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{src}"},
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ]
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = self._process_vision_info(messages)
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                return output_text
            finally:
                try:
                    os.remove(src)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _page_lines(page_index: int, text: str) -> list[dict]:
        lines: list[dict] = []
        for line_index, line_text in enumerate(line for line in text.splitlines() if line.strip()):
            lines.append(
                {
                    "page_index": page_index,
                    "line_index": line_index,
                    "text": line_text,
                    "raw_text": line_text,
                    "confidence": 0.0,
                    "bbox": None,
                    "bbox_source": {"source": "qari_vl_generation"},
                    "bbox_valid": False,
                    "status": "generated_no_bbox",
                }
            )
        return lines

    async def run_raw(self, image_bytes: bytes) -> str:
        pages = load_document_as_rgb_images(image_bytes)
        page_texts = [self._generate_page_text(image) for image in pages]
        return "\n\n".join(page_texts)

    async def run(self, image_bytes: bytes, language: SupportedLanguage | None = None) -> OCRResult:
        try:
            pages = load_document_as_rgb_images(image_bytes)
            page_texts: list[str] = []
            structured_pages: list[dict] = []
            start = self._timer()
            for page_index, image in enumerate(pages):
                page_text = self._generate_page_text(image)
                page_texts.append(page_text)
                lines = self._page_lines(page_index, page_text)
                structured_pages.append(
                    {
                        "page_index": page_index,
                        "text": page_text,
                        "raw_text": page_text,
                        "lines": lines,
                        "accepted_lines": lines,
                        "layout_mode": "generated_text",
                        "avg_confidence": 0.0,
                    }
                )

            raw_text = "\n\n".join(text for text in page_texts if text)
            elapsed_ms = self._elapsed_ms(start)
            char_count = len(raw_text.strip())
            line_count = sum(len(page["lines"]) for page in structured_pages)
            metadata = {
                "page_count": len(pages),
                "final_text": raw_text,
                "raw_text_before_correction": raw_text,
                "selected_variant": "qari_vl_generation",
                "confidence_score": 0.0,
                "layout_mode": "generated_text",
                "pages": structured_pages,
                "quality": {
                    "status": "empty" if char_count == 0 else "generated",
                    "warnings": ["no_text_detected"] if char_count == 0 else [],
                    "line_count": line_count,
                    "char_count": char_count,
                    "avg_confidence": 0.0,
                    "confidence_note": "Qari OCR does not expose calibrated confidence scores.",
                },
                "generation": {
                    "model_id": self.model_id,
                    "max_new_tokens": self.max_new_tokens,
                    "prompt": self.prompt,
                    "mode": "qari_official_generation_prompt",
                },
            }
            return OCRResult(
                model_name=self.name,
                language="",
                raw_text=raw_text,
                words=[],
                inference_time_ms=elapsed_ms,
                avg_confidence=0.0,
                metadata=metadata,
            )
        except Exception as exc:
            logger.exception("[QariOCR] Inference error: %s", exc)
            return OCRResult.from_error(self.name, "", str(exc))
