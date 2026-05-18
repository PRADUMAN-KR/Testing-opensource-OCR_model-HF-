import argparse
import asyncio
import io
import logging
import threading
from typing import Any

from app.models.base import BaseOCRModel, OCRResult

logger = logging.getLogger(__name__)


class QwenOCRModel(BaseOCRModel):
    """
    Official Hugging Face Transformers wrapper for Qwen image-text-to-text OCR.

    Qwen does not expose calibrated OCR confidence scores, so confidence fields
    are set to 0.0.
    """

    name = "qwen_ocr"
    tier = 1

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.6-27B",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 4096,
        prompt: str = (
            "Extract all readable text from this document image. Preserve reading order. "
            "Use Markdown for tables. Return only the extracted text."
        ),
        pdf_dpi: int = 200,
        max_pdf_pages: int = 20,
    ):
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.pdf_dpi = pdf_dpi
        self.max_pdf_pages = max_pdf_pages
        self._processor: Any | None = None
        self._model: Any | None = None
        self._predict_lock = threading.Lock()

    async def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        import torch

        dtype: str | Any = self.torch_dtype
        if self.torch_dtype != "auto":
            dtype = getattr(torch, self.torch_dtype)

        logger.info("[Qwen OCR] Loading model_id=%s device_map=%s", self.model_id, self.device_map)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        model_kwargs = {
            "device_map": self.device_map,
            "dtype": dtype,
        }
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()
        logger.info("[Qwen OCR] Loaded")

    async def unload(self) -> None:
        model = self._model
        processor = self._processor
        self._model = None
        self._processor = None
        del model
        del processor
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("[Qwen OCR] CUDA cache cleanup skipped", exc_info=True)

    @staticmethod
    def _is_pdf(file_bytes: bytes) -> bool:
        return file_bytes[:16].startswith(b"%PDF")

    @staticmethod
    def _line_from_text(page_index: int, text: str) -> dict:
        return {
            "page_index": page_index,
            "line_index": 0,
            "text": text,
            "raw_text": text,
            "confidence": 0.0,
            "bbox": None,
            "bbox_source": {"source": "qwen_generated_text"},
            "bbox_valid": False,
            "status": "generated",
        }

    def _images_from_pdf(self, file_bytes: bytes) -> list[Any]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF OCR with Qwen requires the 'pymupdf' package.") from exc

        from PIL import Image

        images: list[Any] = []
        zoom = self.pdf_dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            page_count = min(len(document), self.max_pdf_pages)
            for page_index in range(page_count):
                pixmap = document[page_index].get_pixmap(matrix=matrix, alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                images.append(image)
        return images

    def _images_from_bytes(self, file_bytes: bytes) -> list[Any]:
        if self._is_pdf(file_bytes):
            return self._images_from_pdf(file_bytes)

        from PIL import Image, ImageSequence

        image = Image.open(io.BytesIO(file_bytes))
        return [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(image)]

    def _generate_text_for_image(self, image: Any) -> str:
        if self._processor is None or self._model is None:
            raise RuntimeError("Qwen OCR model is not initialized. Call load() first.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _metadata_from_page_texts(self, page_texts: list[str]) -> tuple[str, dict]:
        pages: list[dict] = []
        for page_index, page_text in enumerate(page_texts):
            lines = [self._line_from_text(page_index, page_text)] if page_text else []
            pages.append(
                {
                    "page_index": page_index,
                    "text": page_text,
                    "raw_text": page_text,
                    "lines": lines,
                    "accepted_lines": lines,
                    "layout_mode": "qwen_image_text_to_text",
                    "tables": [],
                    "totals": {
                        "block_count": len(lines),
                        "table_count": 0,
                    },
                    "avg_confidence": 0.0,
                }
            )

        raw_text = "\n\n".join(text for text in page_texts if text)
        metadata = {
            "page_count": len(pages),
            "final_text": raw_text,
            "raw_text_before_correction": raw_text,
            "selected_variant": self.model_id,
            "confidence_score": 0.0,
            "layout_mode": "qwen_image_text_to_text",
            "pages": pages,
            "tables": [],
            "markdown_text": raw_text,
            "quality": {
                "status": "empty" if not raw_text.strip() else "generated",
                "warnings": ["no_text_detected"] if not raw_text.strip() else [],
                "line_count": sum(len(page["lines"]) for page in pages),
                "char_count": len(raw_text.strip()),
                "avg_confidence": 0.0,
                "confidence_note": "Qwen image-text-to-text generation does not expose calibrated OCR confidence scores.",
            },
            "generation": {
                "pipeline": "AutoModelForImageTextToText",
                "model_id": self.model_id,
                "device_map": self.device_map,
                "max_new_tokens": self.max_new_tokens,
                "prompt": self.prompt,
                "pdf_dpi": self.pdf_dpi,
                "max_pdf_pages": self.max_pdf_pages,
            },
        }
        return raw_text, metadata

    def _run_sync(self, image_bytes: bytes) -> OCRResult:
        try:
            start = self._timer()
            images = self._images_from_bytes(image_bytes)
            page_texts: list[str] = []
            with self._predict_lock:
                for image in images:
                    page_texts.append(self._generate_text_for_image(image))
            raw_text, metadata = self._metadata_from_page_texts(page_texts)
            elapsed_ms = self._elapsed_ms(start)

            return OCRResult(
                model_name=self.name,
                raw_text=raw_text,
                inference_time_ms=elapsed_ms,
                avg_confidence=0.0,
                metadata=metadata,
            )
        except Exception as exc:
            logger.exception("[Qwen OCR] Inference error: %s", exc)
            return OCRResult.from_error(self.name, str(exc))

    async def run(self, image_bytes: bytes) -> OCRResult:
        return await asyncio.to_thread(self._run_sync, image_bytes)

    def _run_raw_json_sync(self, image_bytes: bytes) -> dict:
        result = self._run_sync(image_bytes)
        if result.error:
            raise RuntimeError(result.error)
        return {
            "res": {
                **result.metadata,
                "model_name": result.model_name,
                "inference_time_ms": result.inference_time_ms,
            }
        }

    async def run_raw_json(self, image_bytes: bytes) -> dict:
        return await asyncio.to_thread(self._run_raw_json_sync, image_bytes)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen OCR using the official Transformers model path.")
    parser.add_argument("--image-path", "-i", required=True, help="Path to the image or PDF to parse.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B", help="Hugging Face model id.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum generated OCR tokens.")
    args = parser.parse_args()

    async def _run():
        model = QwenOCRModel(model_id=args.model_id, max_new_tokens=args.max_new_tokens)
        await model.load()
        with open(args.image_path, "rb") as image_file:
            result = await model.run(image_file.read())
        print(result.raw_text)
        await model.unload()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
