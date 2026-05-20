import argparse
import asyncio
import base64
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any

from app.models.base import BaseOCRModel, OCRResult

logger = logging.getLogger(__name__)


class QwenOCRModel(BaseOCRModel):
    """
    Qwen image-text-to-text OCR wrapper.

    In vLLM mode, this wrapper calls an OpenAI-compatible vLLM server instead
    of loading model weights inside the API process.

    Qwen does not expose calibrated OCR confidence scores, so confidence fields
    are set to 0.0.
    """

    name = "qwen_ocr"
    tier = 1

    def __init__(
        self,
        provider: str = "vllm",
        model_id: str = "Qwen/Qwen3.6-27B",
        vllm_base_url: str = "http://localhost:8000/v1",
        vllm_api_key: str = "EMPTY",
        vllm_timeout: float = 300.0,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 4096,
        prompt: str = (
            "Extract all readable text from this document image. Preserve reading order. "
            "Use Markdown for tables. Return only the extracted text."
        ),
        pdf_dpi: int = 200,
        max_pdf_pages: int = 20,
        low_cpu_mem_usage: bool = True,
        offload_buffers: bool = True,
        offload_folder: str | None = "data/qwen_offload",
        verbose: bool = True,
    ):
        self.provider = provider.lower()
        self.model_id = model_id
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.vllm_api_key = vllm_api_key
        self.vllm_timeout = vllm_timeout
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.pdf_dpi = pdf_dpi
        self.max_pdf_pages = max_pdf_pages
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.offload_buffers = offload_buffers
        self.offload_folder = offload_folder
        self.verbose = verbose
        self._processor: Any | None = None
        self._model: Any | None = None
        self._predict_lock = threading.Lock()

    def _configure_verbose_logging(self) -> None:
        if not self.verbose:
            return

        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        try:
            from huggingface_hub import logging as hf_logging
            from transformers.utils import logging as transformers_logging

            hf_logging.set_verbosity_info()
            hf_logging.enable_progress_bars()
            transformers_logging.set_verbosity_info()
            transformers_logging.enable_progress_bar()
        except Exception:
            logger.debug("[Qwen OCR] Verbose Hugging Face logging setup skipped", exc_info=True)

    @staticmethod
    def _cuda_memory_summary() -> dict[str, int | None]:
        try:
            import torch

            if not torch.cuda.is_available():
                return {"cuda_available": 0, "device_count": 0}
            device_index = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            return {
                "cuda_available": 1,
                "device_count": torch.cuda.device_count(),
                "current_device": device_index,
                "free_gib": round(free_bytes / 1024**3),
                "total_gib": round(total_bytes / 1024**3),
            }
        except Exception:
            logger.debug("[Qwen OCR] CUDA memory summary unavailable", exc_info=True)
            return {"cuda_available": None, "device_count": None}

    async def load(self) -> None:
        if self.provider == "vllm":
            await self._load_vllm()
            return
        if self.provider != "transformers":
            raise ValueError("QWEN_OCR_PROVIDER must be either 'vllm' or 'transformers'.")

        await self._load_transformers()

    async def _load_vllm(self) -> None:
        logger.info(
            "[Qwen OCR] Using vLLM provider model_id=%s base_url=%s",
            self.model_id,
            self.vllm_base_url,
        )
        try:
            import httpx

            headers = self._vllm_headers()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.vllm_base_url}/models", headers=headers)
                response.raise_for_status()
            logger.info("[Qwen OCR] vLLM server is reachable")
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach vLLM at {self.vllm_base_url}. "
                "Start it first, for example: "
                f"`vllm serve {self.model_id} --host 0.0.0.0 --port 8000`."
            ) from exc

    async def _load_transformers(self) -> None:
        self._configure_verbose_logging()
        logger.info("[Qwen OCR] Importing Transformers and Torch")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        import torch

        dtype: str | Any = self.torch_dtype
        if self.torch_dtype != "auto":
            dtype = getattr(torch, self.torch_dtype)

        logger.info("[Qwen OCR] Loading model_id=%s device_map=%s", self.model_id, self.device_map)
        logger.info("[Qwen OCR] torch=%s cuda=%s", torch.__version__, torch.cuda.is_available())
        logger.info("[Qwen OCR] CUDA memory before load: %s", self._cuda_memory_summary())
        logger.info("[Qwen OCR] Loading processor for %s", self.model_id)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info("[Qwen OCR] Loading model weights for %s", self.model_id)
        if self.offload_folder:
            Path(self.offload_folder).mkdir(parents=True, exist_ok=True)
        model_kwargs = {
            "device_map": self.device_map,
            "dtype": dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "offload_buffers": self.offload_buffers,
        }
        if self.offload_folder:
            model_kwargs["offload_folder"] = self.offload_folder
        logger.info(
            "[Qwen OCR] from_pretrained kwargs=%s",
            {key: str(value) for key, value in model_kwargs.items()},
        )
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()
        logger.info("[Qwen OCR] CUDA memory after load: %s", self._cuda_memory_summary())
        logger.info("[Qwen OCR] Loaded model class=%s device=%s", self._model.__class__.__name__, self._model.device)

    async def unload(self) -> None:
        if self.provider == "vllm":
            logger.info("[Qwen OCR] vLLM provider selected; no local model to unload")
            return

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

    def _vllm_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.vllm_api_key and self.vllm_api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"
        return headers

    @staticmethod
    def _image_data_url(image: Any) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _generate_text_for_image(self, image: Any) -> str:
        if self.provider == "vllm":
            return self._generate_text_for_image_vllm(image)
        return self._generate_text_for_image_transformers(image)

    def _generate_text_for_image_vllm(self, image: Any) -> str:
        import httpx

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(image)},
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
        }
        response = httpx.post(
            f"{self.vllm_base_url}/chat/completions",
            headers=self._vllm_headers(),
            json=payload,
            timeout=self.vllm_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()

    def _generate_text_for_image_transformers(self, image: Any) -> str:
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
                "provider": self.provider,
                "pipeline": "vLLM OpenAI-compatible API"
                if self.provider == "vllm"
                else "AutoModelForImageTextToText",
                "model_id": self.model_id,
                "vllm_base_url": self.vllm_base_url if self.provider == "vllm" else None,
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
