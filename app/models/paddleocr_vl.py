import argparse
import asyncio
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from app.models.base import BaseOCRModel, OCRResult

logger = logging.getLogger(__name__)


class PaddleOCRVLModel(BaseOCRModel):
    """
    Official PaddleOCR-VL document parsing wrapper.

    The official pipeline returns generated block content and layout metadata,
    but it does not expose calibrated word confidence scores.
    """

    name = "paddleocr_vl"
    tier = 1

    def __init__(
        self,
        device: str = "gpu:0",
        pipeline_version: str = "v1",
        use_layout_detection: bool = True,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_chart_recognition: bool = False,
        use_seal_recognition: bool = False,
        use_ocr_for_image_block: bool = False,
        format_block_content: bool = True,
        merge_layout_blocks: bool = True,
    ):
        self.device = device
        self.pipeline_version = pipeline_version
        self.use_layout_detection = use_layout_detection
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_chart_recognition = use_chart_recognition
        self.use_seal_recognition = use_seal_recognition
        self.use_ocr_for_image_block = use_ocr_for_image_block
        self.format_block_content = format_block_content
        self.merge_layout_blocks = merge_layout_blocks
        self._pipeline: Any | None = None
        self._predict_lock = threading.Lock()

    async def load(self) -> None:
        from paddleocr import PaddleOCRVL

        logger.info(
            "[PaddleOCR-VL] Loading pipeline_version=%s device=%s",
            self.pipeline_version,
            self.device,
        )
        self._pipeline = PaddleOCRVL(
            pipeline_version=self.pipeline_version,
            device=self.device,
            use_layout_detection=self.use_layout_detection,
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_doc_unwarping=self.use_doc_unwarping,
            use_chart_recognition=self.use_chart_recognition,
            use_seal_recognition=self.use_seal_recognition,
            use_ocr_for_image_block=self.use_ocr_for_image_block,
            format_block_content=self.format_block_content,
            merge_layout_blocks=self.merge_layout_blocks,
        )
        logger.info("[PaddleOCR-VL] Loaded")

    async def unload(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        del pipeline
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except Exception:
            logger.debug("[PaddleOCR-VL] CUDA cache cleanup skipped", exc_info=True)

    @staticmethod
    def _temp_suffix(image_bytes: bytes) -> str:
        head = image_bytes[:16]
        if head.startswith(b"%PDF"):
            return ".pdf"
        if head.startswith(b"\xff\xd8"):
            return ".jpg"
        if head.startswith(b"\x89PNG"):
            return ".png"
        if head.startswith((b"II*\x00", b"MM\x00*")):
            return ".tiff"
        if head.startswith(b"RIFF") and b"WEBP" in image_bytes[:32]:
            return ".webp"
        return ".bin"

    @staticmethod
    def _result_json(result: Any) -> dict:
        if hasattr(result, "json"):
            json_data = result.json
            if isinstance(json_data, dict):
                return json_data.get("res", json_data)
        if isinstance(result, dict):
            return result.get("res", result)
        return {}

    @staticmethod
    def _result_markdown(result: Any) -> dict:
        if not hasattr(result, "markdown"):
            return {}
        markdown = result.markdown
        if not isinstance(markdown, dict):
            return {}
        return {
            "page_index": markdown.get("page_index"),
            "input_path": markdown.get("input_path"),
            "markdown_texts": markdown.get("markdown_texts", ""),
        }

    @staticmethod
    def _block_to_line(page_index: int, line_index: int, block: dict) -> dict:
        text = str(block.get("block_content", ""))
        bbox = block.get("block_bbox")
        return {
            "page_index": page_index,
            "line_index": line_index,
            "text": text,
            "raw_text": text,
            "confidence": 0.0,
            "bbox": bbox,
            "bbox_source": {"source": "paddleocr_vl_block_bbox"},
            "bbox_valid": bbox is not None,
            "status": str(block.get("block_label", "block")),
        }

    def _metadata_from_results(self, results: list[Any]) -> tuple[str, dict]:
        pages: list[dict] = []
        markdown_pages: list[dict] = []
        all_blocks: list[dict] = []
        tables: list[dict] = []
        text_parts: list[str] = []

        for fallback_index, result in enumerate(results):
            res = self._result_json(result)
            markdown = self._result_markdown(result)
            if markdown:
                markdown_pages.append(markdown)

            page_index = res.get("page_index")
            if page_index is None:
                page_index = fallback_index
            page_index = int(page_index)

            blocks = res.get("parsing_res_list", []) or []
            lines = [
                self._block_to_line(page_index, line_index, block)
                for line_index, block in enumerate(blocks)
            ]
            page_text = "\n".join(line["text"] for line in lines if line["text"])
            text_parts.append(page_text)

            page_tables = [
                {
                    "page_index": page_index,
                    "block_index": block_index,
                    "content": block.get("block_content", ""),
                    "bbox": block.get("block_bbox"),
                    "label": block.get("block_label"),
                }
                for block_index, block in enumerate(blocks)
                if str(block.get("block_label", "")).lower() == "table"
            ]
            tables.extend(page_tables)

            for block_index, block in enumerate(blocks):
                block_copy = dict(block)
                block_copy["page_index"] = page_index
                block_copy["block_index"] = block_index
                all_blocks.append(block_copy)

            pages.append(
                {
                    "page_index": page_index,
                    "text": page_text,
                    "raw_text": page_text,
                    "lines": lines,
                    "accepted_lines": lines,
                    "layout_mode": "paddleocr_vl_document_parsing",
                    "tables": page_tables,
                    "totals": {
                        "block_count": len(blocks),
                        "table_count": len(page_tables),
                    },
                    "avg_confidence": 0.0,
                    "vl_json": res,
                    "markdown": markdown,
                }
            )

        raw_text = "\n\n".join(text for text in text_parts if text)
        metadata = {
            "page_count": len(pages),
            "final_text": raw_text,
            "raw_text_before_correction": raw_text,
            "selected_variant": f"official_paddleocr_vl_{self.pipeline_version}",
            "confidence_score": 0.0,
            "layout_mode": "paddleocr_vl_document_parsing",
            "pages": pages,
            "tables": tables,
            "debug_blocks": all_blocks,
            "markdown_pages": markdown_pages,
            "markdown_text": "\n\n".join(
                page.get("markdown_texts", "") for page in markdown_pages if page.get("markdown_texts")
            ),
            "quality": {
                "status": "empty" if not raw_text.strip() else "generated",
                "warnings": ["no_text_detected"] if not raw_text.strip() else [],
                "line_count": sum(len(page["lines"]) for page in pages),
                "char_count": len(raw_text.strip()),
                "avg_confidence": 0.0,
                "confidence_note": "PaddleOCR-VL does not expose calibrated word confidence scores.",
            },
            "generation": {
                "pipeline": "PaddleOCRVL",
                "pipeline_version": self.pipeline_version,
                "device": self.device,
                "format_block_content": self.format_block_content,
            },
        }
        return raw_text, metadata

    def _run_raw_json_sync(self, image_bytes: bytes) -> dict | list[dict]:
        if self._pipeline is None:
            raise RuntimeError("PaddleOCR-VL model is not initialized. Call load() first.")

        temp_path = None
        try:
            suffix = self._temp_suffix(image_bytes)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name

            with self._predict_lock:
                results = self._pipeline.predict(
                    temp_path,
                    format_block_content=self.format_block_content,
                    merge_layout_blocks=self.merge_layout_blocks,
                )
            raw_results = [
                result.json if hasattr(result, "json") else {"res": result}
                for result in results
            ]
            if len(raw_results) == 1:
                return raw_results[0]
            return raw_results
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass

    async def run_raw_json(self, image_bytes: bytes) -> dict | list[dict]:
        return await asyncio.to_thread(self._run_raw_json_sync, image_bytes)

    def _run_sync(self, image_bytes: bytes) -> OCRResult:
        if self._pipeline is None:
            raise RuntimeError("PaddleOCR-VL model is not initialized. Call load() first.")

        temp_path = None
        try:
            start = self._timer()
            suffix = self._temp_suffix(image_bytes)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name

            with self._predict_lock:
                results = self._pipeline.predict(
                    temp_path,
                    format_block_content=self.format_block_content,
                    merge_layout_blocks=self.merge_layout_blocks,
                )
            raw_text, metadata = self._metadata_from_results(results)
            elapsed_ms = self._elapsed_ms(start)

            return OCRResult(
                model_name=self.name,
                raw_text=raw_text,
                inference_time_ms=elapsed_ms,
                avg_confidence=0.0,
                metadata=metadata,
            )
        except Exception as exc:
            logger.exception("[PaddleOCR-VL] Inference error: %s", exc)
            return OCRResult.from_error(
                self.name,
                str(exc),
            )
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass

    async def run(self, image_bytes: bytes) -> OCRResult:
        return await asyncio.to_thread(self._run_sync, image_bytes)


def main():
    parser = argparse.ArgumentParser(description="Run PaddleOCR-VL using the official PaddleOCR pipeline.")
    parser.add_argument(
        "--image-path",
        "-i",
        required=True,
        help="Path to the image or PDF to parse.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output/paddleocr_vl",
        help="Directory where JSON and Markdown outputs will be saved.",
    )
    parser.add_argument(
        "--device",
        default="gpu:0",
        help="Inference device, for example gpu:0 or cpu.",
    )
    parser.add_argument(
        "--pipeline-version",
        default="v1",
        choices=["v1", "v1.5"],
        help="PaddleOCR-VL pipeline version. v1 uses the original PaddleOCR-VL-0.9B model.",
    )
    parser.add_argument(
        "--use-layout-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable layout detection.",
    )
    parser.add_argument(
        "--use-doc-orientation-classify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable document orientation classification.",
    )
    parser.add_argument(
        "--use-doc-unwarping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable document unwarping.",
    )
    args = parser.parse_args()

    from paddleocr import PaddleOCRVL

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PaddleOCRVL(
        pipeline_version=args.pipeline_version,
        device=args.device,
        use_layout_detection=args.use_layout_detection,
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        format_block_content=True,
    )
    output = pipeline.predict(args.image_path, format_block_content=True)

    for res in output:
        res.print()
        res.save_to_json(save_path=str(output_dir))
        res.save_to_markdown(save_path=str(output_dir))

    print(f"\nSaved PaddleOCR-VL outputs to: {output_dir}")


if __name__ == "__main__":
    main()
