import asyncio
import logging
from typing import Literal
from uuid import uuid4

from app.core.model_registry import PADDLEOCR_VL_MODEL_NAME, ModelRegistry
from app.core.task_store import OCRTaskStore

logger = logging.getLogger(__name__)


TaskStatus = Literal["processing", "completed", "failed"]


class OCRTaskManager:
    def __init__(self, registry: ModelRegistry, store: OCRTaskStore):
        self.registry = registry
        self.store = store
        self.queue: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker())

    def is_worker_running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    def queue_size(self) -> int:
        return self.queue.qsize()

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    async def submit(self, image_bytes: bytes, filename: str | None) -> dict:
        task_id = str(uuid4())
        task = await asyncio.to_thread(self.store.create_task, task_id=task_id, filename=filename)
        await self.queue.put((task_id, image_bytes))
        return task

    async def get(self, task_id: str) -> dict | None:
        return await asyncio.to_thread(self.store.get_task, task_id)

    async def _run_worker(self) -> None:
        while True:
            task_id, image_bytes = await self.queue.get()
            try:
                task = await asyncio.to_thread(self.store.get_task, task_id)
                if task is None:
                    continue

                await asyncio.to_thread(self.store.update_task, task_id, status="processing")
                model = self.registry.get(PADDLEOCR_VL_MODEL_NAME)
                if model is None:
                    failure_reason = getattr(self.registry, "failed_models", {}).get(PADDLEOCR_VL_MODEL_NAME)
                    raise RuntimeError(
                        f"Configured OCR model '{PADDLEOCR_VL_MODEL_NAME}' is not loaded at startup."
                        + (f" Startup error: {failure_reason}" if failure_reason else "")
                    )

                logger.info("[OCR Task] Running task_id=%s file=%s", task_id, task.get("filename"))
                result = await model.run_raw_json(image_bytes)
                await asyncio.to_thread(
                    self.store.update_task,
                    task_id,
                    status="completed",
                    result=result,
                    error=None,
                )
            except Exception as exc:
                logger.exception("[OCR Task] Failed task_id=%s: %s", task_id, exc)
                await asyncio.to_thread(
                    self.store.update_task,
                    task_id,
                    status="failed",
                    result=None,
                    error=str(exc),
                )
            finally:
                self.queue.task_done()
