from typing import Any, Literal

from pydantic import BaseModel


TaskStatus = Literal["processing", "completed", "failed"]


class OCRTaskSubmitResponse(BaseModel):
    task_id: str
    status: TaskStatus
    filename: str | None = None


class OCRTaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    filename: str | None = None
    result: Any | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    api: str
    model_loaded: bool
    loaded_models: list[str]
    failed_models: dict[str, str]
    worker_running: bool
    queue_size: int
    database: dict[str, Any]
