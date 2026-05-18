"""
Base class all OCR model wrappers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class OCRResult:
    model_name: str
    raw_text: str
    inference_time_ms: float
    avg_confidence: float
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_error(cls, model_name: str, error: str) -> "OCRResult":
        return cls(
            model_name=model_name,
            raw_text="",
            inference_time_ms=0.0,
            avg_confidence=0.0,
            error=error,
        )


class BaseOCRModel(ABC):
    name: str = "base"
    tier: int = 2  # 1 = VLM, 2 = Traditional

    @abstractmethod
    async def load(self) -> None:
        """Initialize model weights, tokenizer, etc."""

    @abstractmethod
    async def unload(self) -> None:
        """Free GPU/CPU memory."""

    @abstractmethod
    async def run(self, image_bytes: bytes) -> OCRResult:
        """Run inference and return structured OCRResult."""

    def _timer(self):
        return time.perf_counter()

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)
