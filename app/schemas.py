from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class Language(str, Enum):
    ALL = "all"
    ENGLISH = "en"
    ARABIC = "ar"
    HINDI = "hi"
    PUNJABI = "pa"


class WordDetail(BaseModel):
    text: str
    confidence: float
    bbox: Optional[List[int]] = None


class LineDetail(BaseModel):
    page_index: int
    line_index: int
    text: str
    raw_text: str
    confidence: float
    bbox: Optional[List[int]] = None
    bbox_source: Dict[str, Any] = Field(default_factory=dict)
    bbox_valid: bool = True
    status: Optional[str] = None
    noise_score: Optional[float] = None
    filter_reason: Optional[str] = None
    exclude_reason: Optional[str] = None


class PageDetail(BaseModel):
    page_index: int
    text: str
    raw_text: str
    lines: List[LineDetail] = Field(default_factory=list)
    accepted_lines: List[LineDetail] = Field(default_factory=list)
    review_lines: List[LineDetail] = Field(default_factory=list)
    flagged_lines: List[LineDetail] = Field(default_factory=list)
    filtered_lines: List[LineDetail] = Field(default_factory=list)
    excluded_noise_lines: List[LineDetail] = Field(default_factory=list)
    excluded_lines: List[LineDetail] = Field(default_factory=list)
    per_line_confidence: List[Dict[str, Any]] = Field(default_factory=list)
    per_line_noise_score: List[Dict[str, Any]] = Field(default_factory=list)
    layout_mode: str = "text"
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    totals: Dict[str, Any] = Field(default_factory=dict)
    avg_confidence: float = 0.0


class ModelResult(BaseModel):
    model_name: str
    language: str
    final_text: str = ""
    text: str = ""
    raw_text: str
    selected_variant: Optional[str] = None
    confidence_score: float = 0.0
    debug_lines: List[Dict[str, Any]] = Field(default_factory=list)
    debug_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    layout_mode: str = "text"
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    totals: Dict[str, Any] = Field(default_factory=dict)
    rejected_bbox_lines: List[Dict[str, Any]] = Field(default_factory=list)
    corrected_bbox_lines: List[Dict[str, Any]] = Field(default_factory=list)
    words: List[WordDetail]
    pages: List[PageDetail] = Field(default_factory=list)
    accepted_lines: List[LineDetail] = Field(default_factory=list)
    review_lines: List[LineDetail] = Field(default_factory=list)
    flagged_lines: List[LineDetail] = Field(default_factory=list)
    excluded_noise_lines: List[LineDetail] = Field(default_factory=list)
    excluded_lines: List[LineDetail] = Field(default_factory=list)
    per_line_confidence: List[Dict[str, Any]] = Field(default_factory=list)
    per_line_noise_score: List[Dict[str, Any]] = Field(default_factory=list)
    inference_time_ms: float
    avg_confidence: float
    error: Optional[str] = None
    quality: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict = Field(default_factory=dict)


class OCRRequest(BaseModel):
    language: Language
    models: Optional[List[str]] = Field(
        default=None,
        description="Specific model names to run. If null, runs all loaded models.",
    )


class OCRResponse(BaseModel):
    filename: str
    language: str
    final_text: str = ""
    text: str = ""
    raw_text: str = ""
    selected_variant: Optional[str] = None
    confidence_score: float = 0.0
    debug_lines: List[Dict[str, Any]] = Field(default_factory=list)
    debug_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    layout_mode: str = "text"
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    totals: Dict[str, Any] = Field(default_factory=dict)
    rejected_bbox_lines: List[Dict[str, Any]] = Field(default_factory=list)
    corrected_bbox_lines: List[Dict[str, Any]] = Field(default_factory=list)
    pages: List[PageDetail] = Field(default_factory=list)
    accepted_lines: List[LineDetail] = Field(default_factory=list)
    review_lines: List[LineDetail] = Field(default_factory=list)
    flagged_lines: List[LineDetail] = Field(default_factory=list)
    excluded_noise_lines: List[LineDetail] = Field(default_factory=list)
    excluded_lines: List[LineDetail] = Field(default_factory=list)
    per_line_confidence: List[Dict[str, Any]] = Field(default_factory=list)
    per_line_noise_score: List[Dict[str, Any]] = Field(default_factory=list)
    avg_confidence: float = 0.0
    quality: Dict[str, Any] = Field(default_factory=dict)
    results: List[ModelResult]
    models_run: int
    total_time_ms: float


# --- Benchmark (with ground truth) ---

class MetricsDetail(BaseModel):
    cer: float = Field(description="Character Error Rate (lower=better)")
    wer: float = Field(description="Word Error Rate (lower=better)")
    ned: float = Field(description="Normalized Edit Distance (lower=better)")
    char_precision: float
    char_recall: float
    char_f1: float
    word_precision: float
    word_recall: float
    word_f1: float
    overall_accuracy: float = Field(description="Blended OCR accuracy score from 0 to 1 (higher=better)")
    exact_match: bool


class BenchmarkModelResult(BaseModel):
    model_name: str
    language: str
    raw_text: str
    inference_time_ms: float
    avg_confidence: float
    metrics: MetricsDetail
    error: Optional[str] = None


class BenchmarkResponse(BaseModel):
    filename: str
    language: str
    ground_truth: str
    results: List[BenchmarkModelResult]
    best_model_cer: str
    best_model_wer: str
    best_model_f1: str
    total_time_ms: float


# --- Model info ---

class ModelInfo(BaseModel):
    name: str
    tier: int
    supported_languages: List[str]
    loaded: bool


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    total_loaded: int


class OCRRunOptionsResponse(BaseModel):
    loaded_models: List[ModelInfo]
    available_model_names: List[str]
    available_languages: List[Language]
    preset_model_groups: Dict[str, List[str]]
