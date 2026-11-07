from pydantic import BaseModel, ConfigDict, Field

from app.config import DEFAULT_TOP_K, MAX_TOP_K


class PredictRequest(BaseModel):
    image_base64: str = Field(..., min_length=1, description="Base64-encoded JPEG or PNG bytes")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class Prediction(BaseModel):
    class_index: int
    class_name: str
    score: float


class TimingBreakdown(BaseModel):
    """Where the time in a request actually goes. The point of this phase."""
    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    timing: TimingBreakdown


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    # Pydantic v2 reserves the "model_" prefix for its own attributes, so fields
    # named model_* need this opt-out or they raise a warning at import time.
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_path: str


class MetricsResponse(BaseModel):
    requests_total: int
    errors_total: int
    window_size: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
