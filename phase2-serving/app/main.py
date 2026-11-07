import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from app.config import METRICS_WINDOW, MODEL_PATH
from app.inference import BadImageError, Classifier
from app.logging_utils import get_logger
from app.schemas import (HealthResponse, MetricsResponse, PredictRequest, PredictResponse,
                         ReadyResponse)

logger = get_logger(__name__)

classifier = Classifier()
_latencies: deque[float] = deque(maxlen=METRICS_WINDOW)
_requests_total = 0
_errors_total = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything before `yield` runs once at startup, not per request.
    logger.info("Startup: loading model...")
    started = time.perf_counter()
    classifier.load()
    logger.info("Startup: model ready in %.0f ms", (time.perf_counter() - started) * 1000)
    yield
    logger.info("Shutdown")


app = FastAPI(
    title="ResNet-18 INT8 Inference Service",
    description="Phase 2: serving the quantized model from Phase 1.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness: the process is up. Says nothing about the model."""
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse)
def ready():
    """Readiness: the model is loaded and this instance can actually serve traffic."""
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ready",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
    }


# Plain `def`, not `async def`. ONNX inference is blocking CPU work, so FastAPI
# runs this in its threadpool and the event loop stays free for other requests.
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    global _requests_total, _errors_total
    _requests_total += 1
    try:
        predictions, timing = classifier.predict(request.image_base64, request.top_k)
    except BadImageError as exc:
        _errors_total += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _latencies.append(timing["total_ms"])
    return {"predictions": predictions, "timing": timing}


# Deliberately wrong, kept for the Day 2 experiment. Blocking CPU work inside an
# `async def` occupies the single event loop thread, so concurrent requests
# serialize instead of overlapping. Compare this against /predict under load.
@app.post("/predict-blocking", response_model=PredictResponse, include_in_schema=False)
async def predict_blocking(request: PredictRequest):
    try:
        predictions, timing = classifier.predict(request.image_base64, request.top_k)
    except BadImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"predictions": predictions, "timing": timing}


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    window = list(_latencies)
    percentiles = {"p50_ms": None, "p95_ms": None, "p99_ms": None}
    if window:
        values = np.array(window)
        percentiles = {
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
        }
    return {
        "requests_total": _requests_total,
        "errors_total": _errors_total,
        "window_size": len(window),
        **percentiles,
    }
