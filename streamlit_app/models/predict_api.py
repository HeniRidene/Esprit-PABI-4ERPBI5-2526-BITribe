"""
ESPRIT University Project — Rubric-Compliant FastAPI Demo
==========================================================
Features:
  - Loads a .pkl model (temp + humidity → prediction)
  - POST /predict  — main inference endpoint
  - GET  /health   — liveness probe
  - Saves every prediction to predictions.json (append mode)
  - Structured JSON logging on every request and error
  - CORS enabled for n8n / browser access

Run:
    uvicorn predict_api:app --reload --host 0.0.0.0 --port 8000

Model expected: models/weather_model.pkl  (joblib serialised sklearn model)
If the file is absent a LinearRegression fallback is used for demonstration.
"""

import json
import logging
import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────
#  Structured JSON Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":%(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("predict_api")

# ──────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "weather_model.pkl"          # Your .pkl file here
PREDICTIONS_FILE = BASE_DIR / "predictions.json"     # Output storage


# ──────────────────────────────────────────────────────────
#  Thread-safe model registry (lazy load + cache)
# ──────────────────────────────────────────────────────────
_model: Any = None
_lock = threading.Lock()


def get_model() -> Any:
    """Load and cache the .pkl model (thread-safe)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                if MODEL_PATH.exists():
                    logger.info(f'"Loading model from disk: {MODEL_PATH}"')
                    _model = joblib.load(MODEL_PATH)
                    logger.info('"Model loaded and cached successfully"')
                else:
                    # ── Fallback: Linear formula for demonstration ──
                    logger.warning(
                        f'"Model file not found at {MODEL_PATH}. '
                        f'Using LinearRegression fallback demo model."'
                    )
                    from sklearn.linear_model import LinearRegression
                    fallback = LinearRegression()
                    # Fit on synthetic data so predict() works
                    rng = np.random.default_rng(42)
                    X_demo = rng.uniform([0, 0], [50, 100], size=(200, 2))
                    y_demo = 0.6 * X_demo[:, 0] - 0.3 * X_demo[:, 1] + rng.normal(0, 1, 200)
                    fallback.fit(X_demo, y_demo)
                    _model = fallback
    return _model


# ──────────────────────────────────────────────────────────
#  Prediction storage (append to JSON file)
# ──────────────────────────────────────────────────────────
_file_lock = threading.Lock()


def save_prediction(record: dict) -> None:
    """Append a prediction record to predictions.json (thread-safe)."""
    with _file_lock:
        existing: list = []
        if PREDICTIONS_FILE.exists():
            try:
                with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing = []
        existing.append(record)
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    logger.info(f'"Prediction saved to {PREDICTIONS_FILE} (total records: {len(existing)})"')


# ──────────────────────────────────────────────────────────
#  FastAPI App
# ──────────────────────────────────────────────────────────
app = FastAPI(
    title="ESPRIT ML Prediction API",
    description=(
        "Rubric-compliant prediction endpoint for the ESPRIT university project. "
        "Features: temp + humidity → prediction. All results are stored in predictions.json."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
#  Pydantic schemas
# ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    temp: float = Field(
        ...,
        ge=-50.0,
        le=60.0,
        description="Temperature in °C (−50 to 60)",
        examples=[22.5],
    )
    humidity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative humidity in % (0–100)",
        examples=[65.0],
    )

    @field_validator("temp")
    @classmethod
    def temp_must_be_realistic(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("temp must be a finite number")
        return round(v, 4)

    @field_validator("humidity")
    @classmethod
    def humidity_must_be_realistic(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("humidity must be a finite number")
        return round(v, 4)


class PredictResponse(BaseModel):
    status: str
    prediction: float
    input: dict
    latency_ms: float
    timestamp: str
    model_source: str


# ──────────────────────────────────────────────────────────
#  Middleware — request/response logging
# ──────────────────────────────────────────────────────────
@app.middleware("http")
async def request_logger(request: Request, call_next):
    t0 = time.perf_counter()
    logger.info(
        f'"Incoming: {request.method} {request.url.path} '
        f'client={request.client.host if request.client else "unknown"}"'
    )
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f'"Completed: {request.method} {request.url.path} '
        f'status={response.status_code} latency_ms={ms:.2f}"'
    )
    return response


# ──────────────────────────────────────────────────────────
#  Global exception handler
# ──────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f'"Unhandled exception on {request.url.path}: {type(exc).__name__}: {exc}"')
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": f"Internal Server Error: {str(exc)}",
            "path": str(request.url.path),
        },
    )


# ──────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Liveness probe. Returns model source info."""
    model_loaded = MODEL_PATH.exists()
    total_predictions = 0
    if PREDICTIONS_FILE.exists():
        try:
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                total_predictions = len(json.load(f))
        except Exception:
            pass
    return {
        "status": "ok",
        "model_file_present": model_loaded,
        "model_source": "weather_model.pkl" if model_loaded else "fallback_linear_regression",
        "predictions_stored": total_predictions,
        "predictions_file": str(PREDICTIONS_FILE),
    }


@app.get("/predictions", tags=["Storage"])
async def list_predictions(limit: int = 50):
    """Return the last N predictions from storage."""
    if not PREDICTIONS_FILE.exists():
        return {"total": 0, "predictions": []}
    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        all_preds = json.load(f)
    return {"total": len(all_preds), "predictions": all_preds[-limit:]}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(body: PredictRequest):
    """
    Main inference endpoint.

    Request:  { "temp": 22.5, "humidity": 65.0 }
    Response: { "status": "success", "prediction": 12.3, "input": {...}, ... }

    The prediction is automatically saved to predictions.json.
    """
    t_start = time.perf_counter()
    model_loaded = MODEL_PATH.exists()
    model_source = "weather_model.pkl" if model_loaded else "fallback_linear_regression"

    logger.info(f'"Predict request — temp={body.temp} humidity={body.humidity}"')

    try:
        model = get_model()
        features_df = pd.DataFrame([[body.temp, body.humidity]], columns=["temp", "humidity"])
        raw = model.predict(features_df)

        prediction_value = float(raw[0]) if isinstance(raw, np.ndarray) else float(raw)
        latency_ms = round((time.perf_counter() - t_start) * 1000, 3)

        logger.info(
            f'"Prediction success — temp={body.temp} humidity={body.humidity} '
            f'prediction={prediction_value:.4f} latency_ms={latency_ms}"'
        )

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {"temp": body.temp, "humidity": body.humidity},
            "prediction": prediction_value,
            "latency_ms": latency_ms,
            "model_source": model_source,
            "status": "success",
        }
        save_prediction(record)

        return PredictResponse(
            status="success",
            prediction=prediction_value,
            input={"temp": body.temp, "humidity": body.humidity},
            latency_ms=latency_ms,
            timestamp=record["timestamp"],
            model_source=model_source,
        )

    except ValueError as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 3)
        logger.error(f'"ValueError in predict: {exc} latency_ms={latency_ms}"')
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 3)
        logger.error(f'"Unexpected error in predict: {type(exc).__name__}: {exc} latency_ms={latency_ms}"')
        raise HTTPException(status_code=500, detail=f"Pipeline failure: {str(exc)}")


# ──────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict_api:app", host="0.0.0.0", port=8000, reload=True)
