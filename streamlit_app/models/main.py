"""
ESPRIT ML Production Gateway — FastAPI Backend
===============================================
Serves 18 scikit-learn/XGBoost models across 3 actors with:
  - Lazy-loading model registry (loaded on first request, then cached)
  - Full pipeline logic per actor/task (features → encode → scale → predict)
  - Pydantic v2 input validation
  - Structured JSON logging on every request and every error
  - Consistent response schema: {status, actor, task, prediction, latency_ms}

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────
#  Logging configuration (JSON-style, thread-safe)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": %(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("esprit_ml")

# ─────────────────────────────────────────────
#  Base model path — use the actual OneDrive path
# ─────────────────────────────────────────────
MODEL_BASE = Path(r"C:\Users\sbiss\OneDrive - ESPRIT\Desktop\streamlit_app\models")

# ─────────────────────────────────────────────
#  Lazy model registry with thread-safe locking
# ─────────────────────────────────────────────
_registry: Dict[str, Any] = {}
_lock = threading.Lock()


def load_pkl(actor: str, filename: str) -> Any:
    """
    Load a .pkl file lazily from the registry.
    On first access the artifact is deserialized with joblib and cached.
    Subsequent calls return the in-memory object instantly.
    """
    key = f"{actor}/{filename}"
    if key not in _registry:
        with _lock:
            if key not in _registry:  # double-checked locking
                fpath = MODEL_BASE / actor / filename
                if not fpath.exists():
                    raise FileNotFoundError(f"Model file not found: {fpath}")
                logger.info(f'"Loading model from disk: {key}"')
                _registry[key] = joblib.load(fpath)
                logger.info(f'"Model cached: {key}"')
    return _registry[key]


# ─────────────────────────────────────────────
#  FastAPI application
# ─────────────────────────────────────────────
app = FastAPI(
    title="ESPRIT ML Production Gateway",
    description=(
        "Production-ready inference API for 18 ML models across 3 actors. "
        "Implements per-actor pipeline logic: feature ordering → encoding → scaling → prediction."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Pydantic models
# ─────────────────────────────────────────────
VALID_ACTORS = {"actor1", "actor2", "actor3"}
VALID_TASKS = {
    "actor1": {"co2", "nrj", "cluster"},
    "actor2": {"cancellation", "charge"},
    "actor3": {"severity", "risk", "anomaly"},
}


class PredictRequest(BaseModel):
    actor: str = Field(..., description="One of: actor1, actor2, actor3")
    task: str = Field(
        ...,
        description=(
            "actor1 → co2 | nrj | cluster | "
            "actor2 → cancellation | charge | "
            "actor3 → severity | risk | anomaly"
        ),
    )
    features: Dict[str, Any] = Field(
        ..., description="Raw input features as a flat JSON object"
    )

    @field_validator("actor")
    @classmethod
    def validate_actor(cls, v: str) -> str:
        if v not in VALID_ACTORS:
            raise ValueError(f"actor must be one of {sorted(VALID_ACTORS)}, got '{v}'")
        return v

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str, info) -> str:
        actor = info.data.get("actor")
        if actor and actor in VALID_TASKS:
            allowed = VALID_TASKS[actor]
            if v not in allowed:
                raise ValueError(
                    f"For {actor}, task must be one of {sorted(allowed)}, got '{v}'"
                )
        return v


class PredictResponse(BaseModel):
    status: str
    actor: str
    task: str
    prediction: Any
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────
#  Pipeline helpers
# ─────────────────────────────────────────────

def _reorder_columns(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """
    Ensure df has exactly the columns in feature_list, in the right order.
    Raises a 422-style ValueError if any required column is missing.
    """
    missing = set(feature_list) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required feature columns: {sorted(missing)}. "
            f"Expected: {feature_list}. Got: {list(df.columns)}"
        )
    return df[feature_list]


def _apply_mode_encoding(
    df: pd.DataFrame, encoder: Any, cat_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply a categorical encoder (LabelEncoder dict / OrdinalEncoder / etc.).
    Handles both a plain dict of {col: LabelEncoder} and sklearn encoders.
    """
    if isinstance(encoder, dict):
        df = df.copy()
        for col, le in encoder.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
    else:
        # Assume sklearn-compatible transformer
        cols = cat_cols or list(df.columns)
        df[cols] = encoder.transform(df[cols])
    return df


# ─────────────────────────────────────────────
#  Actor pipelines
# ─────────────────────────────────────────────

def _pipeline_actor1_co2(raw_df: pd.DataFrame) -> Any:
    """
    Pipeline: xgboost_features → mode_co2_encoding → clustering_scaler → xgboost_co2
    """
    # 1. Load feature schema and reorder
    feature_list: List[str] = load_pkl("actor1", "xgboost_features.pkl")
    df = _reorder_columns(raw_df, feature_list)
    logger.info(f'"actor1/co2 — feature order validated: {feature_list}"')

    # 2. Apply categorical encoding
    encoder = load_pkl("actor1", "mode_co2_encoding.pkl")
    df = _apply_mode_encoding(df, encoder)
    logger.info('"actor1/co2 — mode encoding applied"')

    # 3. Scale
    scaler = load_pkl("actor1", "clustering_scaler.pkl")
    scaled = scaler.transform(df)
    logger.info('"actor1/co2 — clustering scaler applied"')

    # 4. Predict
    model = load_pkl("actor1", "xgboost_co2.pkl")
    prediction = model.predict(scaled)
    return prediction


def _pipeline_actor1_nrj(raw_df: pd.DataFrame) -> Any:
    """
    Pipeline: xgboost_features → mode_co2_encoding → clustering_scaler → xgboost_nrj
    Shares the same preprocessing artefacts as co2.
    """
    feature_list: List[str] = load_pkl("actor1", "xgboost_features.pkl")
    df = _reorder_columns(raw_df, feature_list)
    logger.info(f'"actor1/nrj — feature order validated: {feature_list}"')

    encoder = load_pkl("actor1", "mode_co2_encoding.pkl")
    df = _apply_mode_encoding(df, encoder)
    logger.info('"actor1/nrj — mode encoding applied"')

    scaler = load_pkl("actor1", "clustering_scaler.pkl")
    scaled = scaler.transform(df)
    logger.info('"actor1/nrj — clustering scaler applied"')

    model = load_pkl("actor1", "xgboost_nrj.pkl")
    prediction = model.predict(scaled)
    return prediction


def _pipeline_actor1_cluster(raw_df: pd.DataFrame) -> Any:
    """
    Pipeline: clustering_scaler → kmeans_pollution_zones (no XGBoost, pure clustering)
    """
    scaler = load_pkl("actor1", "clustering_scaler.pkl")
    scaled = scaler.transform(raw_df)
    model = load_pkl("actor1", "kmeans_pollution_zones.pkl")
    prediction = model.predict(scaled)
    return prediction


def _pipeline_actor2_cancellation(raw_df: pd.DataFrame) -> Any:
    """
    Pipeline: xgboost_cancellation_features → (no scaler) → xgboost_cancellation
    """
    feature_list: List[str] = load_pkl("actor2", "xgboost_cancellation_features.pkl")
    df = _reorder_columns(raw_df, feature_list)
    logger.info(f'"actor2/cancellation — feature order validated: {feature_list}"')

    model = load_pkl("actor2", "xgboost_cancellation.pkl")
    prediction = model.predict(df)
    return prediction


def _pipeline_actor2_charge(raw_df: pd.DataFrame) -> Any:
    """
    Pipeline: xgboost_charge_features → charge_encoding → xgboost_charge
    """
    feature_list: List[str] = load_pkl("actor2", "xgboost_charge_features.pkl")
    df = _reorder_columns(raw_df, feature_list)
    logger.info(f'"actor2/charge — feature order validated: {feature_list}"')

    encoder = load_pkl("actor2", "charge_encoding.pkl")
    df = _apply_mode_encoding(df, encoder)
    logger.info('"actor2/charge — charge encoding applied"')

    model = load_pkl("actor2", "xgboost_charge.pkl")
    prediction = model.predict(df)
    return prediction


def _pipeline_actor3_severity(raw_df: pd.DataFrame) -> Any:
    """
    Pipeline: rf_severity_features → (no scaler) → rf_severity
    """
    feature_list: List[str] = load_pkl("actor3", "rf_severity_features.pkl")
    df = _reorder_columns(raw_df, feature_list)
    logger.info(f'"actor3/severity — feature order validated: {feature_list}"')

    model = load_pkl("actor3", "rf_severity.pkl")
    prediction = model.predict(df)
    return prediction


def _pipeline_actor3_risk(raw_df: pd.DataFrame) -> Any:
    """
    Pipeline: kmeans_features → kmeans_scaler → kmeans_risk
    """
    feature_list: List[str] = load_pkl("actor3", "kmeans_features.pkl")
    df = _reorder_columns(raw_df, feature_list)
    logger.info(f'"actor3/risk — feature order validated: {feature_list}"')

    scaler = load_pkl("actor3", "kmeans_scaler.pkl")
    scaled = scaler.transform(df)
    logger.info('"actor3/risk — kmeans scaler applied"')

    model = load_pkl("actor3", "kmeans_risk.pkl")
    prediction = model.predict(scaled)
    return prediction


def _pipeline_actor3_anomaly(raw_df: pd.DataFrame) -> Any:
    """
    Pipeline: anomaly_features → anomaly_scaler → isolation_forest
    Returns  1 (normal) or -1 (anomaly).
    """
    feature_list: List[str] = load_pkl("actor3", "anomaly_features.pkl")
    df = _reorder_columns(raw_df, feature_list)
    logger.info(f'"actor3/anomaly — feature order validated: {feature_list}"')

    scaler = load_pkl("actor3", "anomaly_scaler.pkl")
    scaled = scaler.transform(df)
    logger.info('"actor3/anomaly — anomaly scaler applied"')

    model = load_pkl("actor3", "isolation_forest.pkl")
    prediction = model.predict(scaled)
    return prediction


# ─────────────────────────────────────────────
#  Router dispatch table
# ─────────────────────────────────────────────
PIPELINE_ROUTER = {
    ("actor1", "co2"):          _pipeline_actor1_co2,
    ("actor1", "nrj"):          _pipeline_actor1_nrj,
    ("actor1", "cluster"):      _pipeline_actor1_cluster,
    ("actor2", "cancellation"): _pipeline_actor2_cancellation,
    ("actor2", "charge"):       _pipeline_actor2_charge,
    ("actor3", "severity"):     _pipeline_actor3_severity,
    ("actor3", "risk"):         _pipeline_actor3_risk,
    ("actor3", "anomaly"):      _pipeline_actor3_anomaly,
}


# ─────────────────────────────────────────────
#  Middleware — request/response logging
# ─────────────────────────────────────────────
@app.middleware("http")
async def request_logger(request: Request, call_next):
    t0 = time.perf_counter()
    logger.info(
        f'"Incoming request: {request.method} {request.url.path} '
        f'client={request.client.host if request.client else "unknown"}"'
    )
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f'"Completed: {request.method} {request.url.path} '
        f'status={response.status_code} latency_ms={elapsed_ms:.2f}"'
    )
    return response


# ─────────────────────────────────────────────
#  Global exception handler
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe — returns 200 when the server is up."""
    return {
        "status": "ok",
        "loaded_models": list(_registry.keys()),
        "total_cached": len(_registry),
    }


@app.get("/models", tags=["System"])
async def list_models():
    """Returns all supported actor/task combinations."""
    return {
        "pipelines": [
            {"actor": a, "task": t} for (a, t) in PIPELINE_ROUTER.keys()
        ]
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Main inference endpoint.

    Request body:
        {
            "actor": "actor1",
            "task": "co2",
            "features": { "feature_a": 1.2, "feature_b": "urban", ... }
        }

    Response:
        {
            "status": "success",
            "actor": "actor1",
            "task": "co2",
            "prediction": 45.3,
            "latency_ms": 12.4,
            "metadata": { "pipeline_key": "actor1/co2" }
        }
    """
    t_start = time.perf_counter()
    pipeline_key = (request.actor, request.task)

    logger.info(
        f'"Predict request — actor={request.actor} task={request.task} '
        f'n_features={len(request.features)}"'
    )

    pipeline_fn = PIPELINE_ROUTER.get(pipeline_key)
    if pipeline_fn is None:
        logger.error(f'"Unknown pipeline: {pipeline_key}"')
        raise HTTPException(
            status_code=404,
            detail=f"No pipeline registered for actor='{request.actor}' task='{request.task}'.",
        )

    try:
        raw_df = pd.DataFrame([request.features])
        raw_prediction = pipeline_fn(raw_df)

        # Normalise output: always return a scalar or list
        if isinstance(raw_prediction, np.ndarray):
            result = raw_prediction.tolist()
            scalar_result = result[0] if len(result) == 1 else result
        else:
            scalar_result = raw_prediction

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f'"Prediction success — actor={request.actor} task={request.task} '
            f'prediction={scalar_result} latency_ms={latency_ms:.2f}"'
        )

        return PredictResponse(
            status="success",
            actor=request.actor,
            task=request.task,
            prediction=scalar_result,
            latency_ms=round(latency_ms, 3),
            metadata={"pipeline_key": f"{request.actor}/{request.task}"},
        )

    except (FileNotFoundError, ValueError) as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.error(
            f'"Pipeline error [{type(exc).__name__}] — actor={request.actor} '
            f'task={request.task} error="{exc}" latency_ms={latency_ms:.2f}"'
        )
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.error(
            f'"Unexpected pipeline error — actor={request.actor} '
            f'task={request.task} error="{exc}" latency_ms={latency_ms:.2f}"'
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal pipeline failure: {str(exc)}",
        )


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)