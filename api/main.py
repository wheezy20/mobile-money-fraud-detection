"""
FastAPI application for mobile money fraud detection.

Serves predictions from the tuned Random Forest model produced in
Section D, Phase 4. All feature engineering is performed internally,
so callers submit raw transaction fields and receive a fraud probability.

Run locally with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Then visit:
    http://localhost:8000/docs   — interactive API documentation
    http://localhost:8000/health — readiness check
    http://localhost:8000/predict — POST endpoint for predictions
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Make src/ importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.preprocessing import engineer_features
from api.schemas import TransactionRequest, PredictionResponse, HealthResponse


MODEL_PATH = PROJECT_ROOT / "models" / "tuned_rf.pkl"
MODEL_VERSION = "tuned_rf_v1"
FEATURE_ORDER = [
    "amount_log",
    "drainage_ratio",
    "error_orig",
    "error_dest",
    "hour_of_day",
    "oldbalanceOrg",
    "oldbalanceDest",
    "type_encoded",
    "is_unusual_hour",
]


# Application state, populated at startup
state = {"model": None, "version": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup. Runs once before the API starts serving."""
    print(f"Loading model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    state["model"] = joblib.load(MODEL_PATH)
    state["version"] = MODEL_VERSION
    print(f"Model loaded: {type(state['model']).__name__} ({MODEL_VERSION})")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Mobile Money Fraud Detection API",
    description=(
        "Production REST API for fraud scoring of mobile money transactions. "
        "Built on the tuned Random Forest model documented in Section D of "
        "the DSA5202 project report."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness check. Returns ok if the model is loaded, degraded otherwise."""
    if state["model"] is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=state["version"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest) -> PredictionResponse:
    """
    Score a single transaction for fraud probability.

    Performs feature engineering (drainage_ratio, error_orig, etc.)
    and then runs the tuned Random Forest model to produce a probability.
    """
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet.",
        )

    # Convert the request to a one-row DataFrame for the engineer_features function
    raw = pd.DataFrame([{
        "step": transaction.step,
        "type": transaction.type,
        "amount": transaction.amount,
        "nameOrig": "C0",  # placeholder; not used by the 9-feature pipeline
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "newbalanceOrig": transaction.newbalanceOrig,
        "nameDest": "M0",  # placeholder
        "oldbalanceDest": transaction.oldbalanceDest,
        "newbalanceDest": transaction.newbalanceDest,
    }])

    # Apply the same feature engineering used during training
    engineered = engineer_features(raw)

    # Select features in the exact order the model expects
    X = engineered[FEATURE_ORDER]

    # Predict
    proba = state["model"].predict_proba(X)[0, 1]
    is_fraud = bool(proba >= 0.5)

    return PredictionResponse(
        fraud_probability=float(proba),
        is_fraud=is_fraud,
        threshold=0.5,
        model_version=state["version"],
    )