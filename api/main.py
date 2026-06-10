"""
FraudGuard — FastAPI backend.

Endpoints:
    GET  /health      -> service status + whether the model is loaded
    POST /predict     -> validate a transaction, return the fraud decision
    GET  /model-info  -> winning model name and its metrics

Run locally:
    uvicorn api.main:app --reload --port 8000
Interactive Swagger docs: http://localhost:8000/docs
"""
from __future__ import annotations

# --- import bootstrap: make project root importable as `src.*` ---------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# -----------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src import config
from src.predict import FraudPredictor

app = FastAPI(
    title="FraudGuard API",
    description="Real-time credit-card fraud detection. Scores a transaction and "
    "returns Approve / Review / Block.",
    version="1.0.0",
)

# Load the predictor once at startup (None if artifacts are missing).
predictor: FraudPredictor | None = None


@app.on_event("startup")
def _load_model() -> None:
    global predictor
    try:
        predictor = FraudPredictor()
    except FileNotFoundError as exc:
        # Keep the API up so /health can report the problem clearly.
        print(f"[FraudGuard] Model not loaded: {exc}")
        predictor = None


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
def _v_field():
    return Field(..., description="PCA-anonymized feature")


class Transaction(BaseModel):
    """A single transaction. All fields are floats (Time, V1..V28, Amount)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                **{"Time": 0.0},
                **{f"V{i}": 0.0 for i in range(1, 29)},
                **{"Amount": 149.62},
            }
        }
    )

    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float; V26: float
    V27: float; V28: float
    Amount: float


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    decision: str
    model_used: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    """Liveness probe + model-loaded status."""
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model": predictor.model_name if predictor else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> dict:
    """Validate a transaction and return the fraud decision."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the pipeline first (python -m src.train).",
        )
    try:
        return predictor.predict(transaction.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # pragma: no cover - safety net
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.get("/model-info")
def model_info() -> dict:
    """Return the winning model name and its metrics."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the pipeline first (python -m src.train).",
        )
    return {
        "model": predictor.model_name,
        "metrics": predictor.metadata.get("metrics", {}),
        "strategy": predictor.metadata.get("strategy"),
        "threshold": predictor.metadata.get("threshold", config.DEFAULT_THRESHOLD),
        "approve_below": config.APPROVE_BELOW,
        "block_above": config.BLOCK_ABOVE,
        "trained_at": predictor.metadata.get("trained_at"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
