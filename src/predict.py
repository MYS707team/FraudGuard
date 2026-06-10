"""
FraudGuard — Prediction service (heart of the system).

``FraudPredictor`` loads the scaler, best model and metadata once (cached in
memory) and turns a raw transaction dict into a decision. Both the FastAPI
backend and the Gradio UI ultimately rely on this class.

Decision logic (fraud probability p):
    p < 0.30          -> APPROVE  (LOW risk)
    0.30 <= p < 0.70  -> REVIEW   (MEDIUM risk)
    p >= 0.70         -> BLOCK    (HIGH risk)
"""
from __future__ import annotations

# --- import bootstrap --------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# -----------------------------------------------------------------------------

import json

import joblib
import pandas as pd

from src import config

# Importing the wrapper class so joblib can unpickle an Isolation-Forest best model.
from src.anomaly import IsolationForestProba  # noqa: F401


class FraudPredictor:
    """Loads artifacts once and scores transactions in real time."""

    def __init__(
        self,
        scaler_path: Path | str = config.SCALER_PATH,
        model_path: Path | str = config.BEST_MODEL_PATH,
        metadata_path: Path | str = config.METADATA_PATH,
    ):
        self.scaler = self._load(scaler_path, "scaler")
        self.model = self._load(model_path, "model")
        self.metadata = self._load_metadata(metadata_path)
        self.model_name = self.metadata.get("model", "unknown")
        self.feature_columns = self.metadata.get(
            "feature_columns", config.FEATURE_COLUMNS
        )

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load(path: Path | str, what: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Could not load {what} from '{path}'. Train the pipeline first "
                "by running `python -m src.train`."
            )
        return joblib.load(path)

    @staticmethod
    def _load_metadata(path: Path | str) -> dict:
        path = Path(path)
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate(self, transaction: dict) -> list[float]:
        """Validate and order the transaction into a feature vector."""
        if not isinstance(transaction, dict):
            raise ValueError("Transaction must be a dictionary of feature values.")

        missing = [c for c in self.feature_columns if c not in transaction]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        ordered: list[float] = []
        for col in self.feature_columns:
            value = transaction[col]
            try:
                ordered.append(float(value))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Field '{col}' must be a number, got {value!r}."
                )
        return ordered

    # ------------------------------------------------------------------ #
    # Core prediction
    # ------------------------------------------------------------------ #
    def _scale(self, ordered: list[float]) -> pd.DataFrame:
        """Apply the fitted scaler to the 'Time' & 'Amount' columns only.

        Returns a single-row DataFrame with named columns so the model receives
        the same feature names it was trained on.
        """
        row = pd.DataFrame([ordered], columns=self.feature_columns)
        row[config.SCALED_COLUMNS] = self.scaler.transform(row[config.SCALED_COLUMNS])
        return row

    @staticmethod
    def _decision(p: float) -> tuple[str, str]:
        """Map a probability to (decision, risk_level)."""
        if p < config.APPROVE_BELOW:
            return "APPROVE", "LOW"
        if p < config.BLOCK_ABOVE:
            return "REVIEW", "MEDIUM"
        return "BLOCK", "HIGH"

    def predict(self, transaction: dict) -> dict:
        """Score a single transaction and return the decision payload."""
        ordered = self._validate(transaction)
        row = self._scale(ordered)
        proba = float(self.model.predict_proba(row)[0, 1])
        decision, risk_level = self._decision(proba)
        return {
            "is_fraud": proba >= self.metadata.get("threshold", config.DEFAULT_THRESHOLD),
            "fraud_probability": round(proba, 4),
            "risk_level": risk_level,
            "decision": decision,
            "model_used": self.model_name,
        }


# Module-level lazy singleton so the API/UI don't reload artifacts per request.
_PREDICTOR: FraudPredictor | None = None


def get_predictor() -> FraudPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = FraudPredictor()
    return _PREDICTOR


if __name__ == "__main__":
    predictor = get_predictor()
    # Example: a neutral all-zeros transaction.
    example = {col: 0.0 for col in config.FEATURE_COLUMNS}
    example["Amount"] = 149.62
    example["Time"] = 0.0
    print("Example prediction:")
    print(json.dumps(predictor.predict(example), indent=2))
