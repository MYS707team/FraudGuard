"""
FraudGuard — Unit tests for the prediction service.

These tests do NOT require trained artifacts on disk. We build a ``FraudPredictor``
with a stub scaler (identity) and a stub model (returns a fixed probability), so
the logic is tested in isolation and runs in CI without the Kaggle dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.predict import FraudPredictor


class _IdentityScaler:
    """Stub scaler that returns its input unchanged."""

    def transform(self, X):
        return np.asarray(X, dtype=float)


class _StubModel:
    """Stub model returning a configurable fraud probability."""

    def __init__(self, fraud_p: float):
        self.fraud_p = fraud_p

    def predict_proba(self, X):
        n = len(X)
        return np.tile([1 - self.fraud_p, self.fraud_p], (n, 1))


def _make_predictor(fraud_p: float = 0.5) -> FraudPredictor:
    """Construct a FraudPredictor without loading files from disk."""
    predictor = object.__new__(FraudPredictor)
    predictor.scaler = _IdentityScaler()
    predictor.model = _StubModel(fraud_p)
    predictor.metadata = {"threshold": config.DEFAULT_THRESHOLD}
    predictor.model_name = "StubModel"
    predictor.feature_columns = config.FEATURE_COLUMNS
    return predictor


def _valid_transaction() -> dict:
    tx = {c: 0.0 for c in config.FEATURE_COLUMNS}
    tx["Amount"] = 100.0
    tx["Time"] = 0.0
    return tx


def test_predict_returns_expected_keys():
    predictor = _make_predictor(0.5)
    result = predictor.predict(_valid_transaction())
    assert set(result.keys()) == {
        "is_fraud",
        "fraud_probability",
        "risk_level",
        "decision",
        "model_used",
    }


@pytest.mark.parametrize(
    "p,expected_decision,expected_risk",
    [
        (0.29, "APPROVE", "LOW"),
        (0.30, "REVIEW", "MEDIUM"),
        (0.69, "REVIEW", "MEDIUM"),
        (0.70, "BLOCK", "HIGH"),
    ],
)
def test_decision_boundaries(p, expected_decision, expected_risk):
    predictor = _make_predictor(p)
    result = predictor.predict(_valid_transaction())
    assert result["decision"] == expected_decision
    assert result["risk_level"] == expected_risk


def test_decision_static_method():
    assert FraudPredictor._decision(0.0) == ("APPROVE", "LOW")
    assert FraudPredictor._decision(0.299) == ("APPROVE", "LOW")
    assert FraudPredictor._decision(0.30) == ("REVIEW", "MEDIUM")
    assert FraudPredictor._decision(0.699) == ("REVIEW", "MEDIUM")
    assert FraudPredictor._decision(0.70) == ("BLOCK", "HIGH")
    assert FraudPredictor._decision(1.0) == ("BLOCK", "HIGH")


def test_missing_field_raises():
    predictor = _make_predictor()
    tx = _valid_transaction()
    del tx["V1"]
    with pytest.raises(ValueError, match="Missing required fields"):
        predictor.predict(tx)


def test_non_numeric_field_raises():
    predictor = _make_predictor()
    tx = _valid_transaction()
    tx["Amount"] = "not-a-number"
    with pytest.raises(ValueError, match="must be a number"):
        predictor.predict(tx)


def test_non_dict_raises():
    predictor = _make_predictor()
    with pytest.raises(ValueError, match="must be a dictionary"):
        predictor.predict([1, 2, 3])
