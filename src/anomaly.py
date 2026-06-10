"""
FraudGuard — Anomaly-model adapter.

Isolation Forest is unsupervised and has no ``predict_proba``. To let it be
compared on the same footing (PR-AUC) and to keep a uniform prediction interface
for the deployed model, we wrap it so it exposes a ``predict_proba`` that returns
a pseudo fraud-probability in [0, 1].

The wrapper stores the min/max of the (inverted) anomaly scores observed on the
training data and min-max scales new scores into [0, 1]. Defining this as a real
class (not a closure) means joblib can pickle/unpickle the saved model.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestProba:
    """Adapter giving an IsolationForest a ``predict_proba``-compatible API.

    Higher ``-score_samples`` => more anomalous => higher fraud probability.
    """

    def __init__(self, iforest: IsolationForest, score_min: float, score_max: float):
        self.iforest = iforest
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        # Mimic a scikit-learn classifier attribute.
        self.classes_ = np.array([0, 1])

    def _anomaly_score(self, X) -> np.ndarray:
        # score_samples: the lower, the more abnormal. Invert so higher = fraud.
        return -self.iforest.score_samples(X)

    def fraud_proba(self, X) -> np.ndarray:
        scores = self._anomaly_score(X)
        span = self.score_max - self.score_min
        if span <= 0:
            return np.clip(scores, 0.0, 1.0)
        p = (scores - self.score_min) / span
        return np.clip(p, 0.0, 1.0)

    def predict_proba(self, X) -> np.ndarray:
        p = self.fraud_proba(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.fraud_proba(X) >= 0.5).astype(int)


def fit_isolation_forest_proba(
    iforest: IsolationForest, X_train_scaled
) -> IsolationForestProba:
    """Fit the score range on training data and return the wrapped model."""
    scores = -iforest.score_samples(X_train_scaled)
    return IsolationForestProba(iforest, scores.min(), scores.max())
