"""
FraudGuard — Preprocessing.

V1..V28 are already normalized via PCA, so only 'Time' and 'Amount' need scaling.

⚠️ Data-leakage rule
--------------------
The StandardScaler is **fit only on the training data**. We then *transform* the
test set (and any real-time transaction) with that already-fitted scaler. Fitting
on the full dataset (or on the test set) would leak information about the test
distribution into preprocessing — a classic and costly mistake. Fitting on train
only keeps the test set a truly unseen hold-out.

This module also provides:
- SMOTE oversampling applied ONLY to the training set (never the test set).
- A helper that returns class weights / scale_pos_weight for the class-weight
  strategy.
"""
from __future__ import annotations

# --- import bootstrap --------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# -----------------------------------------------------------------------------

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from src import config


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training data only ('Time' & 'Amount').

    Fitting exclusively on training data prevents data leakage.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[config.SCALED_COLUMNS])
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Transform 'Time' & 'Amount' of X with an already-fitted scaler."""
    X = X.copy()
    X[config.SCALED_COLUMNS] = scaler.transform(X[config.SCALED_COLUMNS])
    return X


def save_scaler(scaler: StandardScaler, path: Path | str = config.SCALER_PATH) -> None:
    """Persist the fitted scaler so the API/UI reuse the exact same transform."""
    config.ensure_dirs()
    joblib.dump(scaler, path)


def load_scaler(path: Path | str = config.SCALER_PATH) -> StandardScaler:
    """Load a previously saved scaler."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at '{path}'. Train the pipeline first (src/train.py)."
        )
    return joblib.load(path)


def preprocess_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame, save: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train, transform both sets, optionally persist the scaler."""
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(scaler, X_train)
    X_test_scaled = apply_scaler(scaler, X_test)
    if save:
        save_scaler(scaler)
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(
    X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority (fraud) class with SMOTE.

    ⚠️ Apply this to the TRAINING set ONLY. Never resample the test set, or your
    evaluation metrics become meaningless.
    """
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return np.asarray(X_res), np.asarray(y_res)


def get_scale_pos_weight(y_train: pd.Series | np.ndarray) -> float:
    """scale_pos_weight for XGBoost = (#negatives / #positives).

    This tells the model that a fraud (positive) error is far more costly,
    counteracting the imbalance without touching the data.
    """
    y = np.asarray(y_train)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(n_neg / n_pos) if n_pos else 1.0


def get_class_weights(y_train: pd.Series | np.ndarray) -> dict[int, float]:
    """Balanced class weights {0: w0, 1: w1} for LogReg / RandomForest.

    Equivalent to sklearn's class_weight='balanced'; returned explicitly so it can
    be logged / inspected.
    """
    y = np.asarray(y_train)
    n = len(y)
    classes = np.unique(y)
    return {int(c): float(n / (len(classes) * (y == c).sum())) for c in classes}


if __name__ == "__main__":
    from src.data_loader import get_train_test

    X_train, X_test, y_train, y_test = get_train_test()
    X_train_s, X_test_s, scaler = preprocess_train_test(X_train, X_test)
    print("Scaler fitted on train only and saved to:", config.SCALER_PATH)
    print("scale_pos_weight:", round(get_scale_pos_weight(y_train), 2))
    print("class_weights   :", get_class_weights(y_train))
    X_res, y_res = apply_smote(X_train_s, y_train)
    print(f"After SMOTE: {X_res.shape[0]} train rows (was {len(X_train)})")
