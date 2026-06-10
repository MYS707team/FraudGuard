"""
FraudGuard — Data loading & splitting.

Responsibilities:
- Load ``creditcard.csv`` from ``RAW_DATA_PATH`` into a pandas DataFrame.
- Validate that all expected columns exist (clear error if file/columns missing).
- Split features (X) and target (y = 'Class').
- Stratified train/test split so the (tiny) fraud ratio is preserved in both sets.
- Save / load the processed arrays (.npy) to/from ``data/processed``.
"""
from __future__ import annotations

# --- import bootstrap: make project root importable as `src.*` ---------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw_data(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw credit-card CSV and validate its schema.

    Raises:
        FileNotFoundError: if the CSV is not present.
        ValueError: if expected columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download the Kaggle 'Credit Card Fraud Detection' dataset and place "
            "creditcard.csv in data/raw/. See README.md for instructions."
        )

    df = pd.read_csv(path)

    expected = set(config.FEATURE_COLUMNS) | {config.TARGET_COLUMN}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {sorted(missing)}.\n"
            f"Found columns: {list(df.columns)}"
        )
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix (X) from the target vector (y)."""
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].astype(int).copy()
    return X, y


def stratified_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split using settings from ``config``.

    Stratification keeps the fraud ratio (~0.172%) identical in both subsets,
    which is essential on such a strongly imbalanced dataset.
    """
    stratify = y if config.STRATIFY else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test


def get_train_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convenience: load CSV -> split X/y -> stratified train/test split."""
    df = load_raw_data()
    X, y = split_features_target(df)
    return stratified_split(X, y)


# --------------------------------------------------------------------------- #
# Processed array persistence
# --------------------------------------------------------------------------- #
def save_processed(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Save processed numpy arrays into ``data/processed``."""
    config.ensure_dirs()
    np.save(config.X_TRAIN_PATH, np.asarray(X_train))
    np.save(config.X_TEST_PATH, np.asarray(X_test))
    np.save(config.Y_TRAIN_PATH, np.asarray(y_train))
    np.save(config.Y_TEST_PATH, np.asarray(y_test))


def load_processed() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load processed numpy arrays from ``data/processed``."""
    for p in (
        config.X_TRAIN_PATH,
        config.X_TEST_PATH,
        config.Y_TRAIN_PATH,
        config.Y_TEST_PATH,
    ):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Processed array not found: {p}. Run the preprocessing/training "
                "pipeline first."
            )
    X_train = np.load(config.X_TRAIN_PATH)
    X_test = np.load(config.X_TEST_PATH)
    y_train = np.load(config.Y_TRAIN_PATH)
    y_test = np.load(config.Y_TEST_PATH)
    return X_train, X_test, y_train, y_test


def fraud_ratio(y: pd.Series | np.ndarray) -> float:
    """Return the fraction of fraud cases in a label array."""
    y = np.asarray(y)
    return float(y.mean()) if len(y) else 0.0


if __name__ == "__main__":
    df = load_raw_data()
    X, y = split_features_target(df)
    print("FraudGuard dataset summary")
    print("-" * 40)
    print(f"Dataset shape : {df.shape}")
    print(f"Fraud cases   : {int(y.sum())} / {len(y)}")
    print(f"Fraud ratio   : {fraud_ratio(y) * 100:.4f}%")

    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"Train shape   : {X_train.shape}  (fraud {fraud_ratio(y_train)*100:.4f}%)")
    print(f"Test shape    : {X_test.shape}  (fraud {fraud_ratio(y_test)*100:.4f}%)")
