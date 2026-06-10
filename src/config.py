"""
FraudGuard — Central configuration.

All paths, split settings, decision thresholds and feature ordering live here so
that every other module imports its settings from a single source of truth.

Paths are built with ``pathlib`` relative to the project root, so the project
works regardless of the current working directory (local machine, Colab, CI...).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Project root & directories
# --------------------------------------------------------------------------- #
# config.py lives in <root>/src/, so the project root is one level up.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# --------------------------------------------------------------------------- #
# File paths
# --------------------------------------------------------------------------- #
RAW_DATA_PATH: Path = RAW_DATA_DIR / "creditcard.csv"

# Processed arrays
X_TRAIN_PATH: Path = PROCESSED_DIR / "X_train.npy"
X_TEST_PATH: Path = PROCESSED_DIR / "X_test.npy"
Y_TRAIN_PATH: Path = PROCESSED_DIR / "y_train.npy"
Y_TEST_PATH: Path = PROCESSED_DIR / "y_test.npy"

# Model artifacts
SCALER_PATH: Path = MODELS_DIR / "scaler.pkl"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.pkl"
METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
COMPARISON_PATH: Path = MODELS_DIR / "comparison_results.json"

# Evaluation plots
CONFUSION_MATRIX_PATH: Path = MODELS_DIR / "confusion_matrix.png"
PR_CURVE_PATH: Path = MODELS_DIR / "pr_curve.png"

# --------------------------------------------------------------------------- #
# Train / test split
# --------------------------------------------------------------------------- #
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
STRATIFY: bool = True  # keep the fraud ratio identical in train & test

# --------------------------------------------------------------------------- #
# Decision thresholds (fraud probability p in [0, 1])
# --------------------------------------------------------------------------- #
#   p < APPROVE_BELOW            -> APPROVE  (green)
#   APPROVE_BELOW <= p < BLOCK   -> REVIEW   (yellow, manual review)
#   p >= BLOCK_ABOVE             -> BLOCK    (red)
APPROVE_BELOW: float = 0.30
BLOCK_ABOVE: float = 0.70

# Default probability threshold for the binary is_fraud flag.
DEFAULT_THRESHOLD: float = 0.50

# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
# Which metric decides the "best" model. We use F1 (the harmonic mean of
# precision & recall) so the winner is *balanced* — it must both catch fraud
# (recall) AND avoid drowning analysts in false positives (precision). Ranking by
# PR-AUC alone can crown a model with great ranking but unusable precision at the
# operating threshold (e.g. Logistic Regression at 13% precision).
# Tie-breakers are applied in order after the primary metric.
SELECTION_METRIC: str = "f1"
SELECTION_TIEBREAKERS: tuple[str, ...] = ("pr_auc", "recall")

# --------------------------------------------------------------------------- #
# XGBoost training — early stopping (stop at the optimal number of rounds)
# --------------------------------------------------------------------------- #
# Instead of always training a fixed number of boosting rounds, XGBoost trains up
# to XGB_MAX_ROUNDS but stops as soon as the validation PR-AUC stops improving for
# XGB_EARLY_STOPPING_ROUNDS consecutive rounds — keeping the optimal-sized model.
VALIDATION_SIZE: float = 0.2          # inner split carved from TRAIN for early stop
XGB_MAX_ROUNDS: int = 1000            # upper bound on boosting rounds
XGB_EARLY_STOPPING_ROUNDS: int = 30   # stop after N rounds without improvement
XGB_LEARNING_RATE: float = 0.1
XGB_MAX_DEPTH: int = 5
XGB_VERBOSE_EVERY: int = 25           # log validation score every N rounds

# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #
# The exact column order expected by the dataset and the model.
# Time, then V1..V28, then Amount.
V_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS: list[str] = ["Time", *V_FEATURES, "Amount"]
TARGET_COLUMN: str = "Class"

# Columns that get scaled by the StandardScaler (V1..V28 are already PCA-scaled).
SCALED_COLUMNS: list[str] = ["Time", "Amount"]


def ensure_dirs() -> None:
    """Create the data/model directories if they do not exist yet."""
    for directory in (RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("FraudGuard configuration")
    print("-" * 40)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"RAW_DATA_PATH: {RAW_DATA_PATH}")
    print(f"MODELS_DIR   : {MODELS_DIR}")
    print(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"Target       : {TARGET_COLUMN}")
    print(f"Thresholds   : approve<{APPROVE_BELOW}, block>={BLOCK_ABOVE}")
