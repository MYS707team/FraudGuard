"""
FraudGuard — Evaluation & plotting.

Reusable helpers to:
- Build & save a confusion matrix (PNG).
- Build & save a precision-recall curve (PNG).
- Load ``comparison_results.json`` into a formatted pandas DataFrame for the UI.
"""
from __future__ import annotations

# --- import bootstrap --------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# -----------------------------------------------------------------------------

import json

import matplotlib

matplotlib.use("Agg")  # headless backend (works in Colab / servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
)

from src import config


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    path: Path | str = config.CONFUSION_MATRIX_PATH,
) -> Path:
    """Plot and save a confusion matrix as a PNG. Returns the saved path."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    config.ensure_dirs()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def plot_pr_curve(
    y_true,
    y_score,
    title: str = "Precision-Recall Curve",
    path: Path | str = config.PR_CURVE_PATH,
) -> Path:
    """Plot and save a precision-recall curve as a PNG. Returns the saved path."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#2563eb", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    config.ensure_dirs()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def comparison_dataframe(
    path: Path | str = config.COMPARISON_PATH,
) -> pd.DataFrame:
    """Load comparison_results.json into a tidy, display-ready DataFrame."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=["Model", "Strategy", "Precision", "Recall", "F1", "PR-AUC"]
        )
    data = json.loads(path.read_text())
    rows = []
    for model, r in data.items():
        rows.append(
            {
                "Model": model,
                "Strategy": r.get("strategy", ""),
                "Precision": r.get("precision"),
                "Recall": r.get("recall"),
                "F1": r.get("f1"),
                "PR-AUC": r.get("pr_auc"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    return df


def generate_plots_for_best_model() -> tuple[Path, Path]:
    """Regenerate confusion matrix & PR curve for the saved best model.

    Uses the processed test arrays and the persisted best model.
    """
    import joblib

    from src.data_loader import load_processed

    _, X_test, _, y_test = load_processed()
    model = joblib.load(config.BEST_MODEL_PATH)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= config.DEFAULT_THRESHOLD).astype(int)
    cm_path = plot_confusion_matrix(y_test, preds)
    pr_path = plot_pr_curve(y_test, proba)
    return cm_path, pr_path


if __name__ == "__main__":
    df = comparison_dataframe()
    print(df.to_string(index=False) if not df.empty else "No comparison results yet.")
    try:
        cm, pr = generate_plots_for_best_model()
        print(f"Saved: {cm}\nSaved: {pr}")
    except FileNotFoundError as exc:
        print(f"Plots skipped: {exc}")
