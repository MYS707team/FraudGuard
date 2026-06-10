"""
FraudGuard — Model training & comparison.

Trains and compares four models on the (untouched) test set:

  1. Logistic Regression  (class_weight='balanced')
  2. Random Forest         (class_weight='balanced')
  3. XGBoost               (scale_pos_weight)  + an XGBoost variant on SMOTE data
  4. Isolation Forest      (unsupervised anomaly detection)

Why not accuracy? On a dataset that is 99.8% non-fraud, always predicting
"normal" scores 99.8% accuracy yet catches zero fraud. We therefore rank models
by **PR-AUC** (Precision-Recall AUC, robust on imbalanced data) with **Recall**
as the tie-breaker, and also report Precision / Recall / F1.

Artifacts written to ``models/``:
  - best_model.pkl            the winning model (always exposes predict_proba)
  - model_metadata.json       winning model name, threshold, metrics
  - comparison_results.json   metrics for every model (rendered in the UI)
"""
from __future__ import annotations

# --- import bootstrap --------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# -----------------------------------------------------------------------------

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src import config
from src.anomaly import fit_isolation_forest_proba
from src.data_loader import get_train_test, save_processed
from src.preprocessing import (
    apply_smote,
    get_scale_pos_weight,
    preprocess_train_test,
)


def _proba(model, X) -> np.ndarray:
    """Return fraud probability (class 1) for any model with predict_proba."""
    return model.predict_proba(X)[:, 1]


def _evaluate(model, X_test, y_test, threshold: float = config.DEFAULT_THRESHOLD) -> dict:
    """Compute Precision / Recall / F1 / PR-AUC on the test set."""
    proba = _proba(model, X_test)
    preds = (proba >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
    }


def train_all_models(
    X_train_s, X_test_s, y_train, y_test
) -> tuple[dict, dict]:
    """Train every model, return (fitted_models, comparison_metrics)."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    spw = get_scale_pos_weight(y_train)

    models: dict = {}
    results: dict = {}

    # 1) Logistic Regression -------------------------------------------------
    print("Training Logistic Regression ...")
    logreg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE
    )
    logreg.fit(X_train_s, y_train)
    models["Logistic Regression"] = logreg
    results["Logistic Regression"] = {
        "strategy": "class_weight", **_evaluate(logreg, X_test_s, y_test)
    }

    # 2) Random Forest -------------------------------------------------------
    print("Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    rf.fit(X_train_s, y_train)
    models["Random Forest"] = rf
    results["Random Forest"] = {
        "strategy": "class_weight", **_evaluate(rf, X_test_s, y_test)
    }

    # 3a) XGBoost with scale_pos_weight -------------------------------------
    print("Training XGBoost (scale_pos_weight) ...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    xgb.fit(X_train_s, y_train)
    models["XGBoost"] = xgb
    results["XGBoost"] = {
        "strategy": "scale_pos_weight", **_evaluate(xgb, X_test_s, y_test)
    }

    # 3b) XGBoost on SMOTE-resampled training data ---------------------------
    print("Training XGBoost (SMOTE) ...")
    X_res, y_res = apply_smote(X_train_s, y_train)
    xgb_smote = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    xgb_smote.fit(X_res, y_res)
    models["XGBoost (SMOTE)"] = xgb_smote
    results["XGBoost (SMOTE)"] = {
        "strategy": "SMOTE", **_evaluate(xgb_smote, X_test_s, y_test)
    }

    # 4) Isolation Forest (unsupervised) ------------------------------------
    print("Training Isolation Forest ...")
    contamination = float(np.clip(y_train.mean(), 1e-6, 0.5))
    iforest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    iforest.fit(X_train_s)  # unsupervised: no labels
    iforest_proba = fit_isolation_forest_proba(iforest, X_train_s)
    models["Isolation Forest"] = iforest_proba
    results["Isolation Forest"] = {
        "strategy": "unsupervised", **_evaluate(iforest_proba, X_test_s, y_test)
    }

    return models, results


def select_best(results: dict) -> str:
    """Pick the best model by PR-AUC, Recall as tie-breaker."""
    return max(
        results,
        key=lambda name: (results[name]["pr_auc"], results[name]["recall"]),
    )


def save_artifacts(best_name: str, best_model, results: dict) -> None:
    """Persist best_model.pkl, model_metadata.json, comparison_results.json."""
    config.ensure_dirs()
    joblib.dump(best_model, config.BEST_MODEL_PATH)

    metadata = {
        "model": best_name,
        "threshold": config.DEFAULT_THRESHOLD,
        "approve_below": config.APPROVE_BELOW,
        "block_above": config.BLOCK_ABOVE,
        "metrics": {k: v for k, v in results[best_name].items() if k != "strategy"},
        "strategy": results[best_name].get("strategy"),
        "feature_columns": config.FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    config.COMPARISON_PATH.write_text(json.dumps(results, indent=2))


def print_comparison(results: dict, best_name: str) -> None:
    header = f"{'Model':<22}{'Strategy':<16}{'Precision':>10}{'Recall':>9}{'F1':>8}{'PR-AUC':>9}"
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON (ranked by PR-AUC)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    ranked = sorted(results, key=lambda n: results[n]["pr_auc"], reverse=True)
    for name in ranked:
        r = results[name]
        marker = "  <-- BEST" if name == best_name else ""
        print(
            f"{name:<22}{r.get('strategy',''):<16}{r['precision']:>10}"
            f"{r['recall']:>9}{r['f1']:>8}{r['pr_auc']:>9}{marker}"
        )
    print("=" * len(header))


def run(save_processed_arrays: bool = True) -> str:
    """Full training pipeline. Returns the winning model name."""
    print("Loading data and splitting (stratified) ...")
    X_train, X_test, y_train, y_test = get_train_test()

    print("Preprocessing (fit scaler on train only) ...")
    X_train_s, X_test_s, _ = preprocess_train_test(X_train, X_test, save=True)

    if save_processed_arrays:
        save_processed(X_train_s.to_numpy(), X_test_s.to_numpy(),
                       np.asarray(y_train), np.asarray(y_test))

    models, results = train_all_models(X_train_s, X_test_s, y_train, y_test)
    best_name = select_best(results)
    save_artifacts(best_name, models[best_name], results)
    print_comparison(results, best_name)
    print(f"\nBest model: {best_name}")
    print(f"Artifacts saved to: {config.MODELS_DIR}")
    return best_name


if __name__ == "__main__":
    run()
