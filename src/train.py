"""
FraudGuard — Model training & comparison.

Trains and compares five candidates on the (untouched) test set:

  1. Logistic Regression  (class_weight='balanced')
  2. Random Forest         (class_weight='balanced')
  3. XGBoost               (scale_pos_weight)  — early stopping
  4. XGBoost (SMOTE)       (oversampled train) — early stopping
  5. Isolation Forest      (unsupervised anomaly detection)

Why not accuracy? On a dataset that is 99.8% non-fraud, always predicting
"normal" scores 99.8% accuracy yet catches zero fraud.

Which model wins? We select by **F1** (config.SELECTION_METRIC) — the harmonic
mean of precision & recall — so the winner is *balanced*: it must both catch
fraud (recall) and avoid flooding analysts with false alarms (precision).
Ranking by PR-AUC alone can crown a model that ranks well but is unusable at the
operating threshold (e.g. 13% precision). PR-AUC and recall are tie-breakers.

Optimal training length: the XGBoost models train up to ``XGB_MAX_ROUNDS`` rounds
but **stop early** at the round with the best validation PR-AUC, so the saved
model is the optimal size — never over- or under-trained.

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
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src import config
from src.anomaly import fit_isolation_forest_proba
from src.data_loader import get_train_test, save_processed
from src.preprocessing import (
    apply_smote,
    get_scale_pos_weight,
    preprocess_train_test,
)

# Total number of candidate models (used for the "[i/N]" progress counter).
TOTAL_MODELS = 5


# --------------------------------------------------------------------------- #
# Progress logging helpers
# --------------------------------------------------------------------------- #
def _start(step: int, name: str, strategy: str, detail: str = "") -> float:
    """Print a clear 'now training X' banner and return a start timestamp."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"[{step}/{TOTAL_MODELS}]  TRAINING:  {name}   (strategy: {strategy})")
    if detail:
        print(f"          {detail}")
    print(bar)
    return time.perf_counter()


def _done(step: int, name: str, t0: float, metrics: dict) -> None:
    """Print how long the model took and its headline metrics."""
    dt = time.perf_counter() - t0
    print(
        f"[{step}/{TOTAL_MODELS}]  DONE:      {name}  in {dt:5.1f}s  ->  "
        f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
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


def _make_xgb(scale_pos_weight: float | None) -> XGBClassifier:
    """Build an XGBoost classifier configured for early stopping."""
    kwargs = dict(
        n_estimators=config.XGB_MAX_ROUNDS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        eval_metric="aucpr",
        early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**kwargs)


def train_all_models(X_train_s, X_test_s, y_train, y_test) -> tuple[dict, dict]:
    """Train every model, return (fitted_models, comparison_metrics)."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Inner validation split (carved from TRAIN only) for XGBoost early stopping.
    # The test set stays a truly unseen hold-out used only for final scoring.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_s,
        y_train,
        test_size=config.VALIDATION_SIZE,
        stratify=y_train,
        random_state=config.RANDOM_STATE,
    )

    models: dict = {}
    results: dict = {}

    # 1) Logistic Regression -------------------------------------------------
    t0 = _start(1, "Logistic Regression", "class_weight",
                "fitting on full training set ...")
    logreg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE
    )
    logreg.fit(X_train_s, y_train)
    models["Logistic Regression"] = logreg
    results["Logistic Regression"] = {
        "strategy": "class_weight", **_evaluate(logreg, X_test_s, y_test)
    }
    _done(1, "Logistic Regression", t0, results["Logistic Regression"])

    # 2) Random Forest -------------------------------------------------------
    t0 = _start(2, "Random Forest", "class_weight",
                "building 200 trees (verbose progress below) ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        verbose=1,
    )
    rf.fit(X_train_s, y_train)
    models["Random Forest"] = rf
    results["Random Forest"] = {
        "strategy": "class_weight", **_evaluate(rf, X_test_s, y_test)
    }
    _done(2, "Random Forest", t0, results["Random Forest"])

    # 3) XGBoost with scale_pos_weight (early stopping) ----------------------
    spw = get_scale_pos_weight(y_tr)
    t0 = _start(
        3, "XGBoost", "scale_pos_weight",
        f"up to {config.XGB_MAX_ROUNDS} rounds, early-stop after "
        f"{config.XGB_EARLY_STOPPING_ROUNDS} stale rounds "
        f"(val score every {config.XGB_VERBOSE_EVERY}):",
    )
    xgb = _make_xgb(scale_pos_weight=spw)
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=config.XGB_VERBOSE_EVERY)
    print(f"          -> stopped at best round {xgb.best_iteration + 1} "
          f"(best val PR-AUC = {xgb.best_score:.4f})")
    models["XGBoost"] = xgb
    results["XGBoost"] = {
        "strategy": "scale_pos_weight",
        "best_round": int(xgb.best_iteration + 1),
        **_evaluate(xgb, X_test_s, y_test),
    }
    _done(3, "XGBoost", t0, results["XGBoost"])

    # 4) XGBoost on SMOTE-resampled training data (early stopping) -----------
    t0 = _start(
        4, "XGBoost (SMOTE)", "SMOTE",
        "oversampling train with SMOTE, then boosting with early stopping ...",
    )
    X_res, y_res = apply_smote(X_tr, y_tr)
    # SMOTE returns a NumPy array; restore the feature names so XGBoost sees the
    # same named columns in both the (resampled) train set and the eval_set.
    X_res = pd.DataFrame(np.asarray(X_res), columns=X_tr.columns)
    print(f"          SMOTE: {len(y_tr):,} -> {len(y_res):,} train rows "
          f"(balanced {int((y_res == 1).sum()):,} fraud / "
          f"{int((y_res == 0).sum()):,} normal)")
    xgb_smote = _make_xgb(scale_pos_weight=None)
    xgb_smote.fit(
        X_res, y_res, eval_set=[(X_val, y_val)], verbose=config.XGB_VERBOSE_EVERY
    )
    print(f"          -> stopped at best round {xgb_smote.best_iteration + 1} "
          f"(best val PR-AUC = {xgb_smote.best_score:.4f})")
    models["XGBoost (SMOTE)"] = xgb_smote
    results["XGBoost (SMOTE)"] = {
        "strategy": "SMOTE",
        "best_round": int(xgb_smote.best_iteration + 1),
        **_evaluate(xgb_smote, X_test_s, y_test),
    }
    _done(4, "XGBoost (SMOTE)", t0, results["XGBoost (SMOTE)"])

    # 5) Isolation Forest (unsupervised) ------------------------------------
    contamination = float(np.clip(y_train.mean(), 1e-6, 0.5))
    t0 = _start(5, "Isolation Forest", "unsupervised",
                f"200 trees, contamination={contamination:.5f} ...")
    iforest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        verbose=1,
    )
    iforest.fit(X_train_s)  # unsupervised: no labels
    iforest_proba = fit_isolation_forest_proba(iforest, X_train_s)
    models["Isolation Forest"] = iforest_proba
    results["Isolation Forest"] = {
        "strategy": "unsupervised", **_evaluate(iforest_proba, X_test_s, y_test)
    }
    _done(5, "Isolation Forest", t0, results["Isolation Forest"])

    return models, results


def _score_tuple(result: dict) -> tuple:
    """Sort key: primary selection metric, then the configured tie-breakers."""
    keys = [config.SELECTION_METRIC, *config.SELECTION_TIEBREAKERS]
    return tuple(result.get(k, 0.0) for k in keys)


def select_best(results: dict) -> str:
    """Pick the most *balanced* model by F1 (with PR-AUC/recall tie-breakers)."""
    return max(results, key=lambda name: _score_tuple(results[name]))


def save_artifacts(best_name: str, best_model, results: dict) -> None:
    """Persist best_model.pkl, model_metadata.json, comparison_results.json."""
    config.ensure_dirs()
    joblib.dump(best_model, config.BEST_MODEL_PATH)

    metadata = {
        "model": best_name,
        "selection_metric": config.SELECTION_METRIC,
        "threshold": config.DEFAULT_THRESHOLD,
        "approve_below": config.APPROVE_BELOW,
        "block_above": config.BLOCK_ABOVE,
        "metrics": {
            k: v for k, v in results[best_name].items()
            if k not in ("strategy", "best_round")
        },
        "strategy": results[best_name].get("strategy"),
        "best_round": results[best_name].get("best_round"),
        "feature_columns": config.FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    config.COMPARISON_PATH.write_text(json.dumps(results, indent=2))


def print_comparison(results: dict, best_name: str) -> None:
    metric = config.SELECTION_METRIC.upper()
    header = (
        f"{'Model':<22}{'Strategy':<16}{'Precision':>10}"
        f"{'Recall':>9}{'F1':>8}{'PR-AUC':>9}"
    )
    print("\n" + "=" * len(header))
    print(f"MODEL COMPARISON (winner chosen by {metric}, balanced precision+recall)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    ranked = sorted(results, key=lambda n: _score_tuple(results[n]), reverse=True)
    for name in ranked:
        r = results[name]
        marker = "  <-- BEST" if name == best_name else ""
        print(
            f"{name:<22}{r.get('strategy',''):<16}{r['precision']:>10.4f}"
            f"{r['recall']:>9.4f}{r['f1']:>8.4f}{r['pr_auc']:>9.4f}{marker}"
        )
    print("=" * len(header))


def run(save_processed_arrays: bool = True) -> str:
    """Full training pipeline. Returns the winning model name."""
    print("Loading data and splitting (stratified) ...")
    X_train, X_test, y_train, y_test = get_train_test()
    print(f"  train rows: {len(X_train):,}   test rows: {len(X_test):,}")

    print("Preprocessing (fit scaler on train only) ...")
    X_train_s, X_test_s, _ = preprocess_train_test(X_train, X_test, save=True)

    if save_processed_arrays:
        save_processed(X_train_s.to_numpy(), X_test_s.to_numpy(),
                       np.asarray(y_train), np.asarray(y_test))

    models, results = train_all_models(X_train_s, X_test_s, y_train, y_test)
    best_name = select_best(results)
    save_artifacts(best_name, models[best_name], results)
    print_comparison(results, best_name)
    print(f"\nBest model: {best_name}  (selected by {config.SELECTION_METRIC.upper()})")
    print(f"Artifacts saved to: {config.MODELS_DIR}")
    return best_name


if __name__ == "__main__":
    run()
