# 📋 FraudGuard — Build Progress Log

This file logs the work completed after each step/prompt from
[`FraudGuard_promt.md`](FraudGuard_promt.md). Each entry describes what was built,
the key decisions made, and how it fits into the overall architecture.

---

## STEP 0 — Project Scaffolding ✅

**Prompt:** Create the project skeleton, `requirements.txt`, `.gitignore`,
and an empty `README.md`.

**What was done:**
- Created the full folder structure:
  `data/raw`, `data/processed`, `notebooks`, `models`, `src`, `api`, `ui`, `tests`.
- Added `.gitkeep` files so empty folders are tracked by git.
- Created `requirements.txt` with all dependencies: pandas, numpy, scikit-learn,
  xgboost, imbalanced-learn, fastapi, uvicorn, pydantic, gradio, matplotlib,
  seaborn, joblib, requests, pytest.
- Created `.gitignore` for Python that ignores `data/raw/*.csv`, `models/*.pkl`,
  `__pycache__`, virtual environments, and `.ipynb_checkpoints` (while keeping
  `.gitkeep` placeholders).
- Created an initial `README.md` with the project title and one-line description.
- Created this `PROGRESS.md` build log.

**Notes:** Trained `.pkl` model artifacts and the raw Kaggle CSV are git-ignored
because they are large/generated; they are produced by running the training
pipeline (Step 4) or downloaded from Kaggle.


---

## STEP 1 — Configuration (`src/config.py`) ✅

**Prompt:** Central config with paths, split settings, thresholds, feature order.

**What was done:**
- All paths built with `pathlib` relative to `PROJECT_ROOT`, so the code works
  from any working directory (local, Colab, CI).
- Defined raw/processed/model paths, plot paths and JSON artifact paths.
- Train/test split config: `TEST_SIZE=0.2`, `RANDOM_STATE=42`, `STRATIFY=True`.
- Decision thresholds: `APPROVE_BELOW=0.30`, `BLOCK_ABOVE=0.70`,
  `DEFAULT_THRESHOLD=0.50`.
- Feature schema: `FEATURE_COLUMNS = [Time, V1..V28, Amount]`, target `Class`,
  scaled columns `[Time, Amount]`.
- `ensure_dirs()` helper to create directories on demand.

**Verified:** `python -m src.config` prints the resolved configuration.

---

## STEP 2 — Data loading (`src/data_loader.py`) ✅

**Prompt:** Load CSV, validate schema, stratified train/test split, save/load arrays.

**What was done:**
- `load_raw_data()` with clear errors if the file or columns are missing.
- `split_features_target()` and `stratified_split()` (preserves the ~0.17% fraud
  ratio in both subsets — essential for imbalanced data).
- `save_processed()` / `load_processed()` for `.npy` arrays in `data/processed`.
- `fraud_ratio()` utility.

**Verified:** `python -m src.data_loader` printed dataset shape and identical
fraud ratios in train/test.

---

## STEP 3 — Preprocessing (`src/preprocessing.py`) ✅

**Prompt:** Scale Time/Amount (V1..V28 already PCA-scaled). Avoid data leakage.
Provide SMOTE and class-weight strategies.

**What was done:**
- **Anti-leakage rule documented and enforced:** the `StandardScaler` is fit on
  the **training set only**, then used to transform the test set / real-time
  transactions.
- `fit_scaler`, `apply_scaler`, `save_scaler`, `load_scaler`,
  `preprocess_train_test`.
- `apply_smote()` — oversamples the minority class on the **training set only**.
- `get_scale_pos_weight()` (for XGBoost) and `get_class_weights()` (balanced).

**Verified:** `python -m src.preprocessing` fit/saved the scaler, printed
`scale_pos_weight` / class weights, and confirmed SMOTE expanded the training set.

---

## STEP 4 — Training & comparison (`src/train.py` + `src/anomaly.py`) ✅

**Prompt:** Train and compare 4 models; rank by PR-AUC (not accuracy); save the
best model + metadata + comparison.

**What was done:**
- Trained: Logistic Regression (`class_weight`), Random Forest (`class_weight`),
  XGBoost (`scale_pos_weight`) + an XGBoost variant on SMOTE data, and Isolation
  Forest (unsupervised).
- `src/anomaly.py`: `IsolationForestProba` adapter so the unsupervised model
  exposes a `predict_proba` and can be compared/deployed uniformly (a real class
  so joblib can pickle it).
- Models ranked by **PR-AUC** with Recall as tie-breaker; also report
  Precision / Recall / F1.
- Saved `best_model.pkl`, `model_metadata.json`, `comparison_results.json`.
- Documented **why accuracy is the wrong metric** here.

**Verified:** `python -m src.train` trained all five candidates and saved the
winning model + JSON artifacts.

---

## STEP 5 — Evaluation & plots (`src/evaluate.py`) ✅

**Prompt:** Confusion matrix, PR curve, comparison table for the UI.

**What was done:**
- Headless matplotlib backend (`Agg`) for Colab/servers.
- `plot_confusion_matrix()` and `plot_pr_curve()` → PNGs in `models/`.
- `comparison_dataframe()` loads `comparison_results.json` into a tidy,
  PR-AUC-sorted DataFrame for the UI.
- `generate_plots_for_best_model()` regenerates plots from the saved best model.

**Verified:** `python -m src.evaluate` produced `confusion_matrix.png` and
`pr_curve.png` and printed the comparison table.

---

## STEP 6 — Prediction service (`src/predict.py`) ✅

**Prompt:** `FraudPredictor` that loads artifacts once and turns a transaction
dict into a decision.

**What was done:**
- Loads scaler + best model + metadata once (lazy singleton via `get_predictor()`).
- Validates input (missing fields / non-numeric values raise clear `ValueError`s).
- Builds a **named DataFrame** so the model gets the same feature names it was
  trained on (no sklearn warnings).
- Returns `{is_fraud, fraud_probability, risk_level, decision, model_used}`.
- Decision mapping: APPROVE (LOW) / REVIEW (MEDIUM) / BLOCK (HIGH).

**Verified:** `python -m src.predict` returned a clean decision payload.

---

## STEP 7 — FastAPI backend (`api/main.py`) ✅

**Prompt:** REST API with `/health`, `/predict`, `/model-info` and a Pydantic
schema.

**What was done:**
- `Transaction` Pydantic v2 model (Time, V1..V28, Amount) with an example payload
  for Swagger.
- `/health` reports model-loaded status; `/predict` returns the decision;
  `/model-info` returns the winning model + metrics.
- Loads the model once at startup; stays up with a helpful message if artifacts
  are missing (returns 503 on prediction).
- Validation errors → HTTP 422.

**Verified:** all modules import cleanly and `py_compile` passes. (Per request,
the server was not launched interactively here; the Colab notebook runs it.)

---

## STEP 8 — Bilingual Gradio UI (`ui/app.py`) ✅

**Prompt:** Gradio UI that talks to the API over HTTP. Bilingual EN/UZ.

**What was done:**
- Three tabs: **Check a Transaction**, **Model Results**, **Batch Check**.
- **Bilingual (English / O‘zbek)** with a runtime language switch that updates
  titles, labels and buttons; results are rendered in the selected language.
- Color-coded decision card (green/yellow/red) with a probability bar.
- "Random normal / fraud sample" buttons load **raw** rows from the test split so
  the API scales them correctly; gracefully degrades to zeros if no dataset.
- Model-comparison table + confusion-matrix image; CSV batch scoring tab.
- Talks to the backend purely over HTTP (`FRAUDGUARD_API_URL`), keeping the
  microservice separation. `share=True` supported via `FRAUDGUARD_SHARE`.

**Verified:** `py_compile` passes for the UI module.

---

## STEP 9 — Tests (`tests/test_predict.py`) ✅

**Prompt:** Unit tests for the prediction logic.

**What was done:**
- Tests build a `FraudPredictor` with a **stub scaler + stub model**, so they run
  without the dataset or trained artifacts (CI-friendly).
- Cover: response keys, decision boundaries (0.29/0.30/0.69/0.70), the static
  `_decision` mapping, and validation errors (missing field, non-numeric, non-dict).

**Verified:** `python -m pytest tests/ -q` → **9 passed**.

---

## STEP 10 — Google Colab notebook + docs ✅

**Prompt:** Make it runnable in Google Colab; write the README.

**What was done:**
- `notebooks/FraudGuard_Colab.ipynb`: clone → install → get dataset (Kaggle API or
  manual upload) → train → launch API in the background → launch the bilingual UI
  with a public `*.gradio.live` link. Markdown is bilingual.
- Full `README.md`: features, structure, Colab & local instructions, API
  reference, decision thresholds, and how to run tests.
- `.gitignore` updated so generated artifacts (`*.pkl`, `*.png`, `models/*.json`,
  processed `.npy`, raw `.csv`) are not committed — they are reproduced by
  training.

**Verified:** notebook JSON validated; all `src/`, `api/`, `ui/`, `tests/` files
compile; unit tests pass.

---

## ✅ Final status

All steps from `FraudGuard_promt.md` are complete: data → preprocessing → training
& comparison → evaluation → prediction service → FastAPI backend → bilingual
Gradio UI → tests → Colab notebook. The project runs end-to-end locally and in
Google Colab, and was pushed to the `main` branch.


---

## STEP 11 — Realistic dataset, train-&-push, public UI ✅

**What was done:**
- `scripts/generate_dataset.py`: generates a large, realistic synthetic dataset
  (real Kaggle schema, PCA-like features, ~0.17% fraud) so the pipeline trains
  out-of-the-box without the proprietary CSV.
- `scripts/train_and_push.py`: one shell command to train, save weights/plots, and
  commit + push the trained artifacts to GitHub.
- Gradio UI now defaults to a public `*.gradio.live` share link
  (`FRAUDGUARD_SHARE=true`).

---

## STEP 12 — Optimal selection, early stopping, progress, no server ✅

**Prompt:** Fix the evaluation/model selection so the *optimal* model wins, make
training stop at the optimal point, show clear training progress, and remove the
API server.

**What was done:**
- **Balanced model selection:** the winner is now chosen by **F1**
  (`config.SELECTION_METRIC`), with PR-AUC then recall as tie-breakers. This
  replaces ranking by PR-AUC alone, which crowned Logistic Regression despite ~13%
  precision; F1 favors a balanced model (e.g. Random Forest).
- **Early stopping:** the two XGBoost models train up to `XGB_MAX_ROUNDS` (1000) on
  an inner validation split carved from train, stopping at the best validation
  PR-AUC (`XGB_EARLY_STOPPING_ROUNDS` patience). The saved model is the optimal
  size, and `best_round` is recorded in the metrics.
- **Clear progress:** each model prints a `[i/5]` banner (which model + strategy +
  settings), XGBoost streams validation scores every N rounds and reports its best
  round, Random Forest / Isolation Forest log build progress, and every model is
  timed.
- **Removed the server:** deleted `api/` (FastAPI). The Gradio UI now loads the
  model **in-process** via `FraudPredictor` — no HTTP, no backend to run. Dropped
  `fastapi`, `uvicorn`, `pydantic`, and `requests` from `requirements.txt`.
- Updated the Colab notebook (no API cell), README, and architecture doc to match
  the self-contained design.

**Verified:** full training run shows progress + early stopping and selects the
balanced best model; `pytest` passes; the UI imports/compiles with no API.
