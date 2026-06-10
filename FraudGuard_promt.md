# 🚀 FraudGuard — Step-by-Step Build Prompts

This file contains a sequence of prompts you can give to an AI coding assistant
(like Kiro) to build the **FraudGuard** project from scratch, one step at a time.

Each step builds on the previous one. Run them in order. After each step, review the
output, test it, and only then move on to the next prompt.

> Reference: see `FraudGuard_architecture.md` for the full system design that these
> prompts implement.

---

## 📋 Prerequisites

- Python 3.10+
- The Kaggle "Credit Card Fraud Detection" dataset (`creditcard.csv`)
- A GitHub repository (already cloned)

---

## STEP 0 — Project Scaffolding

```
Create the FraudGuard project skeleton with this folder structure:
data/raw, data/processed, notebooks, models, src, api, ui, tests.
Add a requirements.txt with: pandas, numpy, scikit-learn, xgboost,
imbalanced-learn, fastapi, uvicorn, pydantic, gradio, matplotlib, seaborn,
joblib, requests, pytest.
Add a .gitignore for Python that ignores data/raw/*.csv, models/*.pkl,
__pycache__, .venv, and .ipynb_checkpoints.
Add an empty README.md with the project title and one-line description.
```

---

## STEP 1 — Configuration Module

```
Create src/config.py. It should centralize all settings:
- File paths (RAW_DATA_PATH, PROCESSED_DIR, MODELS_DIR, SCALER_PATH,
  BEST_MODEL_PATH, METADATA_PATH, COMPARISON_PATH).
- Train/test split settings (TEST_SIZE=0.2, RANDOM_STATE=42, STRATIFY=True).
- Decision thresholds: APPROVE_BELOW=0.30, BLOCK_ABOVE=0.70 (between = REVIEW).
- Feature column order (Time, V1..V28, Amount) as a list, and the target column.
Use pathlib and make paths relative to the project root so it works anywhere.
```

---

## STEP 2 — Data Loader

```
Create src/data_loader.py. It should:
- Load creditcard.csv from RAW_DATA_PATH into a pandas DataFrame.
- Validate that expected columns exist; raise a clear error if the file is missing.
- Split features (X) and target (y = 'Class').
- Perform a stratified train/test split using settings from config.
- Provide a function to save/load processed arrays (.npy) to/from data/processed.
Add a small main block that prints dataset shape and the fraud ratio.
```

---

## STEP 3 — Preprocessing

```
Create src/preprocessing.py. It should:
- Fit a StandardScaler ONLY on the training data, applied to 'Time' and 'Amount'.
- Transform train and test sets with the fitted scaler (avoid data leakage).
- Save the fitted scaler to SCALER_PATH with joblib.
- Provide a SMOTE function (from imbalanced-learn) that oversamples ONLY the
  training set, never the test set.
- Provide a helper that returns class weights / scale_pos_weight for the
  class-weight strategy.
Explain in comments why the scaler must only be fit on training data.
```

---

## STEP 4 — Model Training & Comparison

```
Create src/train.py. It should train and compare four models:
1. Logistic Regression (class_weight='balanced')
2. Random Forest (class_weight='balanced')
3. XGBoost (scale_pos_weight) — also compare an XGBoost variant trained on
   SMOTE-resampled data
4. Isolation Forest (unsupervised anomaly detection)

For each model:
- Train on the training set.
- Evaluate on the untouched test set using Precision, Recall, F1, and PR-AUC
  (do NOT use accuracy as the main metric).
Pick the best model by PR-AUC (with Recall as a tie-breaker).
Save: best_model.pkl, model_metadata.json (winning model name, threshold,
metrics), and comparison_results.json (metrics for all models).
Print a comparison table to the console.
```

---

## STEP 5 — Evaluation & Plots

```
Create src/evaluate.py. It should:
- Build a confusion matrix and a precision-recall curve for a given model.
- Save plots as PNG files into the models/ directory.
- Provide a function that returns a formatted comparison table (pandas DataFrame)
  from comparison_results.json for display in the UI.
Use matplotlib/seaborn. Keep plotting functions reusable.
```

---

## STEP 6 — Prediction Service (core logic)

```
Create src/predict.py with a FraudPredictor class:
- On init, load the scaler, best model, and metadata once (cache in memory).
- A predict(transaction: dict) method that:
  1. Orders features as in config (Time, V1..V28, Amount).
  2. Scales Time and Amount with the loaded scaler.
  3. Computes fraud probability via predict_proba.
  4. Applies decision logic: <0.30 APPROVE, 0.30–0.70 REVIEW, >=0.70 BLOCK.
  5. Returns a dict: is_fraud, fraud_probability, risk_level, decision, model_used.
- Add input validation with clear error messages for missing/invalid fields.
Include a small example usage in a main block.
```

---

## STEP 7 — FastAPI Backend

```
Create api/main.py using FastAPI:
- A Pydantic request model with fields Time, V1..V28, Amount (all floats).
- GET /health -> returns status and whether the model is loaded.
- POST /predict -> validates input, calls FraudPredictor.predict(), returns JSON.
- GET /model-info -> returns the winning model name and its metrics from metadata.
Load FraudPredictor once at startup. Add clear error handling that returns
helpful messages. Make sure Swagger docs work at /docs.
```

---

## STEP 8 — Gradio UI

```
Create ui/app.py using Gradio with three tabs:
Tab 1 "Check a Transaction":
  - Inputs for Amount, Time, and V1..V28.
  - Two buttons: "Load random normal sample" and "Load random fraud sample"
    that fill the fields from the test set.
  - A "Check" button that sends the data to the FastAPI /predict endpoint via
    requests and displays: decision (Approve/Review/Block) with color, fraud
    probability as a percentage, risk level, and model used.
Tab 2 "Model Results":
  - Show the 4-model comparison table and the confusion matrix image.
Tab 3 "Batch Check":
  - Upload a CSV, run predictions for each row, and show a results table.
The UI must call the backend over HTTP (microservice style), not import the model
directly.
```

---

## STEP 9 — Tests

```
Create tests/test_predict.py with pytest:
- Test that FraudPredictor returns the expected keys.
- Test the decision logic at boundary probabilities (0.29, 0.30, 0.69, 0.70).
- Test that invalid input raises a clear error.
Add instructions in the README on how to run the tests.
```

---

## STEP 10 — Documentation & Polish

```
Update README.md to be portfolio-ready:
- Project overview and the business problem it solves.
- Architecture diagram summary (link to FraudGuard_architecture.md).
- The 4-model comparison results table and chosen model.
- How to run: install deps, train models, start the API, launch the UI.
- Screenshots/GIF placeholders of the Gradio UI.
- Key skills demonstrated: anomaly detection, real-time monitoring, imbalanced
  data handling, API design, and deployment.
```

---

## ✅ Suggested Order of Execution

1. STEP 0 → scaffolding
2. STEP 1 → config
3. STEP 2 → data loader (then download the dataset into data/raw/)
4. STEP 3 → preprocessing
5. STEP 4 → train & compare models
6. STEP 5 → evaluation/plots
7. STEP 6 → prediction service
8. STEP 7 → FastAPI backend
9. STEP 8 → Gradio UI
10. STEP 9 → tests
11. STEP 10 → documentation

After STEP 4 you will have trained models; after STEP 7 you can test the API at
`/docs`; after STEP 8 you can demo the full system end-to-end through the UI.
