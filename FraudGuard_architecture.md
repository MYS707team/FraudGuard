# 🏛️ FraudGuard — Full System Architecture

**FraudGuard** is a real-time fraud detection system for financial transactions. It
uses machine learning to score each transaction, returns a fraud probability
(0–100%), and makes an automated decision: **Approve**, **Review**, or **Block**.

This document describes the complete architecture, layer by layer, down to the
implementation details — from raw data all the way to testing the model with new
data through the UI.

---

## 0. High-Level Overview (Data Flow)

The system is split into two parts:

- **Offline part** — runs once; trains and compares models, then saves artifacts.
- **Online part** — runs in real time; the user tests transactions through the UI.

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRAUDGUARD SYSTEM                          │
└─────────────────────────────────────────────────────────────────┘

   OFFLINE PART (one-time, training)           ONLINE PART (real-time)
   ════════════════════════════════           ═══════════════════════

   ┌──────────────┐                            ┌──────────────────┐
   │ 1. RAW DATA  │                            │  7. GRADIO UI    │
   │ (Kaggle CSV) │                            │  (end user)      │
   └──────┬───────┘                            └────────┬─────────┘
          │                                             │ transaction
          ▼                                             │ data
   ┌──────────────┐                                     ▼
   │ 2. EDA &     │                            ┌──────────────────┐
   │ PREPROCESS   │                            │  6. FASTAPI      │
   └──────┬───────┘                            │  BACKEND (REST)  │
          │                                     └────────┬─────────┘
          ▼                                             │
   ┌──────────────┐                                     ▼
   │ 3. FEATURE   │                            ┌──────────────────┐
   │ ENGINEERING  │                            │ 5. PREDICTION    │
   └──────┬───────┘                            │ SERVICE          │
          │                                     │ (scaler + model  │
          ▼                                     │  + decision)     │
   ┌──────────────┐         ┌──────────────┐   └────────┬─────────┘
   │ 4. TRAIN &   │────────▶│ MODEL        │            │
   │ COMPARE      │  save   │ ARTIFACTS    │◀───────────┘
   │ (4 models)   │         │ (.pkl/.json) │   loads
   └──────────────┘         └──────────────┘
```

---

## 1. 📁 Project Structure

```
fraudguard/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv          # downloaded from Kaggle
│   └── processed/
│       ├── X_train.npy             # prepared data
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # analysis + comparison (for portfolio)
│
├── models/                          # ⭐ Saved artifacts
│   ├── scaler.pkl                  # StandardScaler (Amount, Time)
│   ├── best_model.pkl              # winning model
│   ├── model_metadata.json         # which model, threshold, metrics
│   └── comparison_results.json     # 4-model results (shown in UI)
│
├── src/
│   ├── config.py                   # all settings (paths, thresholds)
│   ├── data_loader.py              # load CSV, train/test split
│   ├── preprocessing.py            # scaling, SMOTE
│   ├── train.py                    # train 4 models + compare
│   ├── evaluate.py                 # metrics, plots
│   └── predict.py                  # PredictionService (core logic)
│
├── api/
│   └── main.py                     # FastAPI application
│
├── ui/
│   └── app.py                      # Gradio interface
│
├── tests/
│   └── test_predict.py
│
├── requirements.txt
├── README.md                        # important for portfolio!
└── .gitignore
```

---

## 2. 🗄️ Layer 1 — Data Layer

**Source:** Kaggle Credit Card Fraud Detection
**Size:** 284,807 transactions, 31 columns

| Column | Description |
|--------|-------------|
| `Time` | Seconds elapsed since the first transaction |
| `V1`–`V28` | Features anonymized via PCA (for privacy) |
| `Amount` | Transaction amount |
| `Class` | **Target:** 0 = normal, 1 = fraud |

**Key fact:** Only **492 fraud cases** (~0.172%). This is the project's biggest
challenge and exactly where the "imbalanced data" skill shines.

**`data_loader.py` responsibilities:**
- Read the CSV
- Separate the `Class` column (X, y)
- **Stratified split** (80% train / 20% test) so the fraud ratio is preserved in
  both sets (very important).

---

## 3. 🔧 Layer 2 — Preprocessing

`V1–V28` are already normalized via PCA. Only two columns need processing:

```
Time   → StandardScaler  (or 0–1 normalization)
Amount → StandardScaler  (most important: amounts span a wide range)
```

**⚠️ Most important rule:** The scaler is fit **only on the training data**
(`.fit()`), then `.transform()` is applied to test and real-time data. Otherwise
"data leakage" occurs — a professional mistake. Doing this correctly is a plus for
the portfolio.

`scaler.pkl` is saved → the same scaler is then used in the API.

---

## 4. ⚖️ Layer 3 — Handling Imbalance

We **compare two strategies** (for each model):

```
STRATEGY A: Class Weights
  → tell the model "a fraud error is ~100x more costly"
  → XGBoost: scale_pos_weight,  RF/LogReg: class_weight='balanced'
  → does not change the data (faster, cleaner)

STRATEGY B: SMOTE (Synthetic Minority Over-sampling)
  → generates synthetic samples for the minority class (fraud)
  → ⚠️ applied ONLY to the training set, never to the test set!
```

We then build a table: which model + which strategy gives the best result → pick
that one.

---

## 5. 🤖 Layer 4 — Model Training & Comparison (`train.py`)

**Four models:**

| Model | Role | Imbalance method |
|-------|------|------------------|
| Logistic Regression | Baseline | class_weight |
| Random Forest | Strong classic | class_weight |
| XGBoost | Main candidate ⭐ | scale_pos_weight + SMOTE comparison |
| Isolation Forest | Anomaly approach | unsupervised (no labels) |

**Evaluation metrics (Accuracy is NOT used!):**

On imbalanced data, accuracy is misleading — predicting "normal" every time still
yields 99.8% accuracy. Therefore:

- **Precision** — "of those flagged as fraud, how many were correct" (reduces false
  blocks)
- **Recall** — "of all real frauds, how many did we catch" (most important!)
- **F1-score** — balance of the two
- **PR-AUC** (Precision-Recall AUC) — more reliable than ROC-AUC on imbalanced data
- **Confusion Matrix** — visual

**Saved artifacts:**
- `best_model.pkl` — the winning model
- `model_metadata.json` — `{"model": "XGBoost", "threshold": 0.5, "pr_auc": 0.87, ...}`
- `comparison_results.json` — results of all 4 models (rendered as a table in the UI)

---

## 6. 🎯 Layer 5 — Prediction Service (`predict.py`) — Heart of the System

This is the most important module. Both the UI and the API call it.

```python
class FraudPredictor:
    def __init__(self):
        # loaded once (stays in memory)
        self.scaler = load("models/scaler.pkl")
        self.model  = load("models/best_model.pkl")
        self.meta   = load("models/model_metadata.json")

    def predict(self, transaction: dict) -> dict:
        # 1. dict -> correctly ordered array
        # 2. transform Amount, Time with the scaler
        # 3. model.predict_proba() -> probability (0..1)
        # 4. decision logic (below)
        # 5. return result
```

**🚦 Decision Logic (the block/review part):**

```
fraud probability (p):

   p < 0.30          ->  ✅ APPROVE        [green]
   0.30 ≤ p < 0.70   ->  ⚠️ REVIEW         [yellow] — manual review
   p ≥ 0.70          ->  🛑 BLOCK          [red]
```

> These thresholds can be tuned later based on metrics. Adding business logic is a
> big plus for recruiters.

**Returned response (JSON):**
```json
{
  "is_fraud": true,
  "fraud_probability": 0.87,
  "risk_level": "HIGH",
  "decision": "BLOCK",
  "model_used": "XGBoost"
}
```

---

## 7. 🔌 Layer 6 — Backend (FastAPI, `api/main.py`)

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | is the system up (model loaded)? |
| `POST` | `/predict` | one transaction -> prediction |
| `GET` | `/model-info` | which model, its metrics |

**`POST /predict` flow:**
```
1. Pydantic validates the request (V1..V28, Amount, Time present and numeric)
2. FraudPredictor.predict() is called
3. JSON response is returned
```

**Pydantic validation** — on invalid data, a clear error message (e.g. "Amount must
be a number"). This makes the API professional and auto-generates Swagger docs at
`/docs`.

---

## 8. 🖥️ Layer 7 — Gradio UI (`ui/app.py`)

The UI is organized into **3 tabs**:

### 📑 Tab 1 — "Check a Transaction" (main)
```
┌────────────────────────────────────────────────────────┐
│  🛡️ FraudGuard — Fraud Detection                         │
├────────────────────────────────────────────────────────┤
│                                                          │
│  INPUT:                                                  │
│   Amount:  [____1250.00____]                             │
│   Time:    [____43200_______]                            │
│   V1:  [___]  V2: [___]  V3: [___] ...  V28: [___]       │
│                                                          │
│   [ 🎲 Random normal sample ]  [ 🎲 Fraud sample ]       │
│   [           🔍 CHECK            ]                       │
│                                                          │
├────────────────────────────────────────────────────────┤
│  RESULT:                                                 │
│                                                          │
│      🛑  BLOCKED                                         │
│      Fraud probability: ████████░░  87%                  │
│      Risk level: HIGH                                    │
│      Model: XGBoost                                      │
│                                                          │
└────────────────────────────────────────────────────────┘
```

**Important UX decision:** Nobody types 30 `V1–V28` values by hand. So we add
**"Load random sample"** buttons that fill a ready normal or fraud transaction from
the test set with one click. This makes the UI easy and impressive to demo.

### 📊 Tab 2 — "Model Results" (for portfolio)
- 4-model comparison table (Precision / Recall / F1 / PR-AUC)
- Confusion matrix image
- Explanation of why the winning model won

### 📈 Tab 3 — "Batch Check" (optional, extra)
- Upload a CSV → check multiple transactions at once
- Results shown as a table

**Gradio ↔ API connection:** the Gradio frontend calls the FastAPI backend over
HTTP (`requests`), giving a true microservice split between frontend and backend.

---

## 9. 🔄 End-to-End Real-Time Flow (UI data → result)

```
1. User clicks "🎲 Fraud sample" in the Gradio UI
        │
        ▼
2. UI fills the fields with one fraudulent transaction from the test set
   {Time: 406, V1: -2.3, ..., V28: 0.2, Amount: 0.0}
        │
        ▼
3. User clicks "🔍 CHECK"
        │
        ▼
4. Gradio → HTTP POST → FastAPI /predict   (JSON sent)
        │
        ▼
5. FastAPI validates with Pydantic → FraudPredictor.predict()
        │
        ▼
6. FraudPredictor:
     a) Amount & Time → scaler.transform()
     b) model.predict_proba() → p = 0.87
     c) decision logic → "BLOCK"
        │
        ▼
7. JSON response → Gradio
        │
        ▼
8. UI renders nicely: 🛑 BLOCKED, 87%, HIGH risk
```

---

## 10. 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| ML | scikit-learn, xgboost, imbalanced-learn (SMOTE) |
| Storage | joblib (pkl), json |
| Backend | FastAPI, uvicorn, pydantic |
| UI | Gradio |
| Visualization | matplotlib, seaborn |
| Testing | pytest |

---

## 11. ✅ Agreed Scope Summary

- Binary classification + probability (0–100%) + decision (Block / Review / Approve)
- FastAPI REST API (separate backend)
- Kaggle Credit Card Fraud dataset
- Gradio UI for interactive testing
- 4-model comparison: Logistic Regression, Random Forest, XGBoost, Isolation Forest
- Imbalance handled by comparing class weights vs. SMOTE
