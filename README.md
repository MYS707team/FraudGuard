# 🛡️ FraudGuard

Real-time credit-card fraud detection system. It scores each transaction with a
machine-learning model, returns a fraud probability (0–100%), and makes an
automated decision: **Approve**, **Review**, or **Block**.

The UI is **bilingual (English / O‘zbek)** and the whole project can be run end-to-end
in **Google Colab**.

---

## ✨ Features

- **Four models compared**: Logistic Regression, Random Forest, XGBoost (with
  `scale_pos_weight` and a SMOTE variant), and Isolation Forest (unsupervised).
- **Imbalance-aware**: ranked by **PR-AUC** (not accuracy) — on a dataset that is
  ~99.8% non-fraud, accuracy is misleading.
- **Microservice design**: a FastAPI backend serves predictions; the Gradio UI
  talks to it over HTTP.
- **Bilingual UI**: switch between English and Uzbek at runtime.
- **Reproducible**: fixed random seeds, scaler fit on training data only (no leakage).

---

## 🗂️ Project structure

```
FraudGuard/
├── api/main.py            # FastAPI backend (/health, /predict, /model-info)
├── ui/app.py              # Gradio UI (bilingual EN/UZ, 3 tabs)
├── src/
│   ├── config.py          # paths, thresholds, feature schema
│   ├── data_loader.py     # load CSV, stratified train/test split
│   ├── preprocessing.py   # scaler (train-only fit), SMOTE, class weights
│   ├── anomaly.py         # Isolation Forest -> predict_proba adapter
│   ├── train.py           # train + compare 4 models, save best
│   ├── evaluate.py        # confusion matrix, PR curve, comparison table
│   └── predict.py         # FraudPredictor: transaction -> decision
├── tests/test_predict.py  # unit tests (no dataset needed)
├── notebooks/FraudGuard_Colab.ipynb   # one-click Colab runner
├── data/{raw,processed}/  # dataset & processed arrays (git-ignored)
├── models/                # trained artifacts (git-ignored)
└── requirements.txt
```

---

## 🚀 Run in Google Colab (easiest)

1. Open `notebooks/FraudGuard_Colab.ipynb` in
   [Google Colab](https://colab.research.google.com/).
2. Run the cells top-to-bottom. They will:
   clone the repo → install deps → download the dataset → train → launch the API
   and the bilingual UI with a public `*.gradio.live` link.

The dataset is the Kaggle
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset. The notebook supports both the Kaggle API (`kaggle.json`) and manual
CSV upload.

---

## 💻 Run locally

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add the dataset
#    Download creditcard.csv from Kaggle and place it in data/raw/
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 3. Train (creates models/best_model.pkl, scaler.pkl, metadata, plots)
python -m src.train

# 4. Start the API (Swagger docs at http://localhost:8000/docs)
uvicorn api.main:app --port 8000

# 5. In another terminal, start the UI (http://localhost:7860)
python ui/app.py
```

The UI reads the backend URL from the `FRAUDGUARD_API_URL` environment variable
(default `http://localhost:8000`). To expose a public link locally, set
`FRAUDGUARD_SHARE=true`.

---

## 🔌 API

| Method | Endpoint      | Description                                  |
|--------|---------------|----------------------------------------------|
| GET    | `/health`     | Service status + whether the model is loaded |
| POST   | `/predict`    | Score a transaction → fraud decision         |
| GET    | `/model-info` | Winning model name + metrics                 |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time":0,"V1":0,"V2":0,"V3":0,"V4":0,"V5":0,"V6":0,"V7":0,"V8":0,"V9":0,
       "V10":0,"V11":0,"V12":0,"V13":0,"V14":0,"V15":0,"V16":0,"V17":0,"V18":0,
       "V19":0,"V20":0,"V21":0,"V22":0,"V23":0,"V24":0,"V25":0,"V26":0,"V27":0,
       "V28":0,"Amount":149.62}'
```

**Example response:**

```json
{
  "is_fraud": false,
  "fraud_probability": 0.0123,
  "risk_level": "LOW",
  "decision": "APPROVE",
  "model_used": "XGBoost"
}
```

### Decision thresholds

| Fraud probability `p` | Decision  | Risk   |
|-----------------------|-----------|--------|
| `p < 0.30`            | APPROVE   | LOW    |
| `0.30 ≤ p < 0.70`     | REVIEW    | MEDIUM |
| `p ≥ 0.70`            | BLOCK     | HIGH   |

(Configurable in `src/config.py`.)

---

## 🧪 Tests

```bash
python -m pytest tests/ -q
```

The unit tests use stubbed model/scaler objects, so they run **without** the
dataset or trained artifacts.

---

## 📚 More

- System design: [`FraudGuard_architecture.md`](FraudGuard_architecture.md)
- Step-by-step build log: [`PROGRESS.md`](PROGRESS.md)
