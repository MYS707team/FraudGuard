# 🛡️ FraudGuard

Real-time credit-card fraud detection system. It scores each transaction with a
machine-learning model, returns a fraud probability (0–100%), and makes an
automated decision: **Approve**, **Review**, or **Block**.

The UI is **bilingual (English / O‘zbek)**, **self-contained** (no separate API
server), and the whole project can be run end-to-end in **Google Colab**.

---

## ✨ Features

- **Five candidates compared**: Logistic Regression, Random Forest, XGBoost (with
  `scale_pos_weight` and a SMOTE variant), and Isolation Forest (unsupervised).
- **Optimal model selection**: the winner is chosen by **F1** — a *balanced*
  metric that rewards catching fraud (recall) without flooding analysts with
  false positives (precision). PR-AUC / recall are tie-breakers. (Accuracy is
  intentionally not used — on data that is ~99.8% non-fraud it is misleading.)
- **Early stopping**: the XGBoost models train up to 1000 boosting rounds but stop
  at the round with the best validation PR-AUC, so the saved model is the optimal
  size — never over- or under-trained.
- **Clear training progress**: each model prints a `[i/5]` banner showing which
  model is training, live validation scores for XGBoost, and per-model timing.
- **Self-contained UI**: the Gradio app loads the model in-process — no backend
  server to run.
- **Bilingual UI**: switch between English and Uzbek at runtime.
- **Reproducible**: fixed random seeds, scaler fit on training data only (no leakage).

---

## 🗂️ Project structure

```
FraudGuard/
├── ui/app.py              # Gradio UI (bilingual EN/UZ, 3 tabs, in-process model)
├── src/
│   ├── config.py          # paths, thresholds, selection metric, XGB settings
│   ├── data_loader.py     # load CSV, stratified train/test split
│   ├── preprocessing.py   # scaler (train-only fit), SMOTE, class weights
│   ├── anomaly.py         # Isolation Forest -> predict_proba adapter
│   ├── train.py           # train + compare models, early stopping, save best
│   ├── evaluate.py        # confusion matrix, PR curve, comparison table
│   └── predict.py         # FraudPredictor: transaction -> decision
├── scripts/
│   ├── generate_dataset.py  # synthetic, realistic dataset generator
│   └── train_and_push.py    # train, save weights, commit & push to GitHub
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
   clone the repo → install deps → get the dataset → train → launch the
   bilingual UI with a public `*.gradio.live` link.

---

## 💻 Run locally

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get the dataset
#    A realistic synthetic dataset is generated automatically by the training
#    script if data/raw/creditcard.csv is missing. To use the real Kaggle data,
#    download creditcard.csv into data/raw/ instead:
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 3. Train (creates models/best_model.pkl, scaler.pkl, metadata, plots)
python -m src.train
#    ...or train, save weights, and push them to GitHub in one go:
python scripts/train_and_push.py

# 4. Start the self-contained UI (public *.gradio.live link printed in the console)
python ui/app.py
```

The UI creates a public shareable link by default. Set `FRAUDGUARD_SHARE=false`
to keep it local-only, and `FRAUDGUARD_UI_PORT` to change the port (default 7860).

---

## 🤖 Models & decision logic

The training pipeline scores every model on the untouched test set and reports
**Precision / Recall / F1 / PR-AUC**, then selects the best by **F1**.

### Decision thresholds

| Fraud probability `p` | Decision  | Risk   |
|-----------------------|-----------|--------|
| `p < 0.30`            | APPROVE   | LOW    |
| `0.30 ≤ p < 0.70`     | REVIEW    | MEDIUM |
| `p ≥ 0.70`            | BLOCK     | HIGH   |

(Configurable in `src/config.py`.)

The prediction payload returned by `FraudPredictor.predict()`:

```json
{
  "is_fraud": false,
  "fraud_probability": 0.0123,
  "risk_level": "LOW",
  "decision": "APPROVE",
  "model_used": "Random Forest"
}
```

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
