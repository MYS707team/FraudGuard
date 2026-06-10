"""
FraudGuard — Gradio UI (bilingual: English / O'zbek).

Three tabs:
  1. Check a Transaction  — single transaction scoring with random-sample loaders.
  2. Model Results        — 4-model comparison table + confusion matrix.
  3. Batch Check          — upload a CSV and score every row.

Microservice style: this UI talks to the FastAPI backend over HTTP (``requests``).
It never imports the model directly. Configure the backend URL with the
``FRAUDGUARD_API_URL`` environment variable (default: http://localhost:8000).
"""
from __future__ import annotations

# --- import bootstrap --------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# -----------------------------------------------------------------------------

import os

import gradio as gr
import pandas as pd
import requests

from src import config

API_URL = os.environ.get("FRAUDGUARD_API_URL", "http://localhost:8000").rstrip("/")

V_COLUMNS = config.V_FEATURES  # V1..V28

# --------------------------------------------------------------------------- #
# Bilingual strings
# --------------------------------------------------------------------------- #
T = {
    "en": {
        "title": "# 🛡️ FraudGuard — Real-time Fraud Detection",
        "subtitle": "Score a transaction and get an automated decision: "
        "**Approve**, **Review**, or **Block**.",
        "lang_label": "Language / Til",
        "tab_check": "🔍 Check a Transaction",
        "tab_results": "📊 Model Results",
        "tab_batch": "📈 Batch Check",
        "inputs": "Transaction inputs",
        "amount": "Amount",
        "time": "Time (seconds since first transaction)",
        "vfeatures": "PCA features V1–V28",
        "btn_normal": "🎲 Load random normal sample",
        "btn_fraud": "🎲 Load random fraud sample",
        "btn_check": "🔍 CHECK",
        "result": "### Result",
        "results_heading": "## 📊 Model Comparison",
        "results_desc": "Four models compared on the untouched test set "
        "(ranked by PR-AUC; accuracy is intentionally not used on imbalanced data).",
        "cm_heading": "### Confusion matrix (best model)",
        "batch_heading": "## 📈 Batch Check",
        "batch_desc": "Upload a CSV with columns Time, V1..V28, Amount. "
        "Each row is scored and shown below.",
        "batch_upload": "Upload CSV",
        "btn_batch": "▶️ Run batch check",
        "api_down": "⚠️ Backend not reachable at {url}. Start the API first.",
        "no_samples": "⚠️ Raw dataset not found, filled with zeros. "
        "Download creditcard.csv into data/raw/ for real samples.",
        "decision": "Decision",
        "fraud_prob": "Fraud probability",
        "risk": "Risk level",
        "model": "Model",
    },
    "uz": {
        "title": "# 🛡️ FraudGuard — Real vaqtda firibgarlikni aniqlash",
        "subtitle": "Tranzaksiyani baholang va avtomatik qaror oling: "
        "**Tasdiqlash**, **Ko'rib chiqish** yoki **Bloklash**.",
        "lang_label": "Language / Til",
        "tab_check": "🔍 Tranzaksiyani tekshirish",
        "tab_results": "📊 Model natijalari",
        "tab_batch": "📈 Ommaviy tekshiruv",
        "inputs": "Tranzaksiya ma'lumotlari",
        "amount": "Summa",
        "time": "Vaqt (birinchi tranzaksiyadan beri soniyalar)",
        "vfeatures": "PCA belgilari V1–V28",
        "btn_normal": "🎲 Tasodifiy oddiy namuna",
        "btn_fraud": "🎲 Tasodifiy firibgar namuna",
        "btn_check": "🔍 TEKSHIRISH",
        "result": "### Natija",
        "results_heading": "## 📊 Modellar taqqoslanishi",
        "results_desc": "To'rt model sinov to'plamida taqqoslangan "
        "(PR-AUC bo'yicha tartiblangan; nomutanosib ma'lumotda aniqlik ishlatilmaydi).",
        "cm_heading": "### Chalkashlik matritsasi (eng yaxshi model)",
        "batch_heading": "## 📈 Ommaviy tekshiruv",
        "batch_desc": "Time, V1..V28, Amount ustunli CSV yuklang. "
        "Har bir qator baholanadi va quyida ko'rsatiladi.",
        "batch_upload": "CSV yuklash",
        "btn_batch": "▶️ Ommaviy tekshiruvni boshlash",
        "api_down": "⚠️ Backend {url} manzilida ishlamayapti. Avval API'ni ishga tushiring.",
        "no_samples": "⚠️ Xom ma'lumotlar topilmadi, nollar bilan to'ldirildi. "
        "Haqiqiy namunalar uchun creditcard.csv ni data/raw/ ichiga yuklang.",
        "decision": "Qaror",
        "fraud_prob": "Firibgarlik ehtimoli",
        "risk": "Xavf darajasi",
        "model": "Model",
    },
}

# Decision labels & colors per language
DECISION_LABELS = {
    "APPROVE": {"en": "✅ APPROVED", "uz": "✅ TASDIQLANDI", "color": "#16a34a"},
    "REVIEW": {"en": "⚠️ REVIEW", "uz": "⚠️ KO'RIB CHIQISH", "color": "#d97706"},
    "BLOCK": {"en": "🛑 BLOCKED", "uz": "🛑 BLOKLANDI", "color": "#dc2626"},
}
RISK_LABELS = {
    "LOW": {"en": "LOW", "uz": "PAST"},
    "MEDIUM": {"en": "MEDIUM", "uz": "O'RTA"},
    "HIGH": {"en": "HIGH", "uz": "YUQORI"},
}


# --------------------------------------------------------------------------- #
# Sample loading (raw, unscaled values so the API can scale them correctly)
# --------------------------------------------------------------------------- #
_SAMPLE_CACHE: dict | None = None


def _load_samples() -> dict | None:
    """Cache raw test-split rows for the random-sample buttons."""
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is not None:
        return _SAMPLE_CACHE
    try:
        from src.data_loader import get_train_test

        _, X_test, _, y_test = get_train_test()
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        _SAMPLE_CACHE = {
            "normal": X_test[y_test == 0],
            "fraud": X_test[y_test == 1],
        }
    except Exception as exc:  # dataset missing -> degrade gracefully
        print(f"[FraudGuard UI] Samples unavailable: {exc}")
        _SAMPLE_CACHE = None
    return _SAMPLE_CACHE


def _sample_row(kind: str):
    """Return field values for a random normal/fraud sample (or zeros)."""
    samples = _load_samples()
    if samples is None or samples.get(kind) is None or samples[kind].empty:
        zeros = {c: 0.0 for c in config.FEATURE_COLUMNS}
        zeros["Amount"] = 100.0
        return zeros
    row = samples[kind].sample(1).iloc[0]
    return {c: float(row[c]) for c in config.FEATURE_COLUMNS}


# --------------------------------------------------------------------------- #
# API calls
# --------------------------------------------------------------------------- #
def _call_predict(payload: dict) -> dict:
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _result_html(res: dict, lang: str) -> str:
    dec = res["decision"]
    info = DECISION_LABELS[dec]
    label = info[lang]
    color = info["color"]
    pct = res["fraud_probability"] * 100
    risk = RISK_LABELS.get(res["risk_level"], {}).get(lang, res["risk_level"])
    t = T[lang]
    bar_w = max(2, min(100, pct))
    return f"""
    <div style="border:2px solid {color};border-radius:12px;padding:18px;
                font-family:system-ui,sans-serif;">
      <div style="font-size:26px;font-weight:700;color:{color};">{label}</div>
      <div style="margin-top:10px;">{t['fraud_prob']}: <b>{pct:.1f}%</b></div>
      <div style="background:#e5e7eb;border-radius:8px;height:16px;margin:6px 0;">
        <div style="width:{bar_w}%;background:{color};height:16px;border-radius:8px;"></div>
      </div>
      <div>{t['risk']}: <b>{risk}</b></div>
      <div>{t['model']}: <b>{res['model_used']}</b></div>
    </div>
    """


def check_transaction(lang, amount, time_val, *v_values):
    """Build payload, call API, return result HTML."""
    payload = {"Time": float(time_val), "Amount": float(amount)}
    for name, val in zip(V_COLUMNS, v_values):
        payload[name] = float(val)
    try:
        res = _call_predict(payload)
    except requests.exceptions.RequestException:
        return f"<p style='color:#dc2626'>{T[lang]['api_down'].format(url=API_URL)}</p>"
    return _result_html(res, lang)


def load_sample(kind: str):
    row = _sample_row(kind)
    return [row["Amount"], row["Time"]] + [row[c] for c in V_COLUMNS]


def get_comparison_table() -> pd.DataFrame:
    from src.evaluate import comparison_dataframe

    return comparison_dataframe()


def get_confusion_image():
    p = config.CONFUSION_MATRIX_PATH
    return str(p) if Path(p).exists() else None


def run_batch(file):
    """Score every row of an uploaded CSV via the API."""
    if file is None:
        return pd.DataFrame()
    df = pd.read_csv(file.name)
    out_rows = []
    for _, row in df.iterrows():
        payload = {c: float(row[c]) for c in config.FEATURE_COLUMNS if c in row}
        try:
            res = _call_predict(payload)
            out_rows.append(
                {
                    "Amount": payload.get("Amount"),
                    "fraud_probability": res["fraud_probability"],
                    "risk_level": res["risk_level"],
                    "decision": res["decision"],
                }
            )
        except requests.exceptions.RequestException:
            out_rows.append({"Amount": payload.get("Amount"), "decision": "API ERROR"})
    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# Build the interface
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    default_lang = "en"
    t = T[default_lang]

    with gr.Blocks(title="FraudGuard", theme=gr.themes.Soft()) as demo:
        lang_state = gr.State(default_lang)

        title_md = gr.Markdown(t["title"])
        subtitle_md = gr.Markdown(t["subtitle"])
        lang_radio = gr.Radio(
            choices=[("English", "en"), ("O'zbek", "uz")],
            value="en",
            label=t["lang_label"],
        )

        with gr.Tabs():
            # ---------------- Tab 1: Check ----------------
            with gr.Tab(t["tab_check"]):
                inputs_md = gr.Markdown(f"#### {t['inputs']}")
                with gr.Row():
                    amount_in = gr.Number(label=t["amount"], value=100.0)
                    time_in = gr.Number(label=t["time"], value=0.0)
                vfeatures_md = gr.Markdown(f"#### {t['vfeatures']}")
                v_inputs = []
                # 4 rows of 7 V-features
                for start in range(0, 28, 7):
                    with gr.Row():
                        for i in range(start, min(start + 7, 28)):
                            v_inputs.append(
                                gr.Number(label=V_COLUMNS[i], value=0.0)
                            )
                with gr.Row():
                    btn_normal = gr.Button(t["btn_normal"])
                    btn_fraud = gr.Button(t["btn_fraud"])
                btn_check = gr.Button(t["btn_check"], variant="primary")
                result_md = gr.Markdown(t["result"])
                result_html = gr.HTML()

                btn_check.click(
                    check_transaction,
                    inputs=[lang_state, amount_in, time_in, *v_inputs],
                    outputs=result_html,
                )
                btn_normal.click(
                    lambda: load_sample("normal"),
                    outputs=[amount_in, time_in, *v_inputs],
                )
                btn_fraud.click(
                    lambda: load_sample("fraud"),
                    outputs=[amount_in, time_in, *v_inputs],
                )

            # ---------------- Tab 2: Model Results ----------------
            with gr.Tab(t["tab_results"]):
                results_heading = gr.Markdown(t["results_heading"])
                results_desc = gr.Markdown(t["results_desc"])
                comparison_df = gr.Dataframe(
                    value=get_comparison_table(), interactive=False
                )
                cm_heading = gr.Markdown(t["cm_heading"])
                cm_image = gr.Image(value=get_confusion_image(), label="", height=320)
                refresh_btn = gr.Button("🔄 Refresh / Yangilash")
                refresh_btn.click(
                    lambda: (get_comparison_table(), get_confusion_image()),
                    outputs=[comparison_df, cm_image],
                )

            # ---------------- Tab 3: Batch Check ----------------
            with gr.Tab(t["tab_batch"]):
                batch_heading = gr.Markdown(t["batch_heading"])
                batch_desc = gr.Markdown(t["batch_desc"])
                file_in = gr.File(label=t["batch_upload"], file_types=[".csv"])
                btn_batch = gr.Button(t["btn_batch"], variant="primary")
                batch_out = gr.Dataframe(interactive=False)
                btn_batch.click(run_batch, inputs=file_in, outputs=batch_out)

        # ------------- Language switching -------------
        def switch_language(lang):
            tt = T[lang]
            return (
                lang,
                gr.update(value=tt["title"]),
                gr.update(value=tt["subtitle"]),
                gr.update(label=tt["lang_label"]),
                gr.update(value=f"#### {tt['inputs']}"),
                gr.update(label=tt["amount"]),
                gr.update(label=tt["time"]),
                gr.update(value=f"#### {tt['vfeatures']}"),
                gr.update(value=tt["btn_normal"]),
                gr.update(value=tt["btn_fraud"]),
                gr.update(value=tt["btn_check"]),
                gr.update(value=tt["result"]),
                gr.update(value=tt["results_heading"]),
                gr.update(value=tt["results_desc"]),
                gr.update(value=tt["cm_heading"]),
                gr.update(value=tt["batch_heading"]),
                gr.update(value=tt["batch_desc"]),
                gr.update(label=tt["batch_upload"]),
                gr.update(value=tt["btn_batch"]),
            )

        lang_radio.change(
            switch_language,
            inputs=lang_radio,
            outputs=[
                lang_state,
                title_md,
                subtitle_md,
                lang_radio,
                inputs_md,
                amount_in,
                time_in,
                vfeatures_md,
                btn_normal,
                btn_fraud,
                btn_check,
                result_md,
                results_heading,
                results_desc,
                cm_heading,
                batch_heading,
                batch_desc,
                file_in,
                btn_batch,
            ],
        )

    return demo


if __name__ == "__main__":
    # Create a public (shareable) URL by default so the app is reachable by
    # anyone, not just on localhost. Set FRAUDGUARD_SHARE=false to disable.
    share = os.environ.get("FRAUDGUARD_SHARE", "true").lower() == "true"
    server_port = int(os.environ.get("FRAUDGUARD_UI_PORT", "7860"))
    print(
        "Launching FraudGuard UI "
        f"(share={'on - public URL will be printed below' if share else 'off'}) ..."
    )
    build_ui().launch(server_name="0.0.0.0", server_port=server_port, share=share)
