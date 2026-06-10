"""
FraudGuard — Synthetic dataset generator.

Generates a large, *hyper-realistic* credit-card transaction dataset that mirrors
the structure and statistics of the real Kaggle "Credit Card Fraud Detection"
dataset (mlg-ulb/creditcardfraud), so that the full training/evaluation pipeline
(`python -m src.train`) can run without downloading the proprietary CSV.

What makes it realistic
------------------------
* **Exact schema** of the real file: ``Time, V1..V28, Amount, Class`` (31 cols).
* **PCA-like V features**: V1..V28 are continuous, ~zero-mean, with *decreasing
  variance* (V1 widest, V28 narrowest) — exactly how PCA components behave.
* **Class-dependent shifts**: fraud transactions shift the means of the features
  that are genuinely discriminative in the real data (V1, V3, V4, V7, V10, V11,
  V12, V14, V16, V17, ...), while non-discriminative features (V13, V15, V22..V26)
  stay essentially identical between classes — this keeps the problem *learnable
  but not trivially separable*.
* **Hard cases**: a fraction of frauds are drawn to look almost like normal
  traffic, and a few normal transactions look suspicious — this produces a
  realistic precision/recall trade-off instead of a perfect classifier.
* **Severe imbalance**: ~0.17% fraud, matching the real dataset.
* **Realistic Amount**: heavy-tailed log-normal amounts; frauds skew toward many
  small "card-testing" charges plus occasional large ones.
* **Realistic Time**: seconds over a ~2-day window with a day/night sinusoidal
  transaction-rate pattern.

Reproducible: fixed RNG seed. Run::

    python scripts/generate_dataset.py                 # default ~284,807 rows
    python scripts/generate_dataset.py --rows 300000   # custom size
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src` importable so we reuse the canonical schema/paths.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

# --------------------------------------------------------------------------- #
# Per-feature statistics (approximating the real creditcard.csv)
# --------------------------------------------------------------------------- #
# Standard deviation of each PCA component for the (dominant) non-fraud class.
# PCA components are ordered by explained variance, so std decreases V1 -> V28.
NONFRAUD_STD = np.array([
    1.95, 1.65, 1.52, 1.42, 1.38, 1.33, 1.24, 1.19, 1.10, 1.09,  # V1..V10
    1.02, 1.00, 0.99, 0.96, 0.92, 0.88, 0.85, 0.84, 0.81, 0.77,  # V11..V20
    0.73, 0.73, 0.62, 0.61, 0.52, 0.48, 0.40, 0.33,              # V21..V28
])

# Mean of each feature for the fraud class (non-fraud mean is ~0 for all).
# These shifts reproduce the directions/magnitudes seen in the real data: the
# strongly discriminative features (V3, V4, V7, V10, V11, V12, V14, V16, V17)
# move a lot; "noise" features (V13, V15, V22..V26) barely move.
FRAUD_MEAN = np.array([
    -4.77,  3.42, -7.03,  4.54, -3.15, -1.40, -5.59,  0.57, -2.58, -5.68,  # V1..V10
     3.80, -6.26, -0.10, -6.97, -0.09, -4.14, -6.67, -2.43,  0.69,  0.37,  # V11..V20
     0.71,  0.01, -0.04, -0.10,  0.05,  0.05,  0.29,  0.16,               # V21..V28
])

# Frauds are also noisier on the discriminative axes: per-feature std multiplier.
FRAUD_STD_SCALE = np.array([
    1.6, 1.7, 2.1, 1.5, 1.8, 1.4, 2.0, 1.5, 1.6, 2.0,  # V1..V10
    1.6, 2.1, 1.0, 2.2, 1.0, 1.8, 2.1, 1.6, 1.3, 1.4,  # V11..V20
    1.4, 1.0, 1.1, 1.0, 1.0, 1.0, 1.5, 1.4,            # V21..V28
])

N_FEATURES = 28
assert NONFRAUD_STD.shape == FRAUD_MEAN.shape == FRAUD_STD_SCALE.shape == (N_FEATURES,)

DEFAULT_ROWS = 284_807          # match the real dataset size
DEFAULT_FRAUD_RATIO = 0.001727  # ~492 / 284807, the real ratio
TIME_WINDOW_SECONDS = 172_792   # ~2 days, as in the real data
HARD_FRAUD_FRACTION = 0.18      # frauds that look almost like normal traffic


def _generate_time(n: int, rng: np.random.Generator) -> np.ndarray:
    """Seconds since first transaction, with a day/night intensity pattern."""
    # Inverse-CDF-ish sampling of a sinusoidal daily rate over the 2-day window.
    base = rng.uniform(0.0, 1.0, size=n)
    # Two daily humps (more transactions during the day).
    daily = 0.5 * (1.0 - np.cos(2.0 * np.pi * base * 2.0))  # 0..1, two peaks
    t = (base * 0.65 + daily * 0.35) * TIME_WINDOW_SECONDS
    return np.clip(t, 0, TIME_WINDOW_SECONDS).astype(np.int64).astype(float)


def _generate_amount(n: int, rng: np.random.Generator, fraud: bool) -> np.ndarray:
    """Heavy-tailed log-normal amounts (in the dataset's currency units)."""
    if fraud:
        # Many small "card-testing" charges + a heavier tail of large ones.
        small = rng.lognormal(mean=2.2, sigma=1.3, size=n)
        big = rng.lognormal(mean=5.4, sigma=0.9, size=n)
        pick_big = rng.random(n) < 0.22
        amt = np.where(pick_big, big, small)
        # A slice of frauds are zero-amount authorizations (as in the real data).
        amt[rng.random(n) < 0.06] = 0.0
    else:
        amt = rng.lognormal(mean=3.0, sigma=1.35, size=n)
    return np.round(np.clip(amt, 0.0, 25_000.0), 2)


def _generate_features(
    n: int, rng: np.random.Generator, fraud: bool
) -> np.ndarray:
    """Generate the V1..V28 matrix for a single class."""
    if not fraud:
        return rng.normal(0.0, NONFRAUD_STD, size=(n, N_FEATURES))

    # Fraud: shifted means + inflated variance on discriminative axes.
    fraud_std = NONFRAUD_STD * FRAUD_STD_SCALE
    feats = rng.normal(FRAUD_MEAN, fraud_std, size=(n, N_FEATURES))

    # "Hard" frauds: a subset is pulled back toward the normal distribution so
    # the classes overlap and the model cannot achieve a perfect score.
    n_hard = int(n * HARD_FRAUD_FRACTION)
    if n_hard:
        idx = rng.choice(n, size=n_hard, replace=False)
        blend = rng.uniform(0.55, 0.9, size=(n_hard, 1))  # how "normal" they look
        normal_like = rng.normal(0.0, NONFRAUD_STD, size=(n_hard, N_FEATURES))
        feats[idx] = blend * normal_like + (1.0 - blend) * feats[idx]
    return feats


def generate_dataset(
    rows: int = DEFAULT_ROWS,
    fraud_ratio: float = DEFAULT_FRAUD_RATIO,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Build the full synthetic dataset as a DataFrame with the real schema."""
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(round(rows * fraud_ratio)))
    n_normal = rows - n_fraud

    # Feature matrices per class.
    Xn = _generate_features(n_normal, rng, fraud=False)
    Xf = _generate_features(n_fraud, rng, fraud=True)

    # A few normal transactions look suspicious (false-positive pressure).
    n_susp = int(n_normal * 0.002)
    if n_susp:
        idx = rng.choice(n_normal, size=n_susp, replace=False)
        blend = rng.uniform(0.3, 0.6, size=(n_susp, 1))
        Xn[idx] = (1.0 - blend) * Xn[idx] + blend * rng.normal(
            FRAUD_MEAN, NONFRAUD_STD, size=(n_susp, N_FEATURES)
        )

    X = np.vstack([Xn, Xf])
    time = np.concatenate([
        _generate_time(n_normal, rng),
        _generate_time(n_fraud, rng),
    ])
    amount = np.concatenate([
        _generate_amount(n_normal, rng, fraud=False),
        _generate_amount(n_fraud, rng, fraud=True),
    ])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])

    # Shuffle so fraud rows are not all at the end, then sort by Time like the
    # real dataset (which is ordered chronologically).
    order = np.argsort(time, kind="stable")
    X, time, amount, y = X[order], time[order], amount[order], y[order]

    data = {"Time": time}
    for i in range(N_FEATURES):
        data[f"V{i + 1}"] = X[:, i]
    data["Amount"] = amount
    data["Class"] = y

    df = pd.DataFrame(data)
    # Enforce the canonical column order from config.
    return df[[*config.FEATURE_COLUMNS, config.TARGET_COLUMN]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FraudGuard dataset.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--fraud-ratio", type=float, default=DEFAULT_FRAUD_RATIO)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument(
        "--out", type=str, default=str(config.RAW_DATA_PATH),
        help="Output CSV path (default: data/raw/creditcard.csv)",
    )
    parser.add_argument(
        "--decimals", type=int, default=6,
        help="Decimal places for V features (controls file size).",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.rows:,} transactions (seed={args.seed}) ...")
    df = generate_dataset(args.rows, args.fraud_ratio, args.seed)

    n_fraud = int(df[config.TARGET_COLUMN].sum())
    print(f"  rows         : {len(df):,}")
    print(f"  fraud cases  : {n_fraud:,} ({n_fraud / len(df) * 100:.4f}%)")
    print(f"  columns      : {df.shape[1]} -> {list(df.columns)[:3]} ... 'Class'")

    df.to_csv(out_path, index=False, float_format=f"%.{args.decimals}f")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved -> {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
