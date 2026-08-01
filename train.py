"""
Train an XGBoost volatility forecasting model per ticker and compare against
the HAR-RV baseline using walk-forward (expanding window) validation.

Usage:
    python train.py --ticker SPY
    python train.py --all
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

import config
import data_fetch
import features


FEATURE_COLS = (
    ["ret", "parkinson", "vol_z", "minutes_since_open"]
    + [f"rv_{w}" for w in config.RV_WINDOWS]
)


def walk_forward_split(n_rows: int, n_folds: int = 5):
    """Expanding-window walk-forward split: train on past, test on next chunk."""
    fold_size = n_rows // (n_folds + 1)
    for i in range(1, n_folds + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)
        yield range(0, train_end), range(train_end, min(test_end, n_rows))


def train_and_evaluate(ticker: str) -> dict:
    print(f"\n=== {ticker} ===")
    raw = data_fetch.fetch_historical_intraday(ticker, period="60d", interval="5m")
    feat = features.build_features(raw)

    if len(feat) < 200:
        print(f"  [skip] Not enough intraday history for {ticker} ({len(feat)} rows).")
        return {}

    X = feat[FEATURE_COLS]
    y = feat["target_rv"]

    model_maes, baseline_maes = [], []
    for train_idx, test_idx in walk_forward_split(len(feat), n_folds=5):
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        X_test, y_test = X.iloc[list(test_idx)], y.iloc[list(test_idx)]
        if len(X_train) < 50 or len(X_test) < 10:
            continue

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        baseline_preds = feat.iloc[list(test_idx)].apply(features.har_rv_baseline_predict, axis=1)

        model_maes.append(mean_absolute_error(y_test, preds))
        baseline_maes.append(mean_absolute_error(y_test, baseline_preds))

    avg_model_mae = float(np.mean(model_maes)) if model_maes else None
    avg_baseline_mae = float(np.mean(baseline_maes)) if baseline_maes else None
    improvement = (
        (avg_baseline_mae - avg_model_mae) / avg_baseline_mae * 100
        if avg_model_mae and avg_baseline_mae else None
    )

    print(f"  Walk-forward MAE — model: {avg_model_mae:.5f} | HAR-RV baseline: {avg_baseline_mae:.5f}")
    if improvement is not None:
        print(f"  Improvement over baseline: {improvement:+.1f}%")

    # Fit final model on ALL available data for live deployment
    final_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    final_model.fit(X, y)
    final_model.save_model(config.MODEL_PATH_TEMPLATE.format(ticker=ticker))

    metrics = {
        "ticker": ticker,
        "model_mae": avg_model_mae,
        "baseline_mae": avg_baseline_mae,
        "improvement_pct": improvement,
        "n_train_rows": len(feat),
    }
    with open(config.BASELINE_PATH_TEMPLATE.format(ticker=ticker), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    tickers = config.TICKERS if args.all else [args.ticker or config.DEFAULT_TICKER]

    results = []
    for t in tickers:
        try:
            results.append(train_and_evaluate(t))
        except Exception as e:
            print(f"  [error] {t}: {e}")

    print("\n=== Summary ===")
    for r in results:
        if r:
            print(f"{r['ticker']}: {r['improvement_pct']:+.1f}% vs HAR-RV baseline"
                  if r.get('improvement_pct') is not None else f"{r['ticker']}: incomplete")
