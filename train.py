"""
Train an XGBoost volatility forecasting model per ticker and compare against
the HAR-RV baseline using walk-forward (expanding window) validation.

The baseline is a Corsi (2009) HAR-RV regression re-fit on each training fold,
not a fixed formula — anything easier than that flatters the model. With a
~1-2% edge and only a handful of folds, the point estimate alone says nothing,
so a Diebold-Mariano test is reported alongside it.

Usage:
    python train.py --ticker SPY
    python train.py --all
"""

import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

import config
import data_fetch
import features
from features import FEATURE_COLS

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)


def walk_forward_split(n_rows: int, n_folds: int = 5):
    """Expanding-window walk-forward split: train on past, test on next chunk."""
    fold_size = n_rows // (n_folds + 1)
    for i in range(1, n_folds + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)
        yield range(0, train_end), range(train_end, min(test_end, n_rows))


def diebold_mariano(err_model: np.ndarray, err_base: np.ndarray, horizon: int):
    """
    Diebold-Mariano test on absolute-error loss differentials.

    Overlapping forecasts (each bar's target shares bars with its neighbours')
    make the differentials strongly autocorrelated, so the variance uses a
    Newey-West HAC correction out to `horizon - 1` lags. Skipping that would
    understate the standard error and manufacture significance.

    Returns (dm_stat, p_value). Negative dm_stat means the model beats baseline.
    """
    d = np.abs(err_model) - np.abs(err_base)
    n = len(d)
    if n < 10:
        return None, None

    d_bar = d.mean()
    dev = d - d_bar
    var = np.mean(dev ** 2)
    for lag in range(1, min(horizon, n)):
        cov = np.mean(dev[lag:] * dev[:-lag])
        var += 2.0 * (1.0 - lag / horizon) * cov
    if var <= 0:
        return None, None

    dm = d_bar / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def train_and_evaluate(ticker: str) -> dict:
    print(f"\n=== {ticker} ===")
    raw = data_fetch.fetch_historical_intraday(ticker, period="60d", interval="5m")
    feat = features.build_features(raw)

    if len(feat) < 200:
        print(f"  [skip] Not enough intraday history for {ticker} ({len(feat)} rows).")
        return {}

    X = feat[FEATURE_COLS]
    y = feat["target_rv"]

    fold_rows = []
    all_err_model, all_err_base = [], []

    for k, (train_idx, test_idx) in enumerate(walk_forward_split(len(feat), n_folds=5), 1):
        tr, te = list(train_idx), list(test_idx)
        X_train, y_train = X.iloc[tr], y.iloc[tr]
        X_test, y_test = X.iloc[te], y.iloc[te]
        if len(X_train) < 50 or len(X_test) < 10:
            continue

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 0.0, None)

        # HAR-RV baseline is re-fit on the same training data the model saw
        har_params = features.fit_har_baseline(X_train, y_train)
        baseline_preds = features.har_baseline_predict(har_params, X_test)

        m_mae = mean_absolute_error(y_test, preds)
        b_mae = mean_absolute_error(y_test, baseline_preds)
        fold_rows.append({
            "fold": k,
            "n_test": len(te),
            "model_mae": m_mae,
            "baseline_mae": b_mae,
            "improvement_pct": (b_mae - m_mae) / b_mae * 100 if b_mae else np.nan,
        })
        all_err_model.append(preds - y_test.to_numpy())
        all_err_base.append(baseline_preds - y_test.to_numpy())

    if not fold_rows:
        print(f"  [skip] No usable walk-forward folds for {ticker}.")
        return {}

    folds = pd.DataFrame(fold_rows)
    avg_model_mae = float(folds["model_mae"].mean())
    avg_baseline_mae = float(folds["baseline_mae"].mean())
    improvement = (avg_baseline_mae - avg_model_mae) / avg_baseline_mae * 100
    folds_won = int((folds["improvement_pct"] > 0).sum())

    dm, p_value = diebold_mariano(
        np.concatenate(all_err_model), np.concatenate(all_err_base),
        horizon=config.FORECAST_HORIZON,
    )

    print("  per-fold MAE:")
    for r in fold_rows:
        print(f"    fold {r['fold']}  model={r['model_mae']:.6f}  "
              f"HAR-RV={r['baseline_mae']:.6f}  {r['improvement_pct']:+.1f}%")
    print(f"  mean MAE — model: {avg_model_mae:.6f} | HAR-RV baseline: {avg_baseline_mae:.6f}")
    print(f"  improvement over baseline: {improvement:+.2f}%  "
          f"(won {folds_won}/{len(folds)} folds)")
    if dm is not None:
        verdict = "significant" if p_value < 0.05 else "NOT significant"
        print(f"  Diebold-Mariano: stat={dm:+.2f}  p={p_value:.3f}  -> {verdict} at 5%")

    # Fit final model + baseline on ALL available data for live deployment
    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X, y)
    final_model.save_model(config.MODEL_PATH_TEMPLATE.format(ticker=ticker))

    har_params = features.fit_har_baseline(X, y)
    with open(config.BASELINE_PATH_TEMPLATE.format(ticker=ticker), "w") as f:
        json.dump(har_params, f, indent=2)

    metrics = {
        "ticker": ticker,
        "model_mae": avg_model_mae,
        "baseline_mae": avg_baseline_mae,
        "improvement_pct": improvement,
        "folds_won": folds_won,
        "n_folds": len(folds),
        "dm_stat": dm,
        "dm_p_value": p_value,
        "n_train_rows": len(feat),
        "folds": fold_rows,
    }
    with open(config.METRICS_PATH_TEMPLATE.format(ticker=ticker), "w") as f:
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
        if not r:
            continue
        sig = "" if r["dm_p_value"] is None else (
            f"  DM p={r['dm_p_value']:.3f}"
            + ("" if r["dm_p_value"] < 0.05 else " (not significant)")
        )
        print(f"{r['ticker']}: {r['improvement_pct']:+.2f}% vs HAR-RV baseline, "
              f"won {r['folds_won']}/{r['n_folds']} folds{sig}")
