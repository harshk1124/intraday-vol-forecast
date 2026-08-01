"""
Produce a live volatility forecast using the trained model + most recent bars.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

import config
import data_fetch
import features

FEATURE_COLS = (
    ["ret", "parkinson", "vol_z", "minutes_since_open"]
    + [f"rv_{w}" for w in config.RV_WINDOWS]
)


def load_model(ticker: str) -> xgb.XGBRegressor:
    path = config.MODEL_PATH_TEMPLATE.format(ticker=ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found for {ticker}. Run `python train.py --ticker {ticker}` first."
        )
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def get_live_forecast(ticker: str) -> dict:
    """Fetch latest bars, build features, and return model + baseline forecasts."""
    raw = data_fetch.fetch_live_bars(ticker, lookback_minutes=240)
    feat = features.build_features(raw)  # will drop last few rows needing forward target; that's fine

    # Use full feature set (including rows without a target, since we're forecasting forward)
    raw_ret = features.compute_log_returns(raw)
    feat_live = pd.DataFrame(index=raw.index)
    feat_live["ret"] = raw_ret
    for w in config.RV_WINDOWS:
        feat_live[f"rv_{w}"] = features.realized_vol(raw_ret, w)
    feat_live["parkinson"] = np.sqrt(
        (1.0 / (4 * np.log(2))) * (np.log(raw["high"] / raw["low"])) ** 2
    ).rolling(config.RV_WINDOWS[0]).mean()
    feat_live["vol_z"] = (raw["volume"] - raw["volume"].rolling(60).mean()) / (
        raw["volume"].rolling(60).std() + 1e-9
    )
    feat_live["minutes_since_open"] = features.minutes_since_open(feat_live.index)
    feat_live = feat_live.dropna()

    if feat_live.empty:
        raise ValueError(f"Not enough recent data to build features for {ticker}.")

    latest_row = feat_live.iloc[[-1]]
    model = load_model(ticker)
    model_forecast = float(model.predict(latest_row[FEATURE_COLS])[0])
    baseline_forecast = float(features.har_rv_baseline_predict(latest_row.iloc[0]))

    result = {
        "ticker": ticker,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "last_price": float(raw["close"].iloc[-1]),
        "model_forecast_rv": model_forecast,
        "baseline_forecast_rv": baseline_forecast,
        "current_rv_short": float(latest_row[f"rv_{config.RV_WINDOWS[0]}"].iloc[0]),
        "market_open": data_fetch.is_market_open(),
    }
    return result


def get_forecast_history(ticker: str, lookback_minutes: int = 240) -> pd.DataFrame:
    """Return recent price + realized vol series for charting."""
    raw = data_fetch.fetch_live_bars(ticker, lookback_minutes=lookback_minutes)
    ret = features.compute_log_returns(raw)
    rv_short = features.realized_vol(ret, config.RV_WINDOWS[0])
    chart_df = pd.DataFrame({"close": raw["close"], "rv_short": rv_short}).dropna()
    return chart_df


def save_latest_forecast(result: dict):
    existing = []
    if os.path.exists(config.LATEST_FORECAST_PATH):
        try:
            with open(config.LATEST_FORECAST_PATH) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing = [r for r in existing if r.get("ticker") != result["ticker"]]
    existing.append(result)
    with open(config.LATEST_FORECAST_PATH, "w") as f:
        json.dump(existing, f, indent=2)
