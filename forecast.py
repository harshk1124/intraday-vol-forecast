"""
Produce a live volatility forecast using the trained model + most recent bars.

Feature construction is delegated entirely to `features.build_features` so the
serving path cannot drift from the training path — an earlier version
re-implemented the feature math inline here, which meant any change to
features.py silently applied to training only.
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
from features import FEATURE_COLS

# Enough history to fill the longest rolling window, plus headroom for the
# overnight gaps that get masked out.
MIN_LOOKBACK_BARS = max(max(config.RV_WINDOWS), config.VOLUME_Z_WINDOW) + 10


def load_model(ticker: str) -> xgb.XGBRegressor:
    path = config.MODEL_PATH_TEMPLATE.format(ticker=ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found for {ticker}. Run `python train.py --ticker {ticker}` first."
        )
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def load_har_baseline(ticker: str) -> dict:
    path = config.BASELINE_PATH_TEMPLATE.format(ticker=ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No fitted HAR-RV baseline for {ticker}. Run `python train.py --ticker {ticker}` first."
        )
    with open(path) as f:
        return json.load(f)


def get_live_forecast(ticker: str) -> dict:
    """Fetch latest bars, build features, and return model + baseline forecasts."""
    raw = data_fetch.fetch_live_bars(ticker, lookback_minutes=240)
    feat_live = features.build_features(raw, with_target=False)

    if feat_live.empty:
        raise ValueError(
            f"Not enough recent data to build features for {ticker} "
            f"(need at least {MIN_LOOKBACK_BARS} bars, got {len(raw)})."
        )

    latest_row = feat_live.iloc[[-1]]
    model = load_model(ticker)
    har_params = load_har_baseline(ticker)

    model_forecast = float(np.clip(model.predict(latest_row[FEATURE_COLS])[0], 0.0, None))
    baseline_forecast = float(features.har_baseline_predict(har_params, latest_row)[0])

    return {
        "ticker": ticker,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bar_timestamp": latest_row.index[-1].isoformat(),
        "last_price": float(raw["close"].iloc[-1]),
        "model_forecast_rv": model_forecast,
        "baseline_forecast_rv": baseline_forecast,
        "current_rv_short": float(latest_row[f"rv_{config.RV_WINDOWS[0]}"].iloc[0]),
        "market_open": data_fetch.is_market_open(),
    }


def _insert_session_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert an all-NaN row just before each new session.

    Without this, a line chart joins the last bar of one session to the first
    bar of the next, drawing a straight diagonal across the entire overnight
    period as though it were a slow continuous move. The NaN splits the trace
    into one segment per session so the gap reads as a gap.
    """
    if df.empty:
        return df
    sess = features.session_ids(df.index)
    starts = np.flatnonzero((sess != sess.shift(1)).to_numpy())[1:]  # skip the first row
    if len(starts) == 0:
        return df
    spacers = pd.DataFrame(
        np.nan,
        index=df.index[starts] - pd.Timedelta(seconds=1),
        columns=df.columns,
    )
    return pd.concat([df, spacers]).sort_index()


def get_forecast_history(ticker: str, lookback_minutes: int = 240) -> pd.DataFrame:
    """
    Return recent price + realized vol series for charting.

    `lookback_minutes` is trimmed in *bars*, not wall-clock time: the underlying
    yfinance fetch always returns a fixed 2-day window regardless of what is
    asked for, and outside market hours a wall-clock cutoff would select an
    empty range.
    """
    raw = data_fetch.fetch_live_bars(ticker, lookback_minutes=lookback_minutes)
    ret = features.intraday_log_returns(raw)
    rv_short = features.realized_vol(ret, config.RV_WINDOWS[0])
    chart = pd.DataFrame({"close": raw["close"], "rv_short": rv_short}).dropna()

    n_bars = max(1, lookback_minutes // config.BAR_TIMEFRAME_MINUTES)
    chart = chart.iloc[-n_bars:]
    return _insert_session_breaks(chart)


def load_metrics(ticker: str) -> dict | None:
    """Walk-forward metrics written by train.py, or None if never trained here."""
    path = config.METRICS_PATH_TEMPLATE.format(ticker=ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_prediction_track(ticker: str, lookback_minutes: int = 390) -> pd.DataFrame:
    """
    Forecast-vs-outcome series: for each bar, what the model predicted for the
    next hour alongside what actually realized over that same hour.

    Everything is indexed by *forecast origin* — the bar the prediction was made
    from — so predicted and realized describe the identical forward window and
    can be compared point for point. `realized_rv` is necessarily NaN for the
    final FORECAST_HORIZON bars, whose windows have not finished yet, and for
    bars whose window would cross the close.

    Caveat: the deployed model was fit on all available history, so over recent
    bars this is in-sample fit. `load_metrics()` carries the walk-forward
    out-of-sample numbers, which are the honest measure of skill.
    """
    raw = data_fetch.fetch_live_bars(ticker, lookback_minutes=lookback_minutes)
    feats = features.build_features(raw, with_target=False)
    if feats.empty:
        raise ValueError(f"Not enough recent data to build a prediction track for {ticker}.")

    model = load_model(ticker)
    har_params = load_har_baseline(ticker)

    ret = features.intraday_log_returns(raw)
    realized = features.forward_realized_vol(
        ret, config.FORECAST_HORIZON, features.session_ids(raw.index)
    ).reindex(feats.index)

    track = pd.DataFrame(
        {
            "predicted_rv": np.clip(model.predict(feats[FEATURE_COLS]), 0.0, None),
            "baseline_rv": features.har_baseline_predict(har_params, feats),
            "realized_rv": realized.to_numpy(dtype=float),
        },
        index=feats.index,
    )

    n_bars = max(1, lookback_minutes // config.BAR_TIMEFRAME_MINUTES)
    return _insert_session_breaks(track.iloc[-n_bars:])


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
