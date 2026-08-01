"""
Feature engineering for intraday realized volatility forecasting.

Core idea (HAR-RV, Corsi 2009): realized vol is well explained by a mix of
short/medium/long lookback realized vol averages. We extend this with a few
extra intraday-specific features and let a gradient-boosted model learn
nonlinear interactions the linear HAR-RV baseline can't capture.
"""

import numpy as np
import pandas as pd

import config


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"] / df["close"].shift(1))


def minutes_since_open(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Minutes elapsed since the 09:30 ET open, clipped at 0.

    `pd.Index` has no `.clip()`, so the arithmetic is done in numpy. The index is
    assumed tz-aware US/Eastern (data_fetch normalizes both yfinance and Alpaca
    bars to that), otherwise the hour component would be meaningless.
    """
    mins = np.asarray(index.hour) * 60 + np.asarray(index.minute) - (9 * 60 + 30)
    return np.clip(mins, 0, None)


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Rolling realized volatility = sqrt of sum of squared returns over window bars."""
    return returns.rolling(window).apply(lambda x: np.sqrt(np.sum(x ** 2)), raw=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix + target for a single ticker's OHLCV bars.
    Target: realized vol over the NEXT `FORECAST_HORIZON` bars (forward-looking, shifted).
    """
    out = pd.DataFrame(index=df.index)
    ret = compute_log_returns(df)
    out["ret"] = ret

    # HAR-RV style features: RV over short/medium/long windows
    for w in config.RV_WINDOWS:
        out[f"rv_{w}"] = realized_vol(ret, w)

    # Range-based vol proxy (Parkinson) as an extra feature — uses high/low, captures
    # intrabar vol that close-to-close returns miss
    out["parkinson"] = np.sqrt(
        (1.0 / (4 * np.log(2))) * (np.log(df["high"] / df["low"])) ** 2
    ).rolling(config.RV_WINDOWS[0]).mean()

    # Volume-based feature: normalized volume (proxy for liquidity/participation shifts)
    out["vol_z"] = (df["volume"] - df["volume"].rolling(60).mean()) / (
        df["volume"].rolling(60).std() + 1e-9
    )

    # Time-of-day feature: minutes since market open (vol is typically higher at open/close)
    out["minutes_since_open"] = minutes_since_open(out.index)

    # Target: forward realized vol over next FORECAST_HORIZON bars
    fwd_ret = ret.shift(-1)
    out["target_rv"] = fwd_ret.rolling(config.FORECAST_HORIZON).apply(
        lambda x: np.sqrt(np.sum(x ** 2)), raw=True
    ).shift(-(config.FORECAST_HORIZON - 1))

    return out.dropna()


def har_rv_baseline_predict(feat_row: pd.Series) -> float:
    """
    Simple HAR-RV baseline: average of short/medium/long RV as the naive forecast.
    This is the benchmark the ML model needs to beat to justify its complexity.
    """
    windows = config.RV_WINDOWS
    return float(np.mean([feat_row[f"rv_{w}"] for w in windows]))
