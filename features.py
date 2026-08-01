"""
Feature engineering for intraday realized volatility forecasting.

Core idea (HAR-RV, Corsi 2009): realized vol is well explained by a mix of
short/medium/long lookback realized vol averages. We extend this with a few
extra intraday-specific features and let a gradient-boosted model learn
nonlinear interactions the linear HAR-RV baseline can't capture.

Two details matter a great deal for getting an honest result:

1. Overnight gaps are not intraday volatility. The close-to-open return is
   roughly 7x the size of a typical 5-minute return, so on 5-min bars ~1.3% of
   observations carry ~40% of all squared returns. Left in, they swamp every
   rolling window and make the time-of-day feature a proxy for "is the next
   session boundary inside my forecast window", which is calendar lookup rather
   than volatility forecasting. Gap returns are therefore masked out.

2. The forecast target must not span a session boundary. Predicting "realized
   vol over the next hour" is meaningless 20 minutes before the close, so those
   targets are dropped rather than being allowed to reach into the next day.
"""

import numpy as np
import pandas as pd

import config

# Feature columns, in a fixed order. Shared by train.py and forecast.py so the
# training and serving paths can never silently drift apart.
FEATURE_COLS = (
    ["ret", "parkinson", "vol_z", "minutes_since_open"]
    + [f"rv_{w}" for w in config.RV_WINDOWS]
)

# The subset the linear HAR-RV baseline is allowed to use.
HAR_COLS = [f"rv_{w}" for w in config.RV_WINDOWS]


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"] / df["close"].shift(1))


def session_ids(index: pd.DatetimeIndex) -> pd.Series:
    """Trading-session label (calendar date in market tz) for each bar."""
    return pd.Series(index.normalize(), index=index)


def intraday_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    Log returns with the overnight gap removed.

    The first bar of each session is a close-to-open gap, not an intraday move;
    it is set to zero so it contributes nothing to realized vol.
    """
    ret = compute_log_returns(df)
    sess = session_ids(df.index)
    is_session_open = (sess != sess.shift(1)).values
    return ret.mask(is_session_open).fillna(0.0)


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
    return np.sqrt((returns ** 2).rolling(window).sum())


def forward_realized_vol(returns: pd.Series, horizon: int, sessions: pd.Series) -> pd.Series:
    """
    Realized vol over the NEXT `horizon` bars, NaN where that window would cross
    into the following session.
    """
    fwd_sq = (returns ** 2).shift(-1)
    tgt = np.sqrt(fwd_sq.rolling(horizon).sum()).shift(-(horizon - 1))
    within_session = (sessions.shift(-horizon) == sessions).values
    return tgt.where(within_session)


def build_features(df: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    """
    Build the feature matrix for a single ticker's OHLCV bars.

    with_target=True  -> training: includes `target_rv`, rows without one dropped.
    with_target=False -> serving: no target, so the most recent bar is retained.
    """
    out = pd.DataFrame(index=df.index)
    ret = intraday_log_returns(df)
    sess = session_ids(df.index)

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
    out["vol_z"] = (df["volume"] - df["volume"].rolling(config.VOLUME_Z_WINDOW).mean()) / (
        df["volume"].rolling(config.VOLUME_Z_WINDOW).std() + 1e-9
    )

    # Time-of-day feature: minutes since market open (vol is typically higher at open/close)
    out["minutes_since_open"] = minutes_since_open(out.index)

    if not with_target:
        return out.dropna(subset=FEATURE_COLS)

    out["target_rv"] = forward_realized_vol(ret, config.FORECAST_HORIZON, sess)
    return out.dropna()


# --- HAR-RV baseline -------------------------------------------------------
#
# The honest benchmark is Corsi's HAR-RV: an OLS regression of forward RV on
# short/medium/long realized vol. Note that a plain *average* of the RV windows
# is NOT a HAR-RV — RV scales with sqrt(window), so averaging rv_5 through rv_60
# to predict 12-bar RV overshoots the target by ~30% before any modelling
# happens. That inflated benchmark is trivially beatable and makes a model look
# far better than it is.


def fit_har_baseline(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit the linear HAR-RV baseline on a training fold. Returns saveable params."""
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression().fit(X[HAR_COLS], y)
    return {
        "columns": HAR_COLS,
        "coef": [float(c) for c in lr.coef_],
        "intercept": float(lr.intercept_),
    }


def har_baseline_predict(params: dict, X: pd.DataFrame) -> np.ndarray:
    """Apply fitted HAR-RV baseline params. Realized vol is non-negative, so clip at 0."""
    coef = np.asarray(params["coef"], dtype=float)
    preds = X[params["columns"]].to_numpy(dtype=float) @ coef + params["intercept"]
    return np.clip(preds, 0.0, None)
