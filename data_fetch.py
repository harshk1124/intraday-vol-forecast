"""
Data fetching utilities.

Historical intraday bars: yfinance (free, ~1min/5min bars, limited lookback ~60 days for
intraday granularity — enough for model training on recent regimes).

Live/near-real-time bars: Alpaca (free paper account, real-time IEX feed, no cost) if
API keys are configured; otherwise falls back to yfinance's delayed quotes.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone

import config

MARKET_TZ = "America/New_York"

# Recent yfinance versions fetch through curl_cffi with browser TLS
# impersonation enabled by default. Any TLS-inspecting middlebox — corporate
# egress proxies, debugging proxies, sandboxed CI networks — re-terminates the
# connection and resets the impersonated handshake, which surfaces as
# `SSLError('curl: (35) Recv failure: Connection reset by peer')` with no
# indication that the fingerprint was the cause. An ordinary session with a
# normal User-Agent skips the impersonation and works both behind such proxies
# and on a direct connection.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _build_session():
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return None
    session = curl_requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


_SESSION = _build_session()


def _download(ticker: str, **kwargs) -> pd.DataFrame:
    """yf.download with the shared non-impersonating session applied."""
    kwargs.setdefault("progress", False)
    kwargs.setdefault("auto_adjust", True)
    if _SESSION is None:
        return yf.download(ticker, **kwargs)
    try:
        return yf.download(ticker, session=_SESSION, **kwargs)
    except TypeError:
        # yfinance builds that don't accept an injected session
        return yf.download(ticker, **kwargs)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw OHLCV frame to a consistent shape:
      - single-level lowercase columns (yfinance returns MultiIndex (Price, Ticker)
        even for a single ticker, which silently makes df["close"] a DataFrame)
      - tz-aware index in US/Eastern, so time-of-day features mean the same thing
        whether the bars came from yfinance (ET) or Alpaca (UTC)
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)

    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx.tz_convert(MARKET_TZ)
    df.index.name = "timestamp"

    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected OHLCV columns {missing}; got {list(df.columns)}")
    return df[["open", "high", "low", "close", "volume"]]


def fetch_historical_intraday(ticker: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """
    Pull historical intraday bars via yfinance for model training.
    yfinance limits intraday history depending on interval (5m -> ~60 days).
    """
    df = _download(ticker, period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol or rate limits.")
    return _normalize(df)


def fetch_daily_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Pull daily history for longer-horizon realized vol baselines / regime context."""
    df = _download(ticker, period=period, interval="1d")
    if df.empty:
        raise ValueError(f"No daily data returned for {ticker}.")
    return _normalize(df)


def fetch_live_bars_alpaca(ticker: str, lookback_minutes: int = 240) -> pd.DataFrame:
    """Fetch recent real-time bars from Alpaca's free market data API."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    start = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)

    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(config.BAR_TIMEFRAME_MINUTES, TimeFrame.Unit.Minute),
        start=start,
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise ValueError(f"No live bars returned for {ticker} from Alpaca.")

    bars = bars.reset_index().set_index("timestamp")
    return _normalize(bars)


def fetch_live_bars(ticker: str, lookback_minutes: int = 240) -> pd.DataFrame:
    """
    Unified live-data entry point: uses Alpaca if configured (true real-time),
    otherwise falls back to yfinance (delayed ~15-20min, fine for 1hr-horizon forecasts).
    """
    if config.USE_ALPACA:
        try:
            return fetch_live_bars_alpaca(ticker, lookback_minutes)
        except Exception as e:
            print(f"[warn] Alpaca fetch failed ({e}), falling back to yfinance.")

    # yfinance fallback: pull last ~1 day of 5m bars, which includes the most recent
    # (delayed) quotes on the free tier
    df = _download(ticker, period="2d", interval=f"{config.BAR_TIMEFRAME_MINUTES}m")
    if df.empty:
        raise ValueError(f"No live bars returned for {ticker} from yfinance.")
    return _normalize(df)


def is_market_open() -> bool:
    """Simple US/Eastern market-hours check (does not account for holidays)."""
    import pytz
    now_et = datetime.now(pytz.timezone("US/Eastern"))
    if now_et.weekday() >= 5:
        return False
    open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_et <= close_t
