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
from datetime import datetime, timedelta

import config


def fetch_historical_intraday(ticker: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """
    Pull historical intraday bars via yfinance for model training.
    yfinance limits intraday history depending on interval (5m -> ~60 days).
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol or rate limits.")
    df = df.rename(columns=str.lower)
    df.index.name = "timestamp"
    return df[["open", "high", "low", "close", "volume"]]


def fetch_daily_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Pull daily history for longer-horizon realized vol baselines / regime context."""
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    df = df.rename(columns=str.lower)
    df.index.name = "timestamp"
    return df[["open", "high", "low", "close", "volume"]]


def fetch_live_bars_alpaca(ticker: str, lookback_minutes: int = 240) -> pd.DataFrame:
    """Fetch recent real-time bars from Alpaca's free market data API."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    start = datetime.utcnow() - timedelta(minutes=lookback_minutes)

    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(config.BAR_TIMEFRAME_MINUTES, TimeFrame.Unit.Minute),
        start=start,
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise ValueError(f"No live bars returned for {ticker} from Alpaca.")

    bars = bars.reset_index().set_index("timestamp")
    bars = bars.rename(columns=str.lower)
    return bars[["open", "high", "low", "close", "volume"]]


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
    df = yf.download(ticker, period="2d", interval=f"{config.BAR_TIMEFRAME_MINUTES}m",
                      progress=False, auto_adjust=True)
    df = df.rename(columns=str.lower)
    df.index.name = "timestamp"
    return df[["open", "high", "low", "close", "volume"]]


def is_market_open() -> bool:
    """Simple US/Eastern market-hours check (does not account for holidays)."""
    import pytz
    now_et = datetime.now(pytz.timezone("US/Eastern"))
    if now_et.weekday() >= 5:
        return False
    open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_et <= close_t
