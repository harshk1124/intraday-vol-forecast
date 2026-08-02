import os
from dotenv import load_dotenv

load_dotenv()

# --- Universe ---
# US broad-market indices plus the complete SPDR sector family. The universe is
# defined by what it *is*, not by what performed — every sector is included and
# reported, including the ones the model fails to beat HAR-RV on. Keeping only
# the names that showed an edge would be selecting on the outcome, which is the
# fastest way to turn an honest result into an unreproducible one.
TICKERS = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]
DEFAULT_TICKER = "SPY"

# --- Realized vol feature windows (in bars) ---
RV_WINDOWS = [5, 10, 20, 60]        # short/medium windows for HAR-RV style features
FORECAST_HORIZON = 12               # predict RV over next N bars (e.g. 12 x 5min = 1hr)
BAR_TIMEFRAME_MINUTES = 5           # intraday bar size
VOLUME_Z_WINDOW = 60                # lookback for the volume z-score feature

# --- Paths ---
MODEL_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH_TEMPLATE = os.path.join(MODEL_DIR, "{ticker}_vol_model.json")
BASELINE_PATH_TEMPLATE = os.path.join(MODEL_DIR, "{ticker}_har_baseline.json")
METRICS_PATH_TEMPLATE = os.path.join(MODEL_DIR, "{ticker}_metrics.json")
LATEST_FORECAST_PATH = os.path.join(DATA_DIR, "latest_forecast.json")

# Append-only record of forecasts, kept at the repo root rather than under
# DATA_DIR because it is committed: it is the only place genuinely
# out-of-sample predictions accumulate, and a hosted runtime's filesystem does
# not survive a restart. Committing it is what makes the history durable.
FORECAST_LOG_PATH = "forecast_log.csv"
FORECAST_LOG_COLUMNS = [
    "ticker", "bar_timestamp", "logged_at", "last_price",
    "predicted_rv", "baseline_rv",
]

# --- Alpaca API (free paper trading account, sign up at alpaca.markets) ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
USE_ALPACA = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)

# --- Market hours (US/Eastern) ---
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
