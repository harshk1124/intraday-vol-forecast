# Intraday Realized Volatility Forecast

ML-based short-horizon (intraday) realized volatility forecasting, benchmarked against
a HAR-RV baseline, with a live Streamlit dashboard.

## Why this project

- **Data:** fully free — historical training via `yfinance`, live data via a free Alpaca
  paper-trading account (real-time IEX feed) with automatic fallback to `yfinance`'s
  delayed quotes if Alpaca isn't configured.
- **Methodology:** walk-forward (expanding window) validation, not a single train/test
  split — the model has to consistently beat the HAR-RV baseline across multiple periods,
  not just get lucky once.
- **Honest baseline:** HAR-RV (Corsi 2009) is a well-known, hard-to-beat benchmark in
  volatility forecasting research. Beating it — even by a little, consistently — is a
  meaningful result. A huge "beat" would be a red flag for overfitting, not a good sign.

## Setup

```bash
pip install -r requirements.txt
```

(Optional but recommended) Set up free real-time data via Alpaca:
1. Create a free account at https://alpaca.markets (paper trading, no cost, no card required)
2. Copy `.env.example` to `.env` and fill in your API keys
3. Without this, the app falls back to yfinance's delayed (~15-20 min) quotes — still
   fine for an hourly-horizon forecast, just not truly real-time

## Usage

**1. Train a model** (per ticker, using ~60 days of 5-min historical bars):
```bash
python train.py --ticker SPY
# or train all tickers in config.py:
python train.py --all
```
This prints walk-forward MAE for the model vs. the HAR-RV baseline and saves the
trained model to `models/`.

**2. Launch the live dashboard:**
```bash
streamlit run app.py
```
Select a ticker in the sidebar to see the live price chart, current realized vol,
and the model's forecast vs. baseline for the next hour.

## Project structure

```
config.py       - tickers, feature windows, paths, API key loading
data_fetch.py   - historical (yfinance) + live (Alpaca/yfinance) data pulls
features.py     - HAR-RV style feature engineering + forward RV target
train.py        - walk-forward training + baseline comparison, saves model
forecast.py     - loads trained model, produces live forecast
app.py          - Streamlit dashboard
```

## Extending this project

- Add cross-sectional ranking across the ticker universe (which names have the most
  underpriced/overpriced short-term vol right now)
- Layer in options IV once you've collected enough historical snapshots, to move from
  RV forecasting toward an actual RV-IV spread signal
- Add transaction-cost-aware backtest if converting the forecast into a trading signal
  (e.g., short-vol/long-vol overlay, or market-making spread adjustment)
