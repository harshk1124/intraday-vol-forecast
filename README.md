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
  The baseline here is an actual HAR-RV *regression*, re-fit on each training fold. A
  fixed average of RV windows is not a HAR-RV and is far too easy to beat: realized vol
  scales with sqrt(window), so averaging 5- through 60-bar RV to predict 12-bar RV
  overshoots the target by ~30% before any modelling happens.

## Results

Walk-forward MAE vs. the HAR-RV baseline, 60 days of 5-min bars, 5 expanding-window
folds. p-values are Diebold-Mariano, Newey-West corrected for overlapping forecasts.

| Ticker | Improvement | Folds won | DM p-value | Significant? |
|--------|------------:|:---------:|-----------:|--------------|
| SPY    |      +0.62% |    4/5    |      0.849 | no           |
| QQQ    |      +1.85% |    3/5    |      0.511 | no           |
| XLF    |     +12.51% |    5/5    |      0.000 | yes          |
| XLE    |      +8.06% |    4/5    |      0.005 | yes          |
| XLK    |      +9.26% |    5/5    |      0.001 | yes          |

The pattern is the interesting part: **no measurable edge on SPY and QQQ**, the two most
liquid and most heavily arbitraged names, and a real edge on the sector ETFs. That is
what you would expect if the effect is genuine rather than an artifact — vol dynamics in
less-trafficked instruments are less efficiently priced. A model that "beat" HAR-RV
everywhere by a wide margin would be much more likely to be measuring its own bugs.

Reproduce with `python train.py --all`. Numbers will shift as the 60-day window rolls.

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
features.py     - feature engineering, forward RV target, HAR-RV baseline
train.py        - walk-forward training, baseline comparison, DM test
forecast.py     - loads trained model, produces live forecast
app.py          - Streamlit dashboard
```

## Known limitations

- `is_market_open()` uses a fixed 09:30-16:00 ET window and ignores holidays and
  half-days; swap in `pandas_market_calendars` if that matters to you.
- 60 days of 5-min bars is the yfinance intraday limit, so every result covers a
  single volatility regime. Treat the numbers as indicative, not as evidence the
  edge survives a regime change.
- MAE on RV levels underweights exactly the vol spikes you most want to forecast.
  QLIKE is the standard alternative and is worth reporting alongside.
- No transaction costs are modelled anywhere — this is a forecast, not a strategy.

## Extending this project

- Add cross-sectional ranking across the ticker universe (which names have the most
  underpriced/overpriced short-term vol right now)
- Layer in options IV once you've collected enough historical snapshots, to move from
  RV forecasting toward an actual RV-IV spread signal
- Add transaction-cost-aware backtest if converting the forecast into a trading signal
  (e.g., short-vol/long-vol overlay, or market-making spread adjustment)
