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

The universe is every name in `config.TICKERS` — four broad-market indices and the
complete SPDR sector family. All fifteen are reported, including the four where the
model fails to beat the baseline. Keeping only the winners would be selecting on the
outcome.

| Ticker | Type   | Improvement | Folds won | DM p-value | Significant? |
|--------|--------|------------:|:---------:|-----------:|--------------|
| XLV    | sector |     +16.87% |    5/5    |      0.000 | yes          |
| XLU    | sector |     +14.81% |    5/5    |      0.000 | yes          |
| XLF    | sector |     +12.51% |    5/5    |      0.000 | yes          |
| XLP    | sector |     +11.30% |    5/5    |      0.000 | yes          |
| XLB    | sector |      +9.47% |    4/5    |      0.001 | yes          |
| XLK    | sector |      +9.26% |    5/5    |      0.001 | yes          |
| XLI    | sector |      +8.91% |    5/5    |      0.000 | yes          |
| XLE    | sector |      +8.06% |    4/5    |      0.005 | yes          |
| XLRE   | sector |      +7.34% |    4/5    |      0.004 | yes          |
| DIA    | broad  |      +5.61% |    4/5    |      0.014 | yes          |
| XLC    | sector |      +4.84% |    3/5    |      0.063 | no           |
| IWM    | broad  |      +4.56% |    4/5    |      0.197 | no           |
| QQQ    | broad  |      +1.85% |    3/5    |      0.511 | no           |
| XLY    | sector |      +1.41% |    1/5    |      0.600 | no           |
| SPY    | broad  |      +0.62% |    4/5    |      0.849 | no           |

Ten of fifteen are significant at 5%. Testing fifteen names would produce roughly one
false positive by chance; seven land at p ≤ 0.005, so the result is not multiple-testing
noise. It is also not uniform, which is the reassuring part — a model that beat HAR-RV
everywhere by a wide margin would more likely be measuring its own bugs.

**On what drives the split:** the three broad indices that fail (SPY, QQQ, IWM) are among
the most heavily arbitraged instruments in existence, and it is tempting to conclude that
liquidity explains the pattern. It does not, at least not on its own. Widening the test
beyond this universe finds NVDA — which trades more dollar volume per bar than any sector
ETF here — at +7.75%, p=0.004, while XBI and XLY sit at mid-liquidity with no edge at all.
Rank the full set by dollar volume and no monotone relationship appears. The defensible
claim is narrower: **broad-index vol is the hardest to forecast, sector and thematic ETFs
are easier, and dollar volume alone does not explain why.**

Every number here comes from a single 60-day window in one volatility regime.

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
forecast.py     - loads trained model, produces live forecast, forecast log
log_forecast.py - appends live forecasts to the committed log (run on a schedule)
app.py          - Streamlit dashboard
```

## Live forecast log

Two of the three accuracy figures the dashboard shows are weaker evidence than they
look. The forecast-vs-realized chart is **in-sample** — the deployed model was fit on
all available history, those bars included — and the walk-forward result, while
genuinely out-of-sample, comes from one 60-day window.

`forecast_log.csv` fixes that by accumulating forecasts recorded *before* their outcome
existed:

```bash
python log_forecast.py --all      # append current forecasts for every ticker
python log_forecast.py --status   # scorecard: MAE of model vs HAR-RV on resolved rows
```

Run it on a schedule during market hours and commit the CSV. Rows are keyed on
`(ticker, bar_timestamp)`, so running more often than the 5-minute bar interval is
harmless — repeats are no-ops. Resolution re-derives the realized outcome from price
history rather than trusting anything stored at prediction time, so the outcome always
comes from data the model never saw.

The log is committed rather than kept under `data/` deliberately: Streamlit Community
Cloud wipes its filesystem on restart and has no push credentials, so the deployed app
can only *read* this history — it cannot grow it. The log grows wherever you run
`log_forecast.py`.

One caveat: a forecast made in the final 12 bars of a session never resolves, because
its forward window would cross the close and the target is session-bounded by design.
Those rows stay pending permanently.

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
