"""
Append the current forecast for each ticker to the committed forecast log.

This is the piece that makes out-of-sample evidence accumulate. Everything the
dashboard shows about model quality is otherwise either in-sample (predictions
over bars the deployed model was trained on) or a walk-forward estimate from a
single 60-day window. A forecast written here is recorded strictly before its
outcome is known, so once resolved it is the real thing.

Intended to run on a schedule during market hours:

    python log_forecast.py --all

then commit the updated forecast_log.csv. A hosted Streamlit runtime cannot do
this itself — its filesystem is wiped on restart and it has no push
credentials — so the log only grows from wherever this script is run.

Rows are keyed on (ticker, bar_timestamp), so running it more often than the
5-minute bar interval is harmless: repeat calls for a bar already logged are
no-ops.
"""

import argparse
import sys

import config
import data_fetch
import forecast


def log_once(tickers: list[str], skip_when_closed: bool = True) -> int:
    if skip_when_closed and not data_fetch.is_market_open():
        print("Market closed — nothing to log. Use --force to log anyway.")
        return 0

    written = 0
    for ticker in tickers:
        try:
            result = forecast.get_live_forecast(ticker)
            if forecast.append_forecast_log(result):
                written += 1
                print(f"  logged {ticker} @ {result['bar_timestamp']} "
                      f"pred={result['model_forecast_rv']:.6f} "
                      f"har={result['baseline_forecast_rv']:.6f}")
            else:
                print(f"  skipped {ticker} @ {result['bar_timestamp']} (already logged)")
        except Exception as e:
            print(f"  [error] {ticker}: {type(e).__name__}: {e}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="log even when the market is closed (the most recent bar)")
    parser.add_argument("--status", action="store_true",
                        help="print the log scorecard and exit without logging")
    args = parser.parse_args()

    if args.status:
        card = forecast.forecast_log_scorecard(forecast.resolve_forecast_log())
        if card is None:
            print("No resolved forecasts in the log yet.")
            sys.exit(0)
        print(f"logged={card['n_logged']}  resolved={card['n_resolved']}  "
              f"pending={card['n_pending']}  tickers={card['tickers']}")
        print(f"window: {card['first']} -> {card['last']}")
        print(f"MAE  model={card['mae_model']:.6f}  HAR-RV={card['mae_baseline']:.6f}  "
              f"({card['improvement_pct']:+.2f}%)")
        sys.exit(0)

    tickers = config.TICKERS if args.all else [args.ticker or config.DEFAULT_TICKER]
    n = log_once(tickers, skip_when_closed=not args.force)
    print(f"\n{n} new row(s) written to {config.FORECAST_LOG_PATH}")
