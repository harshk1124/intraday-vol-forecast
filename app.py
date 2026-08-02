"""
Streamlit dashboard for live intraday volatility forecasting.

Run with:  streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go

import config
import forecast
import data_fetch

st.set_page_config(page_title="Intraday Vol Forecast", layout="wide")

# --- Auto-refresh every 60 seconds ---
REFRESH_SECONDS = 60

# One full 09:30-16:00 session. Kept at a session's width so the chart normally
# shows a single continuous day rather than straddling an overnight gap.
CHART_LOOKBACK_MINUTES = 390
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")
except ImportError:
    st.sidebar.warning("Install `streamlit-autorefresh` for auto-refresh (falling back to manual).")


# Both calls below hit yfinance independently, so an uncached rerun costs two
# downloads — every 60s, per viewer. Yahoo rate-limits by source IP and a hosted
# deployment shares one across all viewers, so that pattern earns 429s quickly.
# Caching on ttl collapses it to one fetch per interval for the whole app.
@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_forecast(ticker: str) -> dict:
    return forecast.get_live_forecast(ticker)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_history(ticker: str, lookback_minutes: int = 240):
    return forecast.get_forecast_history(ticker, lookback_minutes=lookback_minutes)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_track(ticker: str, lookback_minutes: int = 390):
    return forecast.get_prediction_track(ticker, lookback_minutes=lookback_minutes)


# Resolving the log re-downloads 60 days of history per ticker in it, so this is
# far too expensive to repeat on the 60s refresh. The log only grows when
# log_forecast.py runs, so a long TTL costs nothing in freshness.
@st.cache_data(ttl=900, show_spinner=False)
def load_log_scorecard():
    return forecast.forecast_log_scorecard(forecast.resolve_forecast_log())


st.title("📈 Intraday Realized Volatility Forecast")
st.caption("ML-based short-horizon vol forecasting vs. HAR-RV baseline — free data pipeline (Alpaca / yfinance)")

ticker = st.sidebar.selectbox("Ticker", config.TICKERS, index=config.TICKERS.index(config.DEFAULT_TICKER))
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Data source:** {'Alpaca (real-time)' if config.USE_ALPACA else 'yfinance (delayed ~15-20min)'}"
)
st.sidebar.markdown(f"**Refresh interval:** {REFRESH_SECONDS}s")

market_open = data_fetch.is_market_open()
status_color = "🟢" if market_open else "🔴"
st.sidebar.markdown(f"**Market status:** {status_color} {'Open' if market_open else 'Closed'}")

if not market_open:
    st.info(
        "Market is currently closed. Forecasts below reflect the most recent available bars "
        "and won't update until the next session."
    )

try:
    result = load_forecast(ticker)
    forecast.save_latest_forecast(result)

    # Best-effort: on a hosted runtime this writes to a filesystem that is wiped
    # on restart, so the log only durably grows where log_forecast.py is run and
    # the CSV is committed. Failing here must never take the dashboard down.
    try:
        forecast.append_forecast_log(result)
    except Exception:
        pass

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Price", f"${result['last_price']:.2f}")
    col2.metric("Current RV (short)", f"{result['current_rv_short']:.4f}")

    delta = result["model_forecast_rv"] - result["baseline_forecast_rv"]
    col3.metric(
        "ML Forecast (next hr RV)",
        f"{result['model_forecast_rv']:.4f}",
        delta=f"{delta:+.4f} vs HAR-RV",
    )
    col4.metric("HAR-RV Baseline", f"{result['baseline_forecast_rv']:.4f}")

    st.markdown("---")

    # --- Chart: price + rolling realized vol ---
    chart_df = load_history(ticker, lookback_minutes=CHART_LOOKBACK_MINUTES)

    # Plot on a tz-naive ET index. The rangebreak hour bounds below are read in
    # the axis's own coordinates, so a tz-aware index would shift them.
    chart_df = chart_df.copy()
    chart_df.index = chart_df.index.tz_localize(None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["close"], name="Price", yaxis="y1",
        line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["rv_short"], name="Realized Vol (short window)", yaxis="y2",
        line=dict(color="#d62728", dash="dot"),
    ))
    fig.update_layout(
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Realized Vol", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    # Collapse the hours the market is shut, so a multi-session window doesn't
    # spend most of its width on empty overnight time.
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[16, 9.5], pattern="hour"),
    ])
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Forecast vs. what actually realized")

    track = load_track(ticker, lookback_minutes=CHART_LOOKBACK_MINUTES)
    track_plot = track.copy()
    track_plot.index = track_plot.index.tz_localize(None)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=track_plot.index, y=track_plot["realized_rv"], name="Realized (next hr)",
        line=dict(color="#2ca02c", width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=track_plot.index, y=track_plot["predicted_rv"], name="ML forecast",
        line=dict(color="#1f77b4", dash="dash"),
    ))
    fig2.add_trace(go.Scatter(
        x=track_plot.index, y=track_plot["baseline_rv"], name="HAR-RV baseline",
        line=dict(color="#d62728", dash="dot"),
    ))
    fig2.update_layout(
        yaxis=dict(title="Realized vol (next 60 min)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig2.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[16, 9.5], pattern="hour"),
    ])
    st.plotly_chart(fig2, width="stretch")
    st.caption(
        "Each point is plotted at the bar the forecast was made from, so all three series "
        f"describe the same forward hour. The last {config.FORECAST_HORIZON} bars have no "
        "realized value yet — those windows are still open."
    )

    resolved = track.dropna()
    if not resolved.empty:
        mae_model = float((resolved["predicted_rv"] - resolved["realized_rv"]).abs().mean())
        mae_base = float((resolved["baseline_rv"] - resolved["realized_rv"]).abs().mean())
        w1, w2, w3 = st.columns(3)
        w1.metric("Window MAE — model", f"{mae_model:.5f}")
        w2.metric("Window MAE — HAR-RV", f"{mae_base:.5f}")
        w3.metric(
            "Window improvement",
            f"{(mae_base - mae_model) / mae_base * 100:+.1f}%" if mae_base else "n/a",
        )

    metrics = forecast.load_metrics(ticker)
    if metrics and metrics.get("dm_p_value") is not None:
        verdict = (
            "significant at 5%" if metrics["dm_p_value"] < 0.05
            else "**not** significant at 5%"
        )
        st.warning(
            "Those window figures are **in-sample** — the deployed model was fit on all "
            "available history, including these bars, so they overstate real skill. The "
            f"out-of-sample result from walk-forward validation is **{metrics['improvement_pct']:+.2f}%** "
            f"vs HAR-RV, winning {metrics['folds_won']}/{metrics['n_folds']} folds "
            f"(Diebold-Mariano p={metrics['dm_p_value']:.3f}, {verdict})."
        )

    st.markdown("---")
    st.subheader("Live forecast log — genuinely out-of-sample")

    card = load_log_scorecard()
    if card is None:
        st.info(
            "No resolved forecasts logged yet. Run `python log_forecast.py --all` during "
            "market hours and commit `forecast_log.csv`; each row is written before its "
            "outcome exists, so once resolved it is out-of-sample by construction — "
            "unlike the in-sample window above or the single-window walk-forward estimate."
        )
    else:
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Forecasts resolved", f"{card['n_resolved']}")
        l2.metric("Log MAE — model", f"{card['mae_model']:.5f}")
        l3.metric("Log MAE — HAR-RV", f"{card['mae_baseline']:.5f}")
        l4.metric(
            "Improvement",
            f"{card['improvement_pct']:+.1f}%" if card["improvement_pct"] is not None else "n/a",
        )
        st.caption(
            f"{card['n_resolved']} resolved of {card['n_logged']} logged across "
            f"{card['tickers']} ticker(s), {card['first']} → {card['last']}. "
            f"{card['n_pending']} still pending — a forecast made in the final "
            f"{config.FORECAST_HORIZON} bars of a session never resolves, since its "
            "window would cross the close."
        )

    st.caption(
        f"Last updated: {result['timestamp']} UTC · Model forecasts realized vol over the next "
        f"{config.FORECAST_HORIZON * config.BAR_TIMEFRAME_MINUTES} minutes."
    )

except FileNotFoundError as e:
    st.error(str(e))
    st.info(f"Train a model first: `python train.py --ticker {ticker}`")
except Exception as e:
    st.error(f"Error fetching live forecast: {e}")

with st.expander("ℹ️ Methodology"):
    st.markdown(f"""
    - **Target:** realized volatility over the next {config.FORECAST_HORIZON} bars
      ({config.FORECAST_HORIZON * config.BAR_TIMEFRAME_MINUTES} min horizon), never
      spanning a session boundary
    - **Baseline:** HAR-RV (Corsi 2009) — an OLS regression on {config.RV_WINDOWS}-bar
      realized vol, re-fit on each training fold
    - **Model:** XGBoost regressor on RV features + Parkinson range vol + volume z-score + time-of-day
    - **Overnight gaps** are masked out; the close-to-open return is ~7x a typical
      5-min return and otherwise dominates every rolling window
    - **Validation:** walk-forward (expanding window), with a Diebold-Mariano test
      (Newey-West corrected for overlapping forecasts) on the MAE differential
    - **Data:** {'Alpaca free real-time feed' if config.USE_ALPACA else 'yfinance (free, ~15-20min delayed)'}

    Measured edge over HAR-RV is small and not uniform: negligible on SPY/QQQ,
    meaningful on the sector ETFs. See the README for per-ticker numbers.
    """)
