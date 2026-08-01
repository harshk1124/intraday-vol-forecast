"""
Streamlit dashboard for live intraday volatility forecasting.

Run with:  streamlit run app.py
"""

import time
import streamlit as st
import plotly.graph_objects as go

import config
import forecast
import data_fetch

st.set_page_config(page_title="Intraday Vol Forecast", layout="wide")

# --- Auto-refresh every 60 seconds ---
REFRESH_SECONDS = 60
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")
except ImportError:
    st.sidebar.warning("Install `streamlit-autorefresh` for auto-refresh (falling back to manual).")

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
    result = forecast.get_live_forecast(ticker)
    forecast.save_latest_forecast(result)

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
    chart_df = forecast.get_forecast_history(ticker, lookback_minutes=240)

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
    st.plotly_chart(fig, use_container_width=True)

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
      ({config.FORECAST_HORIZON * config.BAR_TIMEFRAME_MINUTES} min horizon)
    - **Baseline:** HAR-RV style average of {config.RV_WINDOWS}-bar realized vol windows
    - **Model:** XGBoost regressor on RV features + Parkinson range vol + volume z-score + time-of-day
    - **Validation:** walk-forward (expanding window), compared against HAR-RV baseline MAE
    - **Data:** {'Alpaca free real-time feed' if config.USE_ALPACA else 'yfinance (free, ~15-20min delayed)'}
    """)
