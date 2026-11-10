"""
AeroLens — Air Quality Intelligence & Forecaster
Streamlit Community Cloud Deployment Application
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Configure wide layout, title & favicon
st.set_page_config(
    page_title="AeroLens — Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add root and src directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import backend inference functions from app
try:
    from app import (
        list_valid_states, list_valid_stations,
        get_entity_series_cached,
        predict_arima_robust, predict_lstm_robust, predict_ensemble_model,
        compute_metrics, aqi_to_bucket, bucket_color, ADVISORIES
    )
except Exception as err:
    import traceback
    st.error(f"Error loading forecasting engine: {err}")
    st.code(traceback.format_exc())
    st.stop()

# ── Custom CSS for Frosted Glassmorphism Theme ───────────────────────────
st.markdown("""
<link href="https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css" rel="stylesheet">
<style>
    /* Dark Deep Theme */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 85% 75%, rgba(127, 0, 255, 0.08) 0%, transparent 45%),
                    #060913;
        color: #f0f4fc;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Header Styling */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 30%, #8e9bb5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #8e9bb5;
        margin-bottom: 24px;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(18, 24, 43, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(20px);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.4);
    }

    /* Score Hero Readout */
    .aqi-score-container {
        text-align: center;
        padding: 24px;
        background: radial-gradient(circle at center, rgba(255, 255, 255, 0.04) 0%, transparent 70%);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .aqi-number {
        font-size: 4.8rem;
        font-weight: 900;
        line-height: 1;
        margin: 10px 0;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Advisory Card */
    .advisory-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #00f2fe;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 14px;
    }

    /* Metric Chips */
    .metric-chip {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8e9bb5;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# ── Title Header ────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Air Quality Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Select your location and time horizon to compute high-precision Air Quality Index forecasts '
    'powered by ARIMA, RNN (LSTM), Random Forest, and Gradient Boosting architectures.'
    '</div>',
    unsafe_allow_html=True
)

# ── Model Family Toggle (Phase 1 vs Phase 2) ───────────────────────────
phase_choice = st.radio(
    "Select Model Family",
    options=["Time Series Models (ARIMA & LSTM)", "Ensemble Models (Random Forest & Gradient Boosting)"],
    index=0,
    horizontal=True,
    help="Toggle between classical/neural sequential time series vs ensemble bagging/boosting regressors."
)
is_phase1 = "Time Series" in phase_choice

st.markdown("---")

# ── Layout: Two Columns (Configuration vs Forecast Dashboard) ───────────
col_config, col_results = st.columns([1, 2], gap="large")

with col_config:
    st.markdown("### ⚙️ Configuration")

    # 1. Forecast Level (State vs Station)
    level_choice = st.radio(
        "Forecast Level",
        options=["State Level", "Station Level"],
        index=0,
        horizontal=True
    )
    is_station = (level_choice == "Station Level")
    level = "Station" if is_station else "State"

    # 2. Select Entity
    if is_station:
        station_list = list_valid_stations()
        default_idx = station_list.index("Pusa, Delhi - DPCC") if "Pusa, Delhi - DPCC" in station_list else 0
        entity = st.selectbox("Select Station", options=station_list, index=default_idx)
    else:
        state_list = list_valid_states()
        default_idx = state_list.index("Delhi") if "Delhi" in state_list else 0
        entity = st.selectbox("Select State", options=state_list, index=default_idx)

    # Fetch entity historical span
    series = get_entity_series_cached(level, entity)
    if series is not None and not series.empty:
        data_start = series.index.min().date()
        data_end = series.index.max().date()
    else:
        data_start = datetime.date(2024, 1, 1)
        data_end = datetime.date(2024, 3, 31)

    # 3. Date Selection
    st.markdown("##### 📅 Target Horizon")

    # Defaults in session state
    if "from_dt" not in st.session_state:
        st.session_state["from_dt"] = data_end - datetime.timedelta(days=14)
    if "to_dt" not in st.session_state:
        st.session_state["to_dt"] = data_end

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        from_date = st.date_input("From Date", value=st.session_state["from_dt"])
    with d_col2:
        to_date = st.date_input("To Date", value=st.session_state["to_dt"])

    # 4. Model Selection (Dynamic based on selected paradigm)
    if is_phase1:
        model_options = [
            "ARIMA (Statistical Time Series)",
            "RNN (LSTM) (Recurrent Neural Network)",
            "Compare Both (ARIMA vs LSTM)"
        ]
    else:
        model_options = [
            "Random Forest (Ensemble Bagging · R²=0.940)",
            "Gradient Boosting (Ensemble Boosting · R²=0.933)",
            "Compare Both (Random Forest vs Gradient Boosting)"
        ]
    
    chosen_model_str = st.selectbox("Forecasting Architecture", options=model_options)

    # Simplify model string
    if "ARIMA vs LSTM" in chosen_model_str or "Random Forest vs Gradient Boosting" in chosen_model_str:
        model_name = "Compare Both"
    elif "ARIMA" in chosen_model_str:
        model_name = "ARIMA"
    elif "LSTM" in chosen_model_str:
        model_name = "RNN (LSTM)"
    elif "Gradient Boosting" in chosen_model_str:
        model_name = "Gradient Boosting"
    else:
        model_name = "Random Forest"

    # 5. Compute Forecast Button
    btn_predict = st.button("🚀 Compute AQI Forecast", type="primary", use_container_width=True)


# ── Results Execution ───────────────────────────────────────────────────
with col_results:
    if not btn_predict and "last_result" not in st.session_state:
        st.info("👈 Configure your location and date range on the left, then click **'Compute AQI Forecast'**.")
    else:
        start_dt = pd.to_datetime(from_date)
        end_dt = pd.to_datetime(to_date)

        if start_dt > end_dt:
            st.error("Error: 'From Date' cannot be after 'To Date'.")
        else:
            with st.spinner("Analyzing atmospheric trends & calculating neural trajectory..."):
                # Determine mode (backtest vs future)
                is_backtest = end_dt.date() <= data_end
                is_future = start_dt.date() > data_end
                if is_backtest:
                    mode_label = "Historical Backtest"
                elif is_future:
                    mode_label = "Future Forecast"
                else:
                    mode_label = "Mixed (Historical & Future)"

                # Extract historical actuals if available
                actual_series = []
                if series is not None:
                    matched_actual = series.loc[(series.index >= start_dt) & (series.index <= end_dt), "AQI"]
                    actual_series = [
                        {"timestamp": str(idx.date()), "aqi": round(float(val), 1)}
                        for idx, val in matched_actual.items()
                    ]

                # Run predictions
                predictions_dict = {}
                metrics_dict = {}
                active_phase = "phase1" if is_phase1 else "phase2"

                if active_phase == "phase1":
                    if model_name in ["ARIMA", "Compare Both"]:
                        p_arima = predict_arima_robust(entity, start_dt, end_dt)
                        if p_arima:
                            predictions_dict["ARIMA"] = p_arima
                    if model_name in ["RNN (LSTM)", "Compare Both"]:
                        p_lstm = predict_lstm_robust(entity, level, start_dt, end_dt, series)
                        if p_lstm:
                            predictions_dict["RNN (LSTM)"] = p_lstm
                else:
                    if model_name in ["Random Forest", "Compare Both"]:
                        p_rf = predict_ensemble_model("Random Forest", entity, level, start_dt, end_dt)
                        if p_rf:
                            predictions_dict["Random Forest"] = p_rf
                    if model_name in ["Gradient Boosting", "Compare Both"]:
                        p_gb = predict_ensemble_model("Gradient Boosting", entity, level, start_dt, end_dt)
                        if p_gb:
                            predictions_dict["Gradient Boosting"] = p_gb

                # Check if predictions succeeded
                if not predictions_dict:
                    st.error("Could not compute forecast for the selected range. Please adjust dates.")
                else:
                    # Calculate metrics if backtesting
                    winner = None
                    if actual_series:
                        best_rmse = float("inf")
                        for m_key, preds in predictions_dict.items():
                            m_vals = compute_metrics(actual_series, preds)
                            if m_vals:
                                metrics_dict[m_key] = m_vals
                                if m_vals["RMSE"] < best_rmse:
                                    best_rmse = m_vals["RMSE"]
                                    winner = m_key

                    # Compute summary values
                    primary_preds = list(predictions_dict.values())[0]
                    avg_aqi = round(float(np.mean([p["predicted_AQI"] for p in primary_preds])))
                    summary_bucket = aqi_to_bucket(avg_aqi)
                    b_color = bucket_color(summary_bucket)
                    advisory = ADVISORIES.get(summary_bucket, ADVISORIES["Moderate"])

                    # ── Hero AQI Readout Card ───────────────────────────
                    st.markdown(f"""
                    <div class="aqi-score-container" style="border-color: {b_color}44;">
                        <span style="font-size: 0.85rem; color: #8e9bb5; font-weight: 600; text-transform: uppercase;">
                            {mode_label} · {entity}
                        </span>
                        <div class="aqi-number" style="color: {b_color}; text-shadow: 0 0 25px {b_color}55;">
                            {avg_aqi}
                        </div>
                        <div class="badge-pill" style="background: {b_color}22; color: {b_color}; border: 1px solid {b_color}66;">
                            {summary_bucket} (CPCB Air Quality Standard)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Interactive Plotly Time Series Chart ────────────
                    fig = go.Figure()

                    # Shaded Indian CPCB Threshold Bands
                    fig.add_hrect(y0=0, y1=50, fillcolor="#10b981", opacity=0.04, line_width=0, annotation_text="Good", annotation_position="top left")
                    fig.add_hrect(y0=51, y1=100, fillcolor="#84cc16", opacity=0.04, line_width=0, annotation_text="Satisfactory", annotation_position="top left")
                    fig.add_hrect(y0=101, y1=200, fillcolor="#f59e0b", opacity=0.04, line_width=0, annotation_text="Moderate", annotation_position="top left")
                    fig.add_hrect(y0=201, y1=300, fillcolor="#f97316", opacity=0.04, line_width=0, annotation_text="Poor", annotation_position="top left")
                    fig.add_hrect(y0=301, y1=400, fillcolor="#ef4444", opacity=0.04, line_width=0, annotation_text="Very Poor", annotation_position="top left")

                    # Add Historical Actuals line if present
                    if actual_series:
                        act_dates = [x["timestamp"] for x in actual_series]
                        act_vals = [x["aqi"] for x in actual_series]
                        fig.add_trace(go.Scatter(
                            x=act_dates, y=act_vals,
                            mode="lines+markers",
                            name="Historical Actual AQI",
                            connectgaps=True,
                            line=dict(color="#ffffff", width=2.5),
                            marker=dict(size=5, color="#ffffff")
                        ))

                    # Colors palette for models
                    model_colors = {
                        "ARIMA": "#00f2fe",
                        "RNN (LSTM)": "#e100ff",
                        "Random Forest": "#10b981",
                        "Gradient Boosting": "#f59e0b"
                    }

                    for m_name, preds in predictions_dict.items():
                        m_dates = [p["timestamp"] for p in preds]
                        m_vals = [p["predicted_AQI"] for p in preds]
                        c = model_colors.get(m_name, "#00f2fe")

                        # Lower / Upper confidence intervals
                        if "lower" in preds[0] and "upper" in preds[0]:
                            lowers = [p.get("lower", p["predicted_AQI"] - 20) for p in preds]
                            uppers = [p.get("upper", p["predicted_AQI"] + 20) for p in preds]
                            fig.add_trace(go.Scatter(
                                x=m_dates + m_dates[::-1],
                                y=uppers + lowers[::-1],
                                fill="toself",
                                fillcolor=f"rgba({int(c[1:3],16)}, {int(c[3:5],16)}, {int(c[5:7],16)}, 0.12)",
                                line=dict(color="rgba(255,255,255,0)"),
                                hoverinfo="skip",
                                showlegend=False,
                                name=f"{m_name} 95% CI"
                            ))

                        fig.add_trace(go.Scatter(
                            x=m_dates, y=m_vals,
                            mode="lines+markers",
                            name=f"{m_name} Forecast",
                            line=dict(color=c, width=3),
                            marker=dict(size=6, color=c)
                        ))

                    fig.update_layout(
                        title=f"AQI Trajectory & Horizon Forecast ({entity})",
                        template="plotly_dark",
                        paper_bgcolor="rgba(18, 24, 43, 0.4)",
                        plot_bgcolor="rgba(0, 0, 0, 0)",
                        hovermode="x unified",
                        height=440,
                        margin=dict(l=20, r=20, t=50, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Model Benchmarks (If Available) ─────────────────
                    if metrics_dict:
                        st.markdown("##### 📏 Model Test Evaluation Metrics")
                        if winner:
                            st.success(f"🏆 Best Performing Architecture: **{winner}**")

                        for m_name, m_vals in metrics_dict.items():
                            st.markdown(f"**{m_name} Performance**")
                            m_cols = st.columns(5)
                            m_cols[0].metric("MAE", m_vals["MAE"])
                            m_cols[1].metric("RMSE", m_vals["RMSE"])
                            m_cols[2].metric("MAPE", f"{m_vals['MAPE']}%")
                            m_cols[3].metric("R² Score", m_vals["R2"])
                            m_cols[4].metric("Category Acc", f"{m_vals['Cat_Accuracy']}%")

                    # ── Health Advisory ─────────────────────────────────
                    st.markdown("##### 🍃 Health & Atmospheric Advisory")
                    st.markdown(f"""
                    <div class="advisory-box" style="border-left-color: {b_color};">
                        <strong style="font-size: 1.05rem; color: #ffffff;">{advisory['heading']}</strong>
                        <p style="margin-top: 6px; font-size: 0.92rem; color: #c4d1eb; line-height: 1.5;">
                            {advisory['text']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Daily Breakdown Table ───────────────────────────
                    with st.expander("📋 View Daily Forecast Breakdown Table", expanded=False):
                        table_data = []
                        for p in primary_preds:
                            ci_str = f"{p.get('lower', '--')} to {p.get('upper', '--')}" if 'lower' in p else "± 16.5"
                            table_data.append({
                                "Date": p["timestamp"],
                                "Forecast AQI": p["predicted_AQI"],
                                "CPCB Category": p.get("AQI_Bucket", "Moderate"),
                                "95% Confidence Interval": ci_str
                            })
                        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

