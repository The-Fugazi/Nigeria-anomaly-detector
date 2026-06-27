import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data.fetcher import fetch_cbn_exchange_rates
from models.detector import run_full_detection
from utils.helpers import load_and_detect, format_rate, anomaly_summary

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigeria Exchange Rate Anomaly Detector",
    page_icon="🇳🇬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Space+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .metric-card {
        background: #0f1117;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b8fa8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'Space Mono', monospace;
    }
    .metric-value.red { color: #ff4d6d; }
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    .stSelectbox label, .stSlider label { color: #8b8fa8; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Known historical events to label on chart ─────────────────────────────────
KNOWN_EVENTS = {
    "2016-07-01": "CBN devaluation: ₦197 → ₦305",
    "2020-07-01": "COVID-19 crash: ₦360 → ₦461",
    "2023-06-01": "Tinubu float: ₦463 → ₦770",
    "2023-12-01": "Further depreciation: ₦1,490",
    "2024-02-01": "All-time high: ₦1,580",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🇳🇬 Nigeria Exchange Rate — Anomaly Detector")
st.markdown(
    "<p style='color:#8b8fa8; margin-top:-10px;'>CBN USD/NGN historical data · "
    "Isolation Forest + Z-Score + Rolling Deviation</p>",
    unsafe_allow_html=True
)
st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Pulling exchange rate data..."):
    df = load_and_detect()

# ── Add % change column ───────────────────────────────────────────────────────
df["pct_change"] = df["usd_ngn_rate"].pct_change() * 100

# ── Add event labels ──────────────────────────────────────────────────────────
df["event_label"] = df["date"].dt.strftime("%Y-%m-%d").map(KNOWN_EVENTS).fillna("")

summary = anomaly_summary(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    date_range = st.date_input(
        "Date range",
        value=[df["date"].min(), df["date"].max()],
        min_value=df["date"].min(),
        max_value=df["date"].max(),
    )
    show_rolling = st.checkbox("Show rolling mean band", value=True)
    show_zscore = st.checkbox("Show Z-score chart", value=True)

    st.divider()

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown("### >> How it works")
    st.markdown("""
This dashboard runs **three anomaly detection methods** on USD/NGN exchange rate data and flags points caught by any of them.

**Z-Score**
Measures how many standard deviations a data point is from the historical mean. A rate that is unusually high or low compared to the full history gets flagged.

**Isolation Forest**
A machine learning algorithm that isolates anomalies by randomly splitting data. Points that are easy to isolate (few splits needed) are outliers. Works well on non linear patterns.

**Rolling Deviation**
Compares each point to its recent local trend using a moving window. Catches sudden spikes that look normal globally but are sharp relative to recent movement like the 2023 naira float.

A point flagged by **any one method** is marked as an anomaly.
    """)

    st.divider()
    st.info("Isolation Forest assumes ~10% of data points are anomalies.")

# ── Filter by date ────────────────────────────────────────────────────────────
if len(date_range) == 2:
    mask = (df["date"] >= pd.Timestamp(date_range[0])) & (df["date"] <= pd.Timestamp(date_range[1]))
    filtered = df[mask].copy()
else:
    filtered = df.copy()

anomalies = filtered[filtered["is_anomaly"]]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Data Points</div>
        <div class="metric-value">{len(filtered):,}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Anomalies Detected</div>
        <div class="metric-value red">{len(anomalies)}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    latest_rate = filtered["usd_ngn_rate"].iloc[-1]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest Rate</div>
        <div class="metric-value">{format_rate(latest_rate)}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    peak = filtered["usd_ngn_rate"].max()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Historical Peak</div>
        <div class="metric-value red">{format_rate(peak)}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Chart ────────────────────────────────────────────────────────────────
fig = go.Figure()

# Rolling band
if show_rolling and "rolling_upper" in filtered.columns:
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["rolling_upper"],
        fill=None, mode="lines",
        line=dict(color="rgba(99,179,237,0.2)", width=0),
        showlegend=False, name="Upper band"
    ))
    fig.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["rolling_lower"],
        fill="tonexty", mode="lines",
        line=dict(color="rgba(99,179,237,0.2)", width=0),
        fillcolor="rgba(99,179,237,0.07)",
        name="Rolling band"
    ))

# Main rate line
fig.add_trace(go.Scatter(
    x=filtered["date"], y=filtered["usd_ngn_rate"],
    mode="lines", name="USD/NGN Rate",
    line=dict(color="#63b3ed", width=1.5)
))

# Anomaly markers with hover labels
fig.add_trace(go.Scatter(
    x=anomalies["date"],
    y=anomalies["usd_ngn_rate"],
    mode="markers+text",
    name="Anomaly",
    marker=dict(color="#ff4d6d", size=10, symbol="circle",
                line=dict(color="#fff", width=1)),
    text=anomalies["event_label"],
    textposition="top center",
    textfont=dict(size=9, color="#ff4d6d"),
    customdata=anomalies[["pct_change", "event_label", "anomaly_score"]].values,
    hovertemplate=(
        "<b>%{x|%b %Y}</b><br>"
        "Rate: ₦%{y:,.2f}<br>"
        "Change: %{customdata[0]:+.1f}%<br>"
        "Event: %{customdata[1]}<br>"
        "Methods flagged: %{customdata[2]}/3"
        "<extra></extra>"
    )
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=420,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(gridcolor="#1e2130"),
    yaxis=dict(gridcolor="#1e2130", title="₦ per $1 USD"),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ── Z-Score Chart ─────────────────────────────────────────────────────────────
if show_zscore and "zscore" in filtered.columns:
    st.markdown("#### Z-Score Deviation")
    fig2 = go.Figure()
    fig2.add_hline(y=1.5, line_dash="dash", line_color="#ff4d6d",
                   annotation_text="Threshold (+1.5σ)")
    fig2.add_hline(y=-1.5, line_dash="dash", line_color="#ff4d6d",
                   annotation_text="Threshold (-1.5σ)")
    fig2.add_trace(go.Bar(
        x=filtered["date"], y=filtered["zscore"],
        marker_color=["#ff4d6d" if a else "#4a5568" for a in filtered["zscore_anomaly"]],
        name="Z-Score",
        hovertemplate="<b>%{x|%b %Y}</b><br>Z-Score: %{y:.2f}<extra></extra>"
    ))
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Anomaly Table ─────────────────────────────────────────────────────────────
st.markdown("#### Flagged Anomalies")
if len(anomalies) > 0:
    display_df = anomalies[["date", "usd_ngn_rate", "pct_change", "zscore", "anomaly_score", "event_label"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%b %Y")
    display_df["usd_ngn_rate"] = display_df["usd_ngn_rate"].apply(format_rate)
    display_df["pct_change"] = display_df["pct_change"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    display_df["zscore"] = display_df["zscore"].round(2)
    display_df["event_label"] = display_df["event_label"].replace("", "—")
    display_df.columns = ["Date", "USD/NGN Rate", "% Change", "Z-Score", "Methods Flagged (of 3)", "Event"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Download button ───────────────────────────────────────────────────────
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download anomalies as CSV",
        data=csv,
        file_name="ngn_anomalies.csv",
        mime="text/csv"
    )
else:
    st.info("No anomalies detected in the selected date range.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='color:#4a5568; font-size:0.75rem;'>Data: CBN / Public exchange rate records · "
    "Detection: Isolation Forest + Z-Score + Rolling Deviation · Built by Tobi Samuel</p>",
    unsafe_allow_html=True
)