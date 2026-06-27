import pandas as pd
import streamlit as st
from data.fetcher import fetch_cbn_exchange_rates
from models.detector import run_full_detection


@st.cache_data(ttl=3600)
def load_and_detect() -> pd.DataFrame:
    df = fetch_cbn_exchange_rates()
    df = run_full_detection(df)
    return df


def format_rate(value: float) -> str:
    return f"₦{value:,.2f}"


def anomaly_summary(df: pd.DataFrame) -> dict:
    anomalies = df[df["is_anomaly"]]
    return {
        "total_points": len(df),
        "anomaly_count": len(anomalies),
        "anomaly_pct": round(len(anomalies) / len(df) * 100, 2),
        "worst_spike": df.loc[df["usd_ngn_rate"].idxmax()],
        "first_anomaly": anomalies.iloc[0] if len(anomalies) > 0 else None,
        "last_anomaly": anomalies.iloc[-1] if len(anomalies) > 0 else None,
    }