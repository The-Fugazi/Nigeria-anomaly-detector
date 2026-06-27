import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def compute_zscore_anomalies(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.DataFrame:
    """
    Flags anomalies using Z-score method.
    Any point beyond `threshold` standard deviations is flagged.
    """
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    df["zscore"] = (df[column] - mean) / std
    df["zscore_anomaly"] = df["zscore"].abs() > threshold
    return df


def compute_isolation_forest_anomalies(df: pd.DataFrame, column: str, contamination: float = 0.1) -> pd.DataFrame:
    """
    Flags anomalies using Isolation Forest.
    contamination = expected proportion of anomalies in the data.
    """
    df = df.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[[column]])

    model = IsolationForest(contamination=contamination, random_state=42)
    df["if_anomaly"] = model.fit_predict(X)
    df["if_anomaly"] = df["if_anomaly"] == -1  # -1 means anomaly in sklearn
    return df


def compute_rolling_anomalies(df: pd.DataFrame, column: str, window: int = 4, threshold: float = 1.2) -> pd.DataFrame:
    """
    Flags anomalies based on rolling mean and std.
    Good for catching sudden spikes relative to recent trend.
    """
    df = df.copy()
    df["rolling_mean"] = df[column].rolling(window=window).mean()
    df["rolling_std"] = df[column].rolling(window=window).std()
    df["rolling_upper"] = df["rolling_mean"] + threshold * df["rolling_std"]
    df["rolling_lower"] = df["rolling_mean"] - threshold * df["rolling_std"]
    df["rolling_anomaly"] = (
        (df[column] > df["rolling_upper"]) | (df[column] < df["rolling_lower"])
    )
    return df


def run_full_detection(df: pd.DataFrame, column: str = "usd_ngn_rate") -> pd.DataFrame:
    """
    Runs all three detection methods and combines results.
    A point is flagged if caught by ANY method.
    """
    df = compute_zscore_anomalies(df, column)
    df = compute_isolation_forest_anomalies(df, column)
    df = compute_rolling_anomalies(df, column)

    df["anomaly_score"] = (
        df["zscore_anomaly"].astype(int) +
        df["if_anomaly"].astype(int) +
        df["rolling_anomaly"].astype(int)
    )
    df["is_anomaly"] = df["anomaly_score"] >= 1
    return df