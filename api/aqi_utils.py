"""
AQI utility functions for serverless (standalone version without Flask)
"""
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_features_from_recent(recent_df):
    """Build features from recent AQI data"""
    # Normalize dates for time-based windows (defensive copy)
    recent_df = recent_df.copy()
    recent_df["Date"] = pd.to_datetime(recent_df["Date"], errors="coerce")
    recent_df = recent_df.dropna(subset=["Date"]).sort_values("Date")

    aqi = recent_df["AQI"]
    last_val = aqi.iloc[-1]

    # AQI-lags
    feats = {
        "aqi_lag_1": last_val,
        "aqi_lag_2": aqi.iloc[-2] if len(aqi) >= 2 else last_val,
        "aqi_lag_3": aqi.iloc[-3] if len(aqi) >= 3 else last_val,
        "aqi_lag_7": aqi.iloc[-7] if len(aqi) >= 7 else last_val,
        "aqi_lag_14": aqi.iloc[-14] if len(aqi) >= 14 else (aqi.iloc[-7] if len(aqi) >= 7 else last_val),
    }

    # Time-based rolling means
    end_date = recent_df["Date"].iloc[-1]

    def mean_last_days(days: int):
        start = end_date - pd.Timedelta(days=days)
        window = recent_df.loc[recent_df["Date"] > start, "AQI"]
        return window.mean() if not window.empty else aqi.mean()

    feats.update({
        "aqi_rolling_7": mean_last_days(7),
        "aqi_rolling_14": mean_last_days(14),
        "aqi_rolling_30": mean_last_days(30),
    })

    return feats, recent_df

