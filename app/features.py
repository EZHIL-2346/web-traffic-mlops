# app/features.py
import pandas as pd
from dateutil import parser as dateparser
from typing import Optional

# This file provides make_features(timestamp, lag_1, lag_2, is_festival)
# and returns a pandas DataFrame with the exact columns the model expects:
# ["lag_1","lag_2","hour","day_of_week","month","is_weekend","is_festival"]

def _parse_timestamp(ts_str: str):
    # Accept ISO or space-separated timestamps, be tolerant
    if ts_str is None:
        raise ValueError("timestamp is required")
    # If it's already a pandas Timestamp, return it
    if isinstance(ts_str, pd.Timestamp):
        return ts_str
    # If it is a datetime.datetime, convert to pandas Timestamp
    try:
        dt = dateparser.parse(ts_str)
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {ts_str}") from e
    # Convert to pandas Timestamp so .dayofweek is available
    return pd.Timestamp(dt)

def make_features(timestamp: str, lag_1: Optional[float] = None, lag_2: Optional[float] = None, is_festival: Optional[int] = 0):
    """
    Build the exact feature DataFrame expected by the model.
    - timestamp: ISO string like "2025-12-07T15:00:00"
    - lag_1, lag_2: optional numeric values (if None, left as NaN)
    - is_festival: 0 or 1
    Returns: pd.DataFrame with one row and columns:
      ["lag_1","lag_2","hour","day_of_week","month","is_weekend","is_festival"]
    """
    ts = _parse_timestamp(timestamp)
    hour = int(ts.hour)
    # now ts is a pandas.Timestamp, so .dayofweek exists
    day_of_week = int(ts.dayofweek)  # 0=Mon .. 6=Sun
    month = int(ts.month)
    is_weekend = 1 if day_of_week >= 5 else 0

    row = {
        "lag_1": float(lag_1) if lag_1 is not None else None,
        "lag_2": float(lag_2) if lag_2 is not None else None,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "is_festival": int(is_festival) if is_festival is not None else 0
    }

    df = pd.DataFrame([row], columns=["lag_1","lag_2","hour","day_of_week","month","is_weekend","is_festival"])
    return df
