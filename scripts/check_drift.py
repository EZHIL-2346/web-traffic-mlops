# scripts/check_drift.py
# ============================================================
# DRIFT DETECTION ENGINE FOR WEB TRAFFIC FORECAST MODEL
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from app.features import make_features
from sklearn.metrics import mean_absolute_error

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "traffic.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "traffic_model.joblib")
DRIFT_REPORT_PATH = os.path.join(ROOT_DIR, "data", "drift_metrics.json")

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
def load_recent_data(window=20):
    """
    Load recent data for drift detection robustly.
    Accepts mixed timestamp formats (with or without seconds).
    Coerces invalid timestamps to NaT, saves them for inspection, drops them,
    and returns the last `window` valid rows.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("No data found. You must use /report to add real data.")
    
    df = pd.read_csv(DATA_PATH)

    # Ensure timestamp column exists and strip whitespace
    if "timestamp" not in df.columns:
        raise ValueError("CSV missing 'timestamp' column")
    df["timestamp"] = df["timestamp"].astype(str).str.strip()

    # Parse timestamps flexibly; invalid parses become NaT
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Count and save bad rows if any
    num_bad = int(df["timestamp"].isna().sum())
    if num_bad > 0:
        print(f"Warning: {num_bad} row(s) in {DATA_PATH} have invalid timestamps and will be dropped.")
        bad_rows = df[df["timestamp"].isna()]
        bad_file = os.path.join(os.path.dirname(DATA_PATH), "bad_timestamp_rows_for_drift.csv")
        bad_rows.to_csv(bad_file, index=False)
        print(f"Bad rows saved to: {bad_file}")

    # drop bad timestamp rows and continue
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ensure we have enough rows after dropping bad ones
    if len(df) < window + 2:
        raise ValueError(f"Not enough valid data for drift detection. Need {window+2}, got {len(df)}")

    # return the last `window` rows
    return df.tail(window).copy()


# -------------------------------------------------------------
# BUILD FEATURES USED DURING TRAINING
# -------------------------------------------------------------
def build_features(df):
    df["lag_1"] = df["page_views"].shift(1)
    df["lag_2"] = df["page_views"].shift(2)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df = df.dropna()
    return df

# -------------------------------------------------------------
# DRIFT CHECK
# -------------------------------------------------------------
def detect_drift():
    df = load_recent_data()
    df = build_features(df)

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No model found. Train model first.")

    model = joblib.load(MODEL_PATH)

    # Features used for training
    feature_cols = ["lag_1", "lag_2", "hour", "day_of_week", "month", "is_weekend", "is_festival"]
    X = df[feature_cols]
    y_true = df["page_views"]

    # Predictions
    y_pred = model.predict(X)

    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)

    # Feature drift: compare mean + std of training-like features
    drift_scores = {}
    for col in feature_cols:
        drift_scores[col] = float(abs(df[col].mean() - df[col].iloc[0]))  # simple drift metric

    # Drift decision
    if mae < 50:
        drift_status = "NO_DRIFT"
    elif mae < 120:
        drift_status = "MILD_DRIFT"
    else:
        drift_status = "SEVERE_DRIFT"

    # Save drift report
    report = {
        "timestamp": datetime.now().isoformat(),
        "mae": float(mae),
        "drift_scores": drift_scores,
        "decision": drift_status
    }

    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)

    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    return report


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":
    report = detect_drift()
    print("\n================ DRIFT REPORT ================")
    print(json.dumps(report, indent=4))
    print("==============================================\n")
