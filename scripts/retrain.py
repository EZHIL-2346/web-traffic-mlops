# scripts/retrain.py
# ------------------
# This script must be able to import the `app` package.
# We add the project root to sys.path so `from app.features import ...` works.

import sys
import os
# Add project root (/app inside the container) to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from dateutil import parser as dateparser

# Paths (relative to project root)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "traffic.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "traffic_model.joblib")

def load_data():
    """
    Load data from CSV and parse timestamps robustly.
    This accepts mixed timestamp formats (with or without seconds).
    It will coerce invalid timestamps to NaT and drop them,
    reporting how many rows were removed.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No data found at {DATA_PATH}. Call /report first.")

    # read CSV
    df = pd.read_csv(DATA_PATH)

    # strip whitespace from timestamp strings (common source of parse errors)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str).str.strip()
    else:
        raise ValueError("CSV missing 'timestamp' column")

    # try parsing with flexible inference (dateutil)
    # coerce invalid parsing to NaT (so we can drop and log them)
    df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True, errors="coerce")

    # Count and drop rows that failed to parse
    num_bad = int(df["timestamp"].isna().sum())
    if num_bad > 0:
        print(f"Warning: {num_bad} row(s) in {DATA_PATH} have invalid timestamps and will be dropped.")
        # (optional) you could save those rows to a separate file for inspection
        bad_rows = df[df["timestamp"].isna()]
        bad_file = os.path.join(os.path.dirname(DATA_PATH), "bad_timestamp_rows.csv")
        bad_rows.to_csv(bad_file, index=False)
        print(f"Bad rows saved to: {bad_file}")

    # drop bad timestamp rows and continue
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_training_features(df):
    # create simple lag + time features for training
    df["lag_1"] = df["page_views"].shift(1)
    df["lag_2"] = df["page_views"].shift(2)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df = df.dropna()

    X = df[["lag_1", "lag_2", "hour", "day_of_week", "month", "is_weekend", "is_festival"]]
    y = df["page_views"]

    return X, y

def train_model():
    df = load_data()
    X, y = build_training_features(df)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    # ensure models folder exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\n=====================================")
    print(" Retraining Completed Successfully")
    print("=====================================")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Total training samples: {len(X)}")
    print("=====================================\n")

if __name__ == "__main__":
    train_model()
