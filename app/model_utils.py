# app/model_utils.py
import os
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "traffic_model.joblib")

# Features model expects (in training order)
FEATURE_COLS = ["lag_1","lag_2","hour","day_of_week","month","is_weekend","is_festival"]

def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    return joblib.load(MODEL_PATH)

def _ensure_columns(X: pd.DataFrame):
    """
    Ensure X has FEATURE_COLS in the right order.
    If a column is missing, fill with safe default (0 or NaN for lags).
    """
    X = X.copy()
    for col in FEATURE_COLS:
        if col not in X.columns:
            # lag features -> NaN; categorical/time features -> 0
            if col in ["lag_1", "lag_2"]:
                X[col] = np.nan
            else:
                X[col] = 0
    # reorder columns
    X = X[FEATURE_COLS]
    return X

def predict_from_df(X: pd.DataFrame):
    """
    Accept a DataFrame X (one or more rows), ensure correct columns,
    fill defaults, and return numpy array of predictions.
    """
    model = _load_model()
    Xp = _ensure_columns(X)

    # sklearn may still complain if there are NaNs; handle lags NaN by simple imputation (mean of available)
    if Xp[["lag_1","lag_2"]].isna().any().any():
        # compute mean from available values in Xp or use 0
        for col in ["lag_1","lag_2"]:
            if Xp[col].isna().all():
                Xp[col] = 0.0
            else:
                mean_val = Xp[col].mean()
                Xp[col] = Xp[col].fillna(mean_val)

    preds = model.predict(Xp)
    return preds
