from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import pandas as pd
import os
import csv
from dateutil import parser as dateparser

# -----------------------------
# Import project-specific utils
# -----------------------------
from app.features import make_features
from app.model_utils import predict_from_df

# FastAPI app initialization
app = FastAPI(title="Web Traffic Forecast API", version="1.0")

# ---------------------------------------
# REQUEST & RESPONSE MODELS FOR /predict
# ---------------------------------------
class PredictRequest(BaseModel):
    timestamp: str
    lag_1: Optional[float] = None
    lag_2: Optional[float] = None
    is_festival: Optional[int] = 0

class PredictResponse(BaseModel):
    prediction: float
    input: dict

# -----------------
# Health check
# -----------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------
# PREDICTION ROUTE
# -----------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Convert input into processed features dataframe
    X = make_features(req.timestamp, req.lag_1, req.lag_2, req.is_festival)

    # Run prediction
    pred = predict_from_df(X)

    # Response structure
    return {
        "prediction": float(pred),
        "input": X.to_dict(orient="records")[0]
    }

# -------------------------------------------
# REPORT ENDPOINT FOR SAVING ACTUAL TRAFFIC
# -------------------------------------------

# Path where data will be stored (inside /app/data)
DATA_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "traffic.csv")

class ReportRequest(BaseModel):
    timestamp: str
    actual_page_views: int
    is_festival: Optional[int] = 0

@app.post("/report")
def report(req: ReportRequest):
    # Validate timestamp
    try:
        ts = dateparser.parse(req.timestamp)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")

    # Ensure /data directory exists
    data_dir = os.path.dirname(DATA_CSV)
    os.makedirs(data_dir, exist_ok=True)

    # Check if CSV exists for header writing
    file_exists = os.path.exists(DATA_CSV)

    # Write / Append actual data
    try:
        with open(DATA_CSV, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if CSV is new
            if not file_exists:
                writer.writerow(["timestamp", "page_views", "is_festival"])

            # Append actual observation
            writer.writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                req.actual_page_views,
                int(req.is_festival)
            ])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write CSV: {e}")

    return {"status": "ok", "message": "Appended observation"}
