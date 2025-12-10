import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

df = pd.read_csv("data/traffic.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Feature Engineering
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# Lag Features
df["lag_1"] = df["page_views"].shift(1)
df["lag_2"] = df["page_views"].shift(2)

# Drop NaN rows from lag
df = df.dropna()

# Input & Output
X = df[["hour", "day", "month", "lag_1", "lag_2", "is_festival"]]
y = df["page_views"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)

# Prediction
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("Model MAE:", mae)

# Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/traffic_model.joblib")

print("✅ Model Saved Successfully!")
