"""
train_model.py

Phase-2 Training Script for:
Smart Patient Risk Band – Time-Aware Risk Prediction
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "dataset.csv"
MODEL_PATH = "model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATASET_PATH)

# -----------------------------
# Split Features & Targets
# -----------------------------
target_columns = [
    "risk_15min",
    "risk_1hour",
    "risk_6hours",
    "risk_24hours"
]

X = df.drop(columns=target_columns)
y = df[target_columns]

# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# -----------------------------
# Model Definition
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# -----------------------------
# Model Training
# -----------------------------
print("🚀 Training Phase-2 Time-Aware Risk Model...")
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

# FIX: Compute RMSE safely
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n📊 Evaluation Results")
print(f"Mean Absolute Error (MAE):  {mae:.4f}")
print(f"Root Mean Squared Error:   {rmse:.4f}")

# Per-horizon MAE (academic reporting)
per_horizon_mae = mean_absolute_error(
    y_test,
    y_pred,
    multioutput="raw_values"
)

print("\n📈 Per-Horizon MAE")
for horizon, score in zip(target_columns, per_horizon_mae):
    print(f"{horizon}: {score:.4f}")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, MODEL_PATH)

print(f"\n✅ Model saved successfully: {MODEL_PATH}")
