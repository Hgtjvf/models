"""
predict_model.py

Phase-2 Inference Script for:
Smart Patient Risk Band – Time-Aware Risk Prediction

Loads the trained multi-output model and predicts
future patient risk probabilities at multiple horizons.
"""

import numpy as np
import joblib

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "model.joblib"

# -----------------------------
# Load Trained Model
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        "Trained model not found. Please run train_model.py first."
    )

# -----------------------------
# Input Feature Order (IMPORTANT)
# Must match training dataset exactly
# -----------------------------
FEATURE_ORDER = [
    "hr_mean",
    "hr_trend",
    "hrv_rmssd",
    "spo2_mean",
    "spo2_trend",
    "desat_events",
    "temp_mean",
    "temp_rise_rate",
    "activity_level",
    "fall_detected",
]

# -----------------------------
# Example Input
# (Derived metrics from sliding window)
# -----------------------------
input_features = {
    "hr_mean": 102.0,
    "hr_trend": 1.2,
    "hrv_rmssd": 25.0,
    "spo2_mean": 91.0,
    "spo2_trend": -0.4,
    "desat_events": 3,
    "temp_mean": 37.9,
    "temp_rise_rate": 0.15,
    "activity_level": 0.2,
    "fall_detected": 0,
}

# -----------------------------
# Convert Input to Model Format
# -----------------------------
sample = np.array([[input_features[f] for f in FEATURE_ORDER]])

# -----------------------------
# Prediction
# -----------------------------
predicted = model.predict(sample)[0]

# -----------------------------
# Output Mapping
# -----------------------------
risk_predictions = {
    "risk_15min": round(float(predicted[0]), 3),
    "risk_1hour": round(float(predicted[1]), 3),
    "risk_6hours": round(float(predicted[2]), 3),
    "risk_24hours": round(float(predicted[3]), 3),
}

# -----------------------------
# Display Results
# -----------------------------
print("\n🩺 Time-Aware Patient Risk Prediction")
print("-----------------------------------")
for horizon, value in risk_predictions.items():
    print(f"{horizon:12s}: {value}")
