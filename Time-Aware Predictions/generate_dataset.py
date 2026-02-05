"""
generate_dataset.py

Phase-2 Dataset Generator for:
Smart Patient Risk Band – Time-Aware Risk Prediction

This script simulates realistic physiological feature windows
and generates multi-horizon future risk probabilities.

Outputs:
- phase2_dataset.csv
"""

import numpy as np
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
RANDOM_SEED = 42
N_SAMPLES = 5000
OUTPUT_FILE = "dataset.csv"

np.random.seed(RANDOM_SEED)

# -----------------------------
# Utility Functions
# -----------------------------
def sigmoid(x: float) -> float:
    """Convert risk score to probability (0–1)."""
    return 1 / (1 + np.exp(-x))


# -----------------------------
# Dataset Generation
# -----------------------------
records = []

for _ in range(N_SAMPLES):

    # -------------------------
    # Heart Rate Features
    # -------------------------
    hr_mean = np.random.normal(loc=75, scale=10)
    hr_trend = np.random.normal(loc=0.0, scale=0.5)
    hrv_rmssd = np.random.normal(loc=40, scale=10)

    # -------------------------
    # SpO2 Features
    # -------------------------
    spo2_mean = np.random.normal(loc=96, scale=2)
    spo2_trend = np.random.normal(loc=0.0, scale=0.2)
    desat_events = np.random.poisson(lam=0.3)

    # -------------------------
    # Temperature Features
    # -------------------------
    temp_mean = np.random.normal(loc=36.8, scale=0.4)
    temp_rise_rate = np.random.normal(loc=0.0, scale=0.05)

    # -------------------------
    # Activity & Motion
    # -------------------------
    activity_level = np.random.uniform(0.0, 1.0)
    fall_detected = np.random.choice([0, 1], p=[0.97, 0.03])

    # -------------------------
    # Hidden Risk Logic
    # (Ground truth for simulation)
    # -------------------------
    base_risk_score = (
        0.04 * (hr_mean > 100) +
        0.06 * (spo2_mean < 92) +
        0.07 * desat_events +
        0.08 * (temp_rise_rate > 0.1) +
        0.20 * fall_detected
    )

    # -------------------------
    # Multi-Horizon Risk Targets
    # -------------------------
    risk_15min = sigmoid(base_risk_score + np.random.normal(0.0, 0.10))
    risk_1hour = sigmoid(base_risk_score + np.random.normal(0.2, 0.15))
    risk_6hours = sigmoid(base_risk_score + np.random.normal(0.4, 0.20))
    risk_24hours = sigmoid(base_risk_score + np.random.normal(0.6, 0.25))

    records.append([
        hr_mean,
        hr_trend,
        hrv_rmssd,
        spo2_mean,
        spo2_trend,
        desat_events,
        temp_mean,
        temp_rise_rate,
        activity_level,
        fall_detected,
        risk_15min,
        risk_1hour,
        risk_6hours,
        risk_24hours
    ])

# -----------------------------
# Create DataFrame
# -----------------------------
columns = [
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
    "risk_15min",
    "risk_1hour",
    "risk_6hours",
    "risk_24hours"
]

df = pd.DataFrame(records, columns=columns)

# -----------------------------
# Save Dataset
# -----------------------------
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Phase-2 dataset generated successfully: {OUTPUT_FILE}")
print(f"Samples: {len(df)} | Features: {len(columns) - 4} | Targets: 4")
