import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "model.joblib"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("dataset.csv")

# =========================
# CREATE TARGET (RISK LEVEL)
# Deterministic + explainable
# =========================
def compute_risk(row):
    if (
        row["heart_rate"] > 120
        or row["spo2_level"] < 93
        or row["body_temperature"] > 38
        or row["fall_detected"] == 1
    ):
        return "high"
    elif (
        row["heart_rate"] > 95
        or row["stress_level_index"] > 70
        or row["blood_glucose"] > 160
    ):
        return "medium"
    else:
        return "low"

df["risk_level"] = df.apply(compute_risk, axis=1)

# =========================
# FEATURE SELECTION
# =========================
FEATURES = [
    "heart_rate",
    "spo2_level",
    "respiration_rate",
    "body_temperature",
    "blood_pressure_sys",
    "blood_pressure_dia",
    "blood_glucose",
    "stress_level_index",
    "step_count",
    "perfusion_index",
    "inter_beat_interval_ms",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
]

X = df[FEATURES]
y = df["risk_level"]

# =========================
# ENCODE TARGET
# =========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =========================
# TRAIN MODEL (ONCE)
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
)

model.fit(X, y_encoded)

# =========================
# SAVE MODEL + ENCODER
# =========================
joblib.dump(
    {
        "model": model,
        "label_encoder": label_encoder,
        "features": FEATURES,
    },
    MODEL_PATH
)

print("✅ Model trained once and saved.")
print(f"📦 Saved as: {MODEL_PATH}")
