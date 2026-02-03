# ============================================================
# SMART PATIENT RISK BAND — COMPLETE MODEL TRAINING SCRIPT
# Fixes: missing risk_level, trains models, saves .joblib
# ============================================================

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder


# ============================================================
# SECTION 2 — CONFIG
# ============================================================
DATASET_PATH = "dataset.csv"
MODEL_PATH = "smart_patient_risk_model.joblib"


# ============================================================
# SECTION 3 — LOAD DATA
# ============================================================
print("\n📥 Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded:", df.shape)


# ============================================================
# SECTION 4 — CREATE RISK LEVEL (VERY IMPORTANT FIX)
# Deterministic, explainable medical logic
# ============================================================
def compute_risk(row):
    if (
        row["spo2_level"] < 92
        or row["heart_rate"] > 125
        or row["body_temperature"] > 38.5
        or row["blood_pressure_sys"] > 160
    ):
        return "high"

    elif (
        row["heart_rate"] > 100
        or row["blood_glucose"] > 180
        or row["stress_level_index"] > 75
        or row["respiration_rate"] > 24
    ):
        return "medium"

    else:
        return "low"


print("\n🧮 Generating risk_level from vitals...")
df["risk_level"] = df.apply(compute_risk, axis=1)
print("✅ risk_level column created.")


# ============================================================
# SECTION 5 — FEATURES
# ============================================================
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
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

X = df[FEATURES]
y = df["risk_level"]


# ============================================================
# SECTION 6 — ENCODE LABELS
# ============================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("✅ Labels encoded:", list(label_encoder.classes_))


# ============================================================
# SECTION 7 — TRAIN BASELINE RISK MODEL (E1)
# ============================================================
print("\n🧠 Training Random Forest Risk Model...")

risk_model = RandomForestClassifier(
    n_estimators=350,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
)

risk_model.fit(X, y_encoded)

print("✅ Risk model trained.")


# ============================================================
# SECTION 8 — TRAIN ANOMALY MODEL (E4)
# ============================================================
print("\n🔍 Training Isolation Forest for anomalies...")

anomaly_model = IsolationForest(
    n_estimators=250,
    contamination=0.05,
    random_state=42,
)

anomaly_model.fit(X)

print("✅ Anomaly model trained.")


# ============================================================
# SECTION 9 — SAVE JOBLIB
# ============================================================
print("\n💾 Saving complete model...")

joblib.dump(
    {
        "risk_model": risk_model,
        "anomaly_model": anomaly_model,
        "label_encoder": label_encoder,
        "features": FEATURES,
        "feature_importances": risk_model.feature_importances_,
    },
    MODEL_PATH
)

print("\n🎉 MODEL TRAINING COMPLETE")
print("📦 Saved as:", MODEL_PATH)
