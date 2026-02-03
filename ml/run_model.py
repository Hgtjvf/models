# ============================================================
# SMART PATIENT RISK BAND — RUN MODEL (E1 to E4)
# Works with your exact folder structure
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib


# ============================================================
# SECTION 1 — PATH HANDLING
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "ml/models/model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "ml/data/testing_dataset.csv")


# ============================================================
# SECTION 2 — LOAD MODEL
# ============================================================
print("\n📦 Loading model from:", MODEL_PATH)
data = joblib.load(MODEL_PATH)

risk_model = data["risk_model"]
anomaly_model = data["anomaly_model"]
label_encoder = data["label_encoder"]
FEATURES = data["features"]


# ============================================================
# SECTION 3 — LOAD TEST DATA
# ============================================================
print("📥 Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

sample = df.iloc[[0]]   # first patient row
X = sample[FEATURES]


# ============================================================
# E1 — BASELINE RISK MODEL
# ============================================================
probs = risk_model.predict_proba(X)[0]
pred_class = np.argmax(probs)
risk_label = label_encoder.inverse_transform([pred_class])[0]
confidence_score = np.max(probs)

top_features = sorted(
    zip(FEATURES, risk_model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)[:5]


# ============================================================
# E2 — TIME-AWARE RISK (Heuristic Projection)
# ============================================================
hr = sample["heart_rate"].values[0]
spo2 = sample["spo2_level"].values[0]

risk_15min = min(1.0, confidence_score + (hr > 110) * 0.05)
risk_1hour = min(1.0, confidence_score + (hr > 120) * 0.10)
risk_6hours = min(1.0, confidence_score + (spo2 < 94) * 0.15)
risk_24hours = min(1.0, confidence_score + (spo2 < 92) * 0.25)


# ============================================================
# E3 — EVENT SPECIFIC RISKS
# ============================================================
heart_attack_risk = (
    (hr > 120) * 0.4 +
    (sample["blood_pressure_sys"].values[0] > 140) * 0.3 +
    (sample["blood_glucose"].values[0] > 180) * 0.3
)

arrhythmia_risk = (sample["inter_beat_interval_ms"].values[0] > 1200) * 0.7
respiratory_failure_risk = (spo2 < 92) * 0.8
stroke_risk = (sample["blood_pressure_sys"].values[0] > 150) * 0.7
sepsis_risk = (sample["body_temperature"].values[0] > 38.5) * 0.6
severe_fall_risk = (sample["accel_z"].values[0] > 3.0) * 0.9


# ============================================================
# E4 — ANOMALY DETECTION
# ============================================================
anomaly_score = anomaly_model.decision_function(X)[0]
baseline_deviation_flag = anomaly_model.predict(X)[0] == -1


# ============================================================
# PRINT OUTPUT
# ============================================================
print("\n================ 🟥 E1: BASELINE RISK =================")
print("Risk Level:", risk_label)
print("Risk Probabilities:", probs)
print("Confidence Score:", confidence_score)
print("Top Contributing Features:", top_features)

print("\n================ 🟧 E2: TIME-AWARE RISK ================")
print("15 Minutes:", risk_15min)
print("1 Hour:", risk_1hour)
print("6 Hours:", risk_6hours)
print("24 Hours:", risk_24hours)

print("\n================ 🟨 E3: EVENT RISKS =====================")
print("Heart Attack Risk:", heart_attack_risk)
print("Arrhythmia Risk:", arrhythmia_risk)
print("Respiratory Failure Risk:", respiratory_failure_risk)
print("Stroke Risk:", stroke_risk)
print("Sepsis Risk:", sepsis_risk)
print("Severe Fall Risk:", severe_fall_risk)

print("\n================ 🟩 E4: ANOMALY =========================")
print("Anomaly Score:", anomaly_score)
print("Deviation Flag:", baseline_deviation_flag)
