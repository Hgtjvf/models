# train_event_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("derived_physio_dataset.csv")

# -------------------------------
# 2. Features & multi-label targets
# -------------------------------
feature_cols = [
    "resting_hr", "hr_trend", "hr_rmssd", "tachycardia", "cardiac_load",
    "avg_spo2", "spo2_trend", "desat_events", "resp_strain",
    "temp_deviation", "temp_rise_rate", "fever_flag",
    "acc_magnitude", "inactivity_duration_min", "fall", "post_fall_immobility"
]

X = df[feature_cols].copy()

# Convert activity_level to numeric safely
activity_map = {"Rest": 0, "Sleep": 1, "Active": 2}
X["activity_level"] = df["activity_level"].map(activity_map)

# Multi-label targets: each event is 0/1
event_cols = [
    "heart_attack_risk",
    "arrhythmia_risk",
    "respiratory_failure_risk",
    "stroke_risk",
    "sepsis_risk",
    "severe_fall_risk"
]

y = df[event_cols].copy()

# -------------------------------
# 3. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Multi-Output Random Forest
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

multi_rf = MultiOutputClassifier(rf)
multi_rf.fit(X_train, y_train)

# -------------------------------
# 5. Evaluate model (per event)
# -------------------------------
y_pred = multi_rf.predict(X_test)

print("=== Event-Specific Classification Reports ===")
for i, event in enumerate(event_cols):
    print(f"\n--- {event} ---")
    print(classification_report(y_test[event], y_pred[:, i]))

# -------------------------------
# 6. Save model (.joblib)
# -------------------------------
joblib.dump(multi_rf, "multi_rf_event_risk.joblib")
print("Multi-output Random Forest saved: multi_rf_event_risk.joblib")
