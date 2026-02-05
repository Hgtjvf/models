# evaluate_event_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# -------------------------------
# 1. Load trained model
# -------------------------------
model_path = "multi_rf_event_risk.joblib"
multi_rf = joblib.load(model_path)
print(f"Loaded model: {model_path}")

# -------------------------------
# 2. Load test dataset
# -------------------------------
test_data_path = "derived_physio_dataset.csv"  # or separate test set
df = pd.read_csv(test_data_path)

feature_cols = [
    "resting_hr", "hr_trend", "hr_rmssd", "tachycardia", "cardiac_load",
    "avg_spo2", "spo2_trend", "desat_events", "resp_strain",
    "temp_deviation", "temp_rise_rate", "fever_flag",
    "acc_magnitude", "inactivity_duration_min", "fall", "post_fall_immobility"
]

X_test = df[feature_cols].copy()

activity_map = {"Rest": 0, "Sleep": 1, "Active": 2}
X_test["activity_level"] = df["activity_level"].map(activity_map)

event_cols = [
    "heart_attack_risk",
    "arrhythmia_risk",
    "respiratory_failure_risk",
    "stroke_risk",
    "sepsis_risk",
    "severe_fall_risk"
]

y_test = df[event_cols].copy()

# -------------------------------
# 3. Predict on test set
# -------------------------------
y_pred = multi_rf.predict(X_test)

# -------------------------------
# 4. Compute metrics per event
# -------------------------------
metrics = []

for i, event in enumerate(event_cols):
    y_true = y_test[event]
    y_p = y_pred[:, i]
    
    accuracy = accuracy_score(y_true, y_p)
    precision = precision_score(y_true, y_p, zero_division=0)
    recall = recall_score(y_true, y_p, zero_division=0)
    
    # False alert rate = predicted 1 when actual 0
    false_alerts = np.sum((y_p == 1) & (y_true == 0))
    false_alert_rate = false_alerts / max(1, np.sum(y_true == 0))  # avoid division by zero
    
    metrics.append({
        "event": event,
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "false_alert_rate": round(false_alert_rate, 3)
    })
    
    # Optional: print confusion matrix
    print(f"\n--- {event} ---")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_p))
    print("Classification Report:\n", classification_report(y_true, y_p, zero_division=0))

# -------------------------------
# 5. Save metrics to CSV
# -------------------------------
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("evaluation_report.csv", index=False)
print("\nEvaluation report saved: evaluation_report.csv")
