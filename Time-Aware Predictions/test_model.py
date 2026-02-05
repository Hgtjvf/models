"""
test_model.py

Phase-2 Testing Script for:
Smart Patient Risk Band – Time-Aware Risk Prediction

Features:
- Loads Phase-2 dataset
- Loads trained multi-output Random Forest model
- Computes evaluation metrics:
    * MAE
    * RMSE
    * R² score (accuracy)
- Generates a detailed report saved as reports.txt
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "dataset.csv"
MODEL_PATH = "model.joblib"
REPORT_FILE = "reports.txt"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATASET_PATH)

target_columns = [
    "risk_15min",
    "risk_1hour",
    "risk_6hours",
    "risk_24hours"
]

X = df.drop(columns=target_columns)
y = df[target_columns]

# -----------------------------
# Train/Test Split
# -----------------------------
_, X_test, _, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# -----------------------------
# Load Trained Model
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Trained model not found at '{MODEL_PATH}'. Please run train_model.py first."
    )

# -----------------------------
# Make Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Compute Metrics
# -----------------------------
# Overall metrics
mae_overall = mean_absolute_error(y_test, y_pred)
rmse_overall = np.sqrt(mean_squared_error(y_test, y_pred))
r2_overall = r2_score(y_test, y_pred)

# Per-horizon metrics
per_horizon_mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
per_horizon_rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2, axis=0))
per_horizon_r2 = [r2_score(y_test[col], y_pred[:, i]) for i, col in enumerate(target_columns)]

# -----------------------------
# Prepare Report
# -----------------------------
report_lines = []
report_lines.append("📄 Phase-2 Model Evaluation Report")
report_lines.append("----------------------------------\n")
report_lines.append("Overall Metrics:")
report_lines.append(f"MAE : {mae_overall:.4f}")
report_lines.append(f"RMSE: {rmse_overall:.4f}")
report_lines.append(f"R²  : {r2_overall:.4f}\n")

report_lines.append("Per-Horizon Metrics:")
for i, horizon in enumerate(target_columns):
    report_lines.append(
        f"{horizon:12s} | MAE: {per_horizon_mae[i]:.4f} | RMSE: {per_horizon_rmse[i]:.4f} | R²: {per_horizon_r2[i]:.4f}"
    )

report_lines.append("\n✅ Report generated successfully.\n")

# -----------------------------
# Save Report (UTF-8 safe)
# -----------------------------
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    for line in report_lines:
        f.write(line + "\n")

# -----------------------------
# Display Summary
# -----------------------------
print("\n".join(report_lines))
print(f"Report saved to: {REPORT_FILE}")
