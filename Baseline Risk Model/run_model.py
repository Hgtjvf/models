import pandas as pd
import joblib
import numpy as np

# ========================
# Load model and columns
# ========================
model_file = "baseline_risk_rf_model.joblib"
columns_file = "baseline_risk_model_columns.joblib"

rf_model = joblib.load(model_file)
feature_columns = joblib.load(columns_file)

print("Model and feature columns loaded successfully!")

# ========================
# Example input data
# ========================
input_data = {
    "resting_hr": [75],
    "hr_trend": [0.2],
    "rmssd": [30],
    "tachycardia": [0],
    "cardiac_load": [1.1],
    "avg_spo2": [97],
    "spo2_trend": [0.1],
    "desat_events": [0],
    "recovery_time": [0],
    "resp_strain": [0.77],
    "temp_baseline": [36.5],
    "temp_deviation": [0.2],
    "temp_rise_rate": [0.01],
    "fever": [0],
    "activity_level_rest": [1],
    "activity_level_active": [0],
    "activity_level_sleep": [0],
    "inactivity_minutes": [10],
    "fall": [0],
    "post_fall_immobility": [0]
}

df = pd.DataFrame(input_data)

# ========================
# Align columns with training data
# ========================
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0
df = df[feature_columns]

# ========================
# Run prediction
# ========================
risk_level = rf_model.predict(df)[0]

# Random Forest predict_proba
risk_probability = rf_model.predict_proba(df)[0]  # array of [Low, Moderate, High]

# Confidence score: maximum probability
confidence_score = np.max(risk_probability)

# Top contributing features using feature importances
importances = rf_model.feature_importances_
feature_importances = pd.Series(importances, index=feature_columns)
top_contributing_features = feature_importances.sort_values(ascending=False).head(5).to_dict()

# ========================
# Print outputs
# ========================
print(f"Risk Level (0=Low,1=Moderate,2=High): {risk_level}")
print(f"Risk Probabilities [Low, Moderate, High]: {risk_probability}")
print(f"Confidence Score: {confidence_score}")
print(f"Top Contributing Features: {top_contributing_features}")
