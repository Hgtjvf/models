from pathlib import Path
from joblib import load
import pandas as pd

# -------------------------------------------------
# Locate model file
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

# -------------------------------------------------
# Load saved bundle
# -------------------------------------------------
bundle = load(MODEL_PATH)
print("✅ Model file loaded")

model = bundle["model"]
label_encoder = bundle["label_encoder"]
features = bundle["features"]

print("ℹ️ Model, encoder, and feature list extracted")

# -------------------------------------------------
# ONE patient snapshot (dict → DataFrame)
# -------------------------------------------------
sample_input = {
    "heart_rate": 72,
    "spo2_level": 98,
    "respiration_rate": 16,
    "body_temperature": 36.8,
    "blood_pressure_sys": 120,
    "blood_pressure_dia": 80,
    "blood_glucose": 95,
    "stress_level_index": 30,
    "step_count": 1200,
    "perfusion_index": 5.2,
    "inter_beat_interval_ms": 830,
    "accel_x": 0.01,
    "accel_y": -0.02,
    "accel_z": 0.98,
    "gyro_x": 0.001,
    "gyro_y": 0.002,
    "gyro_z": 0.0005,
}

# Create DataFrame with correct feature order
X_input = pd.DataFrame([sample_input], columns=features)

# -------------------------------------------------
# Predict
# -------------------------------------------------
encoded_pred = model.predict(X_input)[0]
risk_label = label_encoder.inverse_transform([encoded_pred])[0]

print("🩺 Predicted Risk Level:", risk_label.upper())
