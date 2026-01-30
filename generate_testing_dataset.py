import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------
# Output path
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "testing_dataset.csv"

np.random.seed(42)

N_SAMPLES = 4000

# -------------------------------------------------
# Generate realistic sensor data
# -------------------------------------------------
df = pd.DataFrame({
    "heart_rate": np.random.randint(55, 140, N_SAMPLES),
    "spo2_level": np.random.randint(88, 100, N_SAMPLES),
    "respiration_rate": np.random.randint(10, 30, N_SAMPLES),
    "body_temperature": np.round(np.random.uniform(35.5, 40.0, N_SAMPLES), 1),
    "blood_pressure_sys": np.random.randint(90, 180, N_SAMPLES),
    "blood_pressure_dia": np.random.randint(60, 120, N_SAMPLES),
    "blood_glucose": np.random.randint(70, 220, N_SAMPLES),
    "stress_level_index": np.random.randint(10, 100, N_SAMPLES),
    "step_count": np.random.randint(0, 20000, N_SAMPLES),
    "perfusion_index": np.round(np.random.uniform(0.5, 8.0, N_SAMPLES), 2),
    "inter_beat_interval_ms": np.random.randint(500, 1200, N_SAMPLES),
    "accel_x": np.round(np.random.uniform(-3, 3, N_SAMPLES), 2),
    "accel_y": np.round(np.random.uniform(-3, 3, N_SAMPLES), 2),
    "accel_z": np.round(np.random.uniform(-3, 3, N_SAMPLES), 2),
    "gyro_x": np.round(np.random.uniform(-250, 250, N_SAMPLES), 2),
    "gyro_y": np.round(np.random.uniform(-250, 250, N_SAMPLES), 2),
    "gyro_z": np.round(np.random.uniform(-250, 250, N_SAMPLES), 2),
})

# -------------------------------------------------
# Risk level logic (MATCHES TRAINING CODE)
# -------------------------------------------------
def compute_risk(row):
    if (
        row["heart_rate"] > 120
        or row["spo2_level"] < 93
        or row["body_temperature"] > 38
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

# -------------------------------------------------
# Save CSV
# -------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("✅ testing_dataset.csv generated successfully")
print(f"📁 Location: {OUTPUT_PATH}")
print("\n📊 Class distribution:")
print(df["risk_level"].value_counts())
