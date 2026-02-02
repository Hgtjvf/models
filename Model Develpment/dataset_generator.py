import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================
# CONFIGURATION
# =========================
NUM_RECORDS = 500
NUM_PATIENTS = 50
START_TIME = datetime(2025, 7, 8, 6, 0, 0)

ACTIVITIES = ["resting", "sleeping", "walking", "running"]

ACTIVITY_PROFILES = {
    "sleeping": {
        "hr": (55, 70),
        "resp": (10, 14),
        "steps": (0, 50),
        "accel": (0.00, 0.00, 0.99),
        "gyro": (0.00, 0.00, 0.00),
    },
    "resting": {
        "hr": (65, 85),
        "resp": (12, 16),
        "steps": (0, 200),
        "accel": (0.01, 0.01, 0.98),
        "gyro": (0.01, 0.01, 0.01),
    },
    "walking": {
        "hr": (75, 100),
        "resp": (14, 18),
        "steps": (200, 1500),
        "accel": (0.12, 0.18, 0.95),
        "gyro": (0.08, 0.10, 0.06),
    },
    "running": {
        "hr": (95, 130),
        "resp": (18, 24),
        "steps": (500, 2500),
        "accel": (0.35, 0.45, 0.88),
        "gyro": (0.22, 0.25, 0.18),
    },
}

# =========================
# HELPERS
# =========================
def choose_activity():
    return np.random.choice(ACTIVITIES, p=[0.25, 0.20, 0.35, 0.20])

def bounded(mean, low, high):
    return max(low, min(high, int(mean)))

# =========================
# DATA GENERATION
# =========================
records = []

for i in range(NUM_RECORDS):
    patient_id = f"P{str(np.random.randint(1, NUM_PATIENTS + 1)).zfill(3)}"
    activity = choose_activity()
    profile = ACTIVITY_PROFILES[activity]

    heart_rate = np.random.randint(*profile["hr"])
    respiration_rate = np.random.randint(*profile["resp"])
    spo2 = np.round(np.random.uniform(92.0, 99.9), 1)

    body_temp = np.round(np.random.uniform(36.0, 37.8), 1)
    bp_sys = bounded(heart_rate + 50, 105, 145)
    bp_dia = bounded(bp_sys - np.random.randint(35, 50), 65, 90)

    glucose = np.round(np.random.uniform(80, 180), 1)
    steps = np.random.randint(*profile["steps"])
    fall = int(activity == "running" and np.random.rand() < 0.05)

    stress = bounded(heart_rate - 40 + np.random.randint(-5, 15), 0, 100)

    timestamp = START_TIME - timedelta(minutes=i)

    pulse_waveform = (
        "low_amplitude" if heart_rate < 65 else
        "high_amplitude" if heart_rate > 110 else
        "normal"
    )

    ibi = int(60000 / heart_rate)
    perfusion = round((spo2 / 100) * (heart_rate / 100), 2)

    ax, ay, az = profile["accel"]
    gx, gy, gz = profile["gyro"]

    records.append([
        patient_id, heart_rate, spo2, respiration_rate, body_temp,
        bp_sys, bp_dia, glucose, fall, activity, steps,
        round(12.90 + np.random.uniform(0, 0.10), 5),
        round(77.50 + np.random.uniform(0, 0.10), 5),
        stress, timestamp.isoformat(),
        pulse_waveform, ibi, perfusion,
        ax, ay, az, gx, gy, gz
    ])

# =========================
# CREATE DATAFRAME
# =========================
columns = [
    "patient_id", "heart_rate", "spo2_level", "respiration_rate",
    "body_temperature", "blood_pressure_sys", "blood_pressure_dia",
    "blood_glucose", "fall_detected", "activity_type", "step_count",
    "latitude", "longitude", "stress_level_index", "timestamp",
    "pulse_waveform", "inter_beat_interval_ms", "perfusion_index",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z"
]

df = pd.DataFrame(records, columns=columns)

# =========================
# SAVE CSV
# =========================
df.to_csv("dataset.csv", index=False)

print("✅ Synthetic realistic dataset generated.")
print("📁 File saved as: synthetic_wearable_health_dataset.csv")
