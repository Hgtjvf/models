# ============================================================
# SMART PATIENT RISK BAND — TEST DATASET GENERATOR
# Generates realistic sensor data + correct risk_level
# ============================================================

import pandas as pd
import numpy as np
import random

OUTPUT_FILE = "testing_dataset.csv"
NUM_SAMPLES = 1000


# ============================================================
# SECTION 1 — REALISTIC SENSOR VALUE GENERATORS
# ============================================================

def rand_range(a, b):
    return round(random.uniform(a, b), 2)


def generate_row():
    heart_rate = rand_range(55, 140)
    spo2_level = rand_range(85, 100)
    respiration_rate = rand_range(10, 30)
    body_temperature = rand_range(35.5, 40.0)
    blood_pressure_sys = rand_range(95, 180)
    blood_pressure_dia = rand_range(60, 110)
    blood_glucose = rand_range(70, 250)
    stress_level_index = rand_range(10, 100)
    step_count = rand_range(0, 8000)
    perfusion_index = rand_range(0.5, 10)
    inter_beat_interval_ms = rand_range(600, 1500)
    accel_x = rand_range(-4, 4)
    accel_y = rand_range(-4, 4)
    accel_z = rand_range(-4, 4)
    gyro_x = rand_range(-300, 300)
    gyro_y = rand_range(-300, 300)
    gyro_z = rand_range(-300, 300)

    # ========================================================
    # SECTION 2 — SAME RISK LOGIC USED IN TRAINING
    # ========================================================
    if (
        spo2_level < 92
        or heart_rate > 125
        or body_temperature > 38.5
        or blood_pressure_sys > 160
    ):
        risk_level = "high"

    elif (
        heart_rate > 100
        or blood_glucose > 180
        or stress_level_index > 75
        or respiration_rate > 24
    ):
        risk_level = "medium"

    else:
        risk_level = "low"

    return [
        heart_rate,
        spo2_level,
        respiration_rate,
        body_temperature,
        blood_pressure_sys,
        blood_pressure_dia,
        blood_glucose,
        stress_level_index,
        step_count,
        perfusion_index,
        inter_beat_interval_ms,
        accel_x,
        accel_y,
        accel_z,
        gyro_x,
        gyro_y,
        gyro_z,
        risk_level,
    ]


# ============================================================
# SECTION 3 — GENERATE DATASET
# ============================================================

columns = [
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
    "risk_level",
]

print("\n🧪 Generating test dataset...")

data = [generate_row() for _ in range(NUM_SAMPLES)]

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)

print("✅ Test dataset created:", OUTPUT_FILE)
print("Rows:", len(df))
