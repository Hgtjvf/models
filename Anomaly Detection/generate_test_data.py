"""
generate_test_data.py

Generates a simulated test dataset for Smart Patient Risk Band models.

- Columns: patient_id, HR, HRV, SpO2, Temperature, Activity, FallEvent, label
- Normal readings: label=0
- Anomalous readings: label=1
- Saves as test_patient_data.csv
"""

import pandas as pd
import numpy as np
import os

# -----------------------
# Configuration
# -----------------------
NUM_PATIENTS = 5           # Number of patients
NORMAL_SAMPLES = 50        # Number of normal readings per patient
ANOMALY_SAMPLES = 5        # Number of anomalous readings per patient
OUTPUT_FILE = 'test_patient_data.csv'
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# -----------------------
# Function to generate readings
# -----------------------
def generate_readings(patient_id, num_normal, num_anomaly):
    # Normal ranges
    HR_mean, HR_std = 75, 10
    HRV_mean, HRV_std = 40, 5
    SpO2_mean, SpO2_std = 97, 1
    Temp_mean, Temp_std = 36.8, 0.3
    Activity_mean, Activity_std = 1, 0.2
    FallEvent_mean, FallEvent_std = 0, 0.1

    # Generate normal readings
    normal_df = pd.DataFrame({
        'patient_id': patient_id,
        'HR': np.random.normal(HR_mean, HR_std, num_normal).astype(int),
        'HRV': np.random.normal(HRV_mean, HRV_std, num_normal).astype(int),
        'SpO2': np.random.normal(SpO2_mean, SpO2_std, num_normal).round(1),
        'Temperature': np.random.normal(Temp_mean, Temp_std, num_normal).round(1),
        'Activity': np.random.normal(Activity_mean, Activity_std, num_normal).round().astype(int),
        'FallEvent': np.random.normal(FallEvent_mean, FallEvent_std, num_normal).round().astype(int),
        'label': 0
    })

    # Generate anomalous readings (extreme values)
    anomaly_df = pd.DataFrame({
        'patient_id': patient_id,
        'HR': np.random.choice([40, 130, 150], num_anomaly),
        'HRV': np.random.choice([5, 80], num_anomaly),
        'SpO2': np.random.choice([85, 99], num_anomaly),
        'Temperature': np.random.choice([34, 39, 41], num_anomaly),
        'Activity': np.random.choice([0, 5], num_anomaly),
        'FallEvent': np.random.choice([0, 1], num_anomaly),
        'label': 1
    })

    return pd.concat([normal_df, anomaly_df], ignore_index=True)

# -----------------------
# Generate dataset for all patients
# -----------------------
all_patients = []

for pid in range(1, NUM_PATIENTS + 1):
    patient_df = generate_readings(pid, NORMAL_SAMPLES, ANOMALY_SAMPLES)
    all_patients.append(patient_df)

test_df = pd.concat(all_patients, ignore_index=True)

# Shuffle rows
test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Save to CSV
test_df.to_csv(OUTPUT_FILE, index=False)
print(f"[INFO] Test dataset generated: {OUTPUT_FILE} ({test_df.shape[0]} rows)")
