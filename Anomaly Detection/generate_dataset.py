"""
generate_dataset.py

Simulates patient physiological data for anomaly detection in the
Smart Patient Risk Band project.

Features:
- HR: Heart Rate (bpm)
- HRV: Heart Rate Variability (ms)
- SpO2: Oxygen Saturation (%)
- Temperature: Body Temperature (°C)
- Activity: Binary (0=inactive, 1=active)
- FallEvent: Binary (0=no fall, 1=fall)

Outputs:
- anomaly_score: Max absolute z-score deviation across vitals
- baseline_deviation_flag: True if deviation exceeds threshold
"""

import numpy as np
import pandas as pd
import random

# -----------------------
# Simulation Parameters
# -----------------------
NUM_PATIENTS = 10
MINUTES_PER_DAY = 24 * 60
DAYS = 7
Z_THRESHOLD = 3  # For baseline deviation flag
ANOMALY_PROB = 0.01  # Probability of random anomaly per reading

# -----------------------
# Step 1: Simulate Patient Data
# -----------------------
def simulate_patient_data():
    data = []
    for patient_id in range(1, NUM_PATIENTS + 1):
        # Patient-specific baseline vitals
        hr_base = random.randint(60, 80)
        hrv_base = random.uniform(20, 50)
        spo2_base = random.uniform(95, 99)
        temp_base = random.uniform(36.4, 37.2)
        
        for minute in range(MINUTES_PER_DAY * DAYS):
            # Generate normal reading with small variations
            hr = np.random.normal(hr_base, 5)
            hrv = np.random.normal(hrv_base, 5)
            spo2 = np.random.normal(spo2_base, 0.5)
            temp = np.random.normal(temp_base, 0.2)
            activity = np.random.choice([0, 1], p=[0.7, 0.3])
            fall_event = np.random.choice([0, 1], p=[0.995, 0.005])
            
            # Introduce rare anomalies
            if np.random.rand() < ANOMALY_PROB:
                hr += np.random.choice([-30, 30])
                hrv += np.random.choice([-20, 20])
                spo2 -= np.random.uniform(2, 5)
                temp += np.random.uniform(1, 2)
            
            data.append([patient_id, minute, hr, hrv, spo2, temp, activity, fall_event])
    
    columns = ['patient_id', 'minute', 'HR', 'HRV', 'SpO2', 'Temperature', 'Activity', 'FallEvent']
    df = pd.DataFrame(data, columns=columns)
    return df

# -----------------------
# Step 2: Compute Baselines
# -----------------------
def compute_baselines(df):
    # Mean and std per patient
    baseline_stats = df.groupby('patient_id')[['HR','HRV','SpO2','Temperature']].agg(['mean','std'])
    # Flatten multi-index columns
    baseline_stats.columns = ['_'.join(col) for col in baseline_stats.columns]
    baseline_stats.reset_index(inplace=True)
    # Merge baseline back to main dataframe
    df = df.merge(baseline_stats, on='patient_id', how='left')
    return df

# -----------------------
# Step 3: Compute Z-Scores and Flags
# -----------------------
def compute_anomalies(df):
    for col in ['HR','HRV','SpO2','Temperature']:
        df[f'{col}_z'] = (df[col] - df[f'{col}_mean']) / df[f'{col}_std']
    
    # Anomaly score = max absolute z-score
    df['anomaly_score'] = df[['HR_z','HRV_z','SpO2_z','Temperature_z']].abs().max(axis=1)
    # Binary deviation flag
    df['baseline_deviation_flag'] = df['anomaly_score'] > Z_THRESHOLD
    
    return df

# -----------------------
# Step 4: Save Dataset
# -----------------------
def save_dataset(df, filename='simulated_patient_data.csv'):
    df.to_csv(filename, index=False)
    print(f"[INFO] Dataset saved to {filename}")

# -----------------------
# Main Function
# -----------------------
def main():
    print("[INFO] Simulating patient data...")
    df = simulate_patient_data()
    
    print("[INFO] Computing baselines...")
    df = compute_baselines(df)
    
    print("[INFO] Calculating anomaly scores and flags...")
    df = compute_anomalies(df)
    
    print("[INFO] Saving dataset...")
    save_dataset(df)
    
    print("[INFO] Dataset generation complete.")
    print(df.head())

# Run script
if __name__ == "__main__":
    main()
