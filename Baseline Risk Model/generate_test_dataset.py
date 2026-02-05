import pandas as pd
import numpy as np

# Number of test samples
num_samples = 50

# Generate random plausible physiological metrics
test_data = {
    "resting_hr": np.random.randint(60, 100, size=num_samples),
    "hr_trend": np.random.uniform(-0.5, 0.5, size=num_samples),
    "rmssd": np.random.randint(20, 80, size=num_samples),
    "tachycardia": np.random.randint(0, 2, size=num_samples),
    "cardiac_load": np.random.uniform(0.8, 1.5, size=num_samples),
    "avg_spo2": np.random.randint(90, 100, size=num_samples),
    "spo2_trend": np.random.uniform(-0.2, 0.2, size=num_samples),
    "desat_events": np.random.randint(0, 5, size=num_samples),
    "recovery_time": np.random.randint(0, 10, size=num_samples),
    "resp_strain": np.random.uniform(0.5, 1.5, size=num_samples),
    "temp_baseline": np.random.uniform(36, 37.5, size=num_samples),
    "temp_deviation": np.random.uniform(-0.5, 1.0, size=num_samples),
    "temp_rise_rate": np.random.uniform(0, 0.05, size=num_samples),
    "fever": np.random.randint(0, 2, size=num_samples),
    "activity_level_rest": np.random.randint(0, 2, size=num_samples),
    "activity_level_active": np.random.randint(0, 2, size=num_samples),
    "activity_level_sleep": np.random.randint(0, 2, size=num_samples),
    "inactivity_minutes": np.random.randint(0, 60, size=num_samples),
    "fall": np.random.randint(0, 2, size=num_samples),
    "post_fall_immobility": np.random.randint(0, 2, size=num_samples),
    # Optional: generate corresponding risk_level for evaluation
    "risk_level": np.random.randint(0, 3, size=num_samples)
}

df_test = pd.DataFrame(test_data)
df_test.to_csv("baseline_risk_test_dataset.csv", index=False)

print("Test dataset saved as baseline_risk_test_dataset.csv")
