# dataset_generator.py

import pandas as pd
import numpy as np

# -------------------------------
# 1. Parameters
# -------------------------------
NUM_SAMPLES = 1200  # number of simulated readings
WINDOW_SIZE = 5     # for trend calculations (simplified)

# -------------------------------
# 2. Helper functions
# -------------------------------
def slope(values):
    """Compute linear slope of a sequence."""
    x = np.arange(len(values))
    return np.polyfit(x, values, 1)[0]

def compute_rmssd(ibi):
    """Heart rate variability RMSSD from inter-beat intervals."""
    ibi = np.array(ibi)
    return np.sqrt(np.mean(np.diff(ibi)**2))

def classify_activity(acc_mag):
    """Classify activity level based on acceleration magnitude."""
    if acc_mag < 0.05:
        return "Rest"
    elif acc_mag < 0.3:
        return "Sleep"
    else:
        return "Active"

def generate_event_labels(row):
    """Generate event-specific risk labels based on simple rules."""
    labels = {
        "heart_attack_risk": int(row["resting_hr"] > 100 and row["tachycardia"] == 1),
        "arrhythmia_risk": int(row["hr_rmssd"] < 20 and row["tachycardia"] == 1),
        "respiratory_failure_risk": int(row["avg_spo2"] < 90 or row["desat_events"] > 2),
        "stroke_risk": int(row["tachycardia"] == 1 and row["fever_flag"] == 1),
        "sepsis_risk": int(row["fever_flag"] == 1 and row["temp_rise_rate"] > 0.5),
        "severe_fall_risk": int(row["fall"] == 1 and row["post_fall_immobility"] > 5)
    }
    return pd.Series(labels)

# -------------------------------
# 3. Generate simulated dataset
# -------------------------------
data = []

for _ in range(NUM_SAMPLES):
    # Simulate raw sensor readings
    hr_window = np.random.randint(60, 120, size=WINDOW_SIZE)  # BPM
    spo2_window = np.random.randint(88, 100, size=WINDOW_SIZE) # %
    temp_window = np.random.uniform(36.0, 39.0, size=WINDOW_SIZE)  # °C
    acc_window = np.random.uniform(0, 1, size=WINDOW_SIZE)  # acceleration magnitude
    
    # Derived features
    resting_hr = np.mean(hr_window)
    hr_trend = slope(hr_window)
    hr_rmssd = compute_rmssd(hr_window)
    tachycardia = int(resting_hr > 100)  # simple threshold
    cardiac_load = resting_hr / 70  # baseline_hr ~70
    
    avg_spo2 = np.mean(spo2_window)
    spo2_trend = slope(spo2_window)
    desat_events = np.sum(spo2_window < 90)
    resp_strain = resting_hr / avg_spo2
    
    temp_baseline = np.median(temp_window)
    temp_deviation = temp_window[-1] - temp_baseline
    temp_rise_rate = slope(temp_window)
    fever_flag = int(temp_window[-1] > 38 and temp_rise_rate > 0)
    
    acc_magnitude = np.mean(acc_window)
    inactivity_duration_min = np.random.randint(0, 60) if acc_magnitude < 0.05 else 0
    fall = int(acc_magnitude > 0.7 and np.random.rand() > 0.5)
    post_fall_immobility = np.random.randint(0, 20) if fall else 0
    
    activity_level = classify_activity(acc_magnitude)
    
    # Combine all features
    row = {
        "resting_hr": resting_hr,
        "hr_trend": hr_trend,
        "hr_rmssd": hr_rmssd,
        "tachycardia": tachycardia,
        "cardiac_load": cardiac_load,
        "avg_spo2": avg_spo2,
        "spo2_trend": spo2_trend,
        "desat_events": desat_events,
        "resp_strain": resp_strain,
        "temp_deviation": temp_deviation,
        "temp_rise_rate": temp_rise_rate,
        "fever_flag": fever_flag,
        "acc_magnitude": acc_magnitude,
        "inactivity_duration_min": inactivity_duration_min,
        "fall": fall,
        "post_fall_immobility": post_fall_immobility,
        "activity_level": activity_level
    }
    
    # Add event-specific labels
    row.update(generate_event_labels(row))
    
    data.append(row)

# -------------------------------
# 4. Create DataFrame & save
# -------------------------------
df = pd.DataFrame(data)
df.to_csv("derived_physio_dataset.csv", index=False)
print("Dataset generated: derived_physio_dataset.csv")
