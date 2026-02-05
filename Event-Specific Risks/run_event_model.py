# run_event_model.py

import pandas as pd
import numpy as np
import joblib

# -------------------------------
# 1. Load trained model
# -------------------------------
multi_rf = joblib.load("multi_rf_event_risk.joblib")

event_cols = [
    "heart_attack_risk",
    "arrhythmia_risk",
    "respiratory_failure_risk",
    "stroke_risk",
    "sepsis_risk",
    "severe_fall_risk"
]

# -------------------------------
# 2. Prepare input function
# -------------------------------
def prepare_input(sensor_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([sensor_data])
    
    # Convert activity_level to numeric
    activity_map = {"Rest": 0, "Sleep": 1, "Active": 2}
    df["activity_level"] = df["activity_level"].map(activity_map)
    
    return df

# -------------------------------
# 3. Predict function with CI
# -------------------------------
def predict_event_risks(sensor_data: dict, n_bootstrap: int = 1000):
    """
    Returns event-specific risk probabilities with 95% confidence intervals.
    """
    X_new = prepare_input(sensor_data)
    
    # Predict probabilities from each tree in Random Forest
    all_probs = []
    for estimator in multi_rf.estimators_:
        probs = estimator.predict_proba(X_new)[:, 1]  # prob for class 1
        all_probs.append(probs)
    
    all_probs = np.array(all_probs)  # shape: n_events x n_estimators
    
    results = {}
    for i, event in enumerate(event_cols):
        event_probs = all_probs[i]  # array of probabilities for this event
        mean_prob = np.mean(event_probs)
        lower_ci = np.percentile(event_probs, 2.5)
        upper_ci = np.percentile(event_probs, 97.5)
        results[event] = {
            "probability": round(float(mean_prob), 3),
            "95_CI": (round(float(lower_ci), 3), round(float(upper_ci), 3))
        }
    
    return results

# -------------------------------
# 4. Example usage
# -------------------------------
if __name__ == "__main__":
    example_data = {
        "resting_hr": 78,
        "hr_trend": 0.5,
        "hr_rmssd": 35,
        "tachycardia": 0,
        "cardiac_load": 1.05,
        "avg_spo2": 94,
        "spo2_trend": -0.2,
        "desat_events": 1,
        "resp_strain": 0.83,
        "temp_deviation": 0.5,
        "temp_rise_rate": 0.2,
        "fever_flag": 0,
        "acc_magnitude": 0.03,
        "inactivity_duration_min": 15,
        "fall": 0,
        "post_fall_immobility": 0,
        "activity_level": "Rest"
    }
    
    risks = predict_event_risks(example_data)
    
    print("\nE3. Event-Specific Risk Probabilities\n")
    for event, info in risks.items():
        print(f"{event}: {info['probability']} (95% CI: {info['95_CI'][0]}–{info['95_CI'][1]})")
