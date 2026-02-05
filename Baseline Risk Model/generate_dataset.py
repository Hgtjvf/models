import csv
import random

# ========================
# CSV file path
# ========================
csv_file = "baseline_risk_dataset.csv"

# ========================
# CSV columns
# ========================
columns = [
    # Inputs (Derived Physiological Metrics)
    "resting_hr", "hr_trend", "rmssd", "tachycardia", "cardiac_load",
    "avg_spo2", "spo2_trend", "desat_events", "recovery_time", "resp_strain",
    "temp_baseline", "temp_deviation", "temp_rise_rate", "fever",
    "activity_level", "inactivity_minutes", "fall", "post_fall_immobility",
    # Outputs (RiskPrediction)
    "risk_level", "risk_probability", "confidence_score"
]

# ========================
# Activity choices
# ========================
activity_choices = ["rest", "active", "sleep"]

# ========================
# Generate synthetic dataset
# ========================
num_samples = 500  # Adjust as needed

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)

    for _ in range(num_samples):
        # ---- Inputs ----
        resting_hr = random.uniform(50, 100)
        hr_trend = random.uniform(-2, 2)
        rmssd = random.uniform(20, 100)
        tachycardia = 1 if resting_hr > 100 else 0
        cardiac_load = random.uniform(0.8, 1.5)

        avg_spo2 = random.uniform(85, 100)
        spo2_trend = random.uniform(-1, 1)
        desat_events = random.randint(0, 5)
        recovery_time = random.uniform(0, 5)
        resp_strain = resting_hr / avg_spo2

        temp_baseline = random.uniform(36.0, 37.5)
        temp_deviation = random.uniform(-0.5, 1.5)
        temp_rise_rate = random.uniform(-0.2, 0.5)
        fever = 1 if temp_baseline + temp_deviation > 38 else 0

        activity_level = random.choice(activity_choices)
        inactivity_minutes = random.randint(0, 120)
        fall = random.choice([0, 1])
        post_fall_immobility = random.choice([0, 1]) if fall else 0

        # ---- Outputs ----
        # Simple risk calculation for simulation
        risk_score = 0
        if tachycardia or fever or avg_spo2 < 92 or fall:
            risk_score = 2  # High
        elif desat_events > 1 or temp_rise_rate > 0.3:
            risk_score = 1  # Moderate
        else:
            risk_score = 0  # Low

        risk_probability = random.uniform(0.0, 1.0)
        confidence_score = random.uniform(0.6, 0.99)

        row = [
            resting_hr, hr_trend, rmssd, tachycardia, cardiac_load,
            avg_spo2, spo2_trend, desat_events, recovery_time, resp_strain,
            temp_baseline, temp_deviation, temp_rise_rate, fever,
            activity_level, inactivity_minutes, fall, post_fall_immobility,
            risk_score, risk_probability, confidence_score
        ]

        writer.writerow(row)

print(f"Synthetic CSV dataset generated: {csv_file}")
