"""
predict_anomalies.py

Performs anomaly detection using patient-specific Isolation Forest models
for the Smart Patient Risk Band project.

- Automatically selects the correct model for each patient
- Supports single or batch predictions
- Outputs anomaly_score (severity) and ml_anomaly_flag (binary)
"""

import pandas as pd
from joblib import load
import os

# -----------------------
# Configuration
# -----------------------
MODEL_DIR = 'ml_models'  # Folder containing patient_1_iso_forest.joblib, etc.
NUMERIC_FEATURES = ['HR', 'HRV', 'SpO2', 'Temperature', 'Activity', 'FallEvent']
ALL_FEATURES = ['patient_id'] + NUMERIC_FEATURES  # For DataFrame organization

# -----------------------
# Step 1: Load Patient-Specific Model
# -----------------------
def load_patient_model(patient_id):
    """
    Loads the Isolation Forest model for a specific patient.
    Ensures patient_id is an integer to match the filename.
    """
    patient_id_int = int(patient_id)
    model_path = os.path.join(MODEL_DIR, f'patient_{patient_id_int}_iso_forest.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for patient {patient_id_int} not found: {model_path}")
    return load(model_path)

# -----------------------
# Step 2: Predict Anomalies
# -----------------------
def predict_anomalies(df):
    """
    df: pd.DataFrame with ALL_FEATURES columns
    Returns: DataFrame with anomaly_score and ml_anomaly_flag added
    """
    # Ensure patient_id is integer
    df['patient_id'] = df['patient_id'].astype(int)
    
    results = []

    for _, row in df.iterrows():
        patient_id = row['patient_id']
        model = load_patient_model(patient_id)
        
        # Only pass numeric features to the model (exclude patient_id)
        reading_df = pd.DataFrame([row[NUMERIC_FEATURES].values], columns=NUMERIC_FEATURES)
        
        # Isolation Forest prediction
        preds = model.predict(reading_df)  # 1 = normal, -1 = anomaly
        anomaly_flag = preds[0] == -1
        anomaly_score = -model.decision_function(reading_df)[0]  # Higher = more anomalous
        
        # Append results including patient_id
        results.append({
            **row.to_dict(),
            'anomaly_score': anomaly_score,
            'ml_anomaly_flag': anomaly_flag
        })
    
    return pd.DataFrame(results)

# -----------------------
# Step 3: Main Function (Example Usage)
# -----------------------
def main():
    # Example new readings for prediction
    new_readings = pd.DataFrame([
        {'patient_id': 1, 'HR': 85, 'HRV': 40, 'SpO2': 97, 'Temperature': 36.8, 'Activity': 1, 'FallEvent': 0},
        {'patient_id': 2, 'HR': 120, 'HRV': 10, 'SpO2': 88, 'Temperature': 38.5, 'Activity': 0, 'FallEvent': 0},
        {'patient_id': 3, 'HR': 70, 'HRV': 35, 'SpO2': 96, 'Temperature': 36.9, 'Activity': 1, 'FallEvent': 0}
    ])
    
    results = predict_anomalies(new_readings)
    
    print("[INFO] Anomaly detection results:")
    print(results[['patient_id', 'HR', 'HRV', 'SpO2', 'Temperature', 'Activity', 'FallEvent',
                   'anomaly_score', 'ml_anomaly_flag']])

# -----------------------
# Run Script
# -----------------------
if __name__ == "__main__":
    main()
