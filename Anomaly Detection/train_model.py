"""
train_model.py

Trains patient-specific anomaly detection models using Isolation Forest
for the Smart Patient Risk Band project.

- Loads simulated or real patient dataset
- Trains one Isolation Forest model per patient
- Saves models as .joblib for later inference
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump
import os

# -----------------------
# Configuration
# -----------------------
DATASET_PATH = 'simulated_patient_data.csv'  # Path to dataset CSV
MODEL_DIR = 'ml_models'  # Directory to save trained models
RANDOM_STATE = 42
CONTAMINATION = 0.01  # Expected fraction of anomalies

# Features to use for anomaly detection
FEATURES = ['HR', 'HRV', 'SpO2', 'Temperature', 'Activity', 'FallEvent']

# -----------------------
# Step 1: Load Dataset
# -----------------------
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file {path} not found.")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# -----------------------
# Step 2: Train Patient-Specific Models
# -----------------------
def train_models(df, features, contamination=CONTAMINATION, random_state=RANDOM_STATE):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"[INFO] Created model directory: {MODEL_DIR}")

    patient_models = {}

    for pid in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == pid][features]
        
        # Initialize Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=random_state)
        model.fit(patient_data)
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f'patient_{pid}_iso_forest.joblib')
        dump(model, model_path)
        print(f"[INFO] Trained and saved model for patient {pid} -> {model_path}")

        # Keep in dictionary for reference
        patient_models[pid] = model
    
    return patient_models

# -----------------------
# Step 3: Main Function
# -----------------------
def main():
    print("[INFO] Loading dataset...")
    df = load_dataset(DATASET_PATH)
    
    print("[INFO] Training patient-specific Isolation Forest models...")
    models = train_models(df, FEATURES)
    
    print("[INFO] Training complete for all patients.")
    print(f"[INFO] Models saved in folder: {MODEL_DIR}")

# Run script
if __name__ == "__main__":
    main()
