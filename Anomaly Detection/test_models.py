"""
test_models.py

Tests patient-specific Isolation Forest models and generates a reports.txt file
with evaluation metrics for each patient.

- Loads models from ml_models/
- Uses a labeled test dataset for evaluation
- Outputs accuracy, precision, recall, F1 score for each patient
"""

import pandas as pd
from joblib import load
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------
# Configuration
# -----------------------
MODEL_DIR = 'ml_models'
TEST_DATASET_PATH = 'test_patient_data.csv'  # Should include patient_id + numeric features + label
NUMERIC_FEATURES = ['HR', 'HRV', 'SpO2', 'Temperature', 'Activity', 'FallEvent']
ALL_FEATURES = ['patient_id'] + NUMERIC_FEATURES
REPORT_FILE = 'reports.txt'

# -----------------------
# Load Patient Model
# -----------------------
def load_patient_model(patient_id):
    patient_id_int = int(patient_id)
    model_path = os.path.join(MODEL_DIR, f'patient_{patient_id_int}_iso_forest.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for patient {patient_id_int} not found: {model_path}")
    return load(model_path)

# -----------------------
# Predict for a Patient
# -----------------------
def predict_patient(model, df):
    """
    df: DataFrame with numeric features only
    Returns: array of predicted labels (0=normal, 1=anomaly)
    """
    preds = model.predict(df)  # 1 = normal, -1 = anomaly
    # Convert to 0=normal, 1=anomaly to match labels
    return (preds == -1).astype(int)

# -----------------------
# Evaluate Models
# -----------------------
def evaluate_models(test_df):
    patients = test_df['patient_id'].unique()
    report_lines = []

    for pid in patients:
        patient_data = test_df[test_df['patient_id'] == pid]
        X = patient_data[NUMERIC_FEATURES]
        y_true = patient_data['label'].astype(int)

        model = load_patient_model(pid)
        y_pred = predict_patient(model, X)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        report_lines.append(f"Patient {pid} Performance:")
        report_lines.append(f"  Accuracy : {acc:.4f}")
        report_lines.append(f"  Precision: {prec:.4f}")
        report_lines.append(f"  Recall   : {rec:.4f}")
        report_lines.append(f"  F1 Score : {f1:.4f}")
        report_lines.append("-" * 40)

    return report_lines

# -----------------------
# Main
# -----------------------
def main():
    if not os.path.exists(TEST_DATASET_PATH):
        raise FileNotFoundError(f"Test dataset not found: {TEST_DATASET_PATH}")
    
    test_df = pd.read_csv(TEST_DATASET_PATH)
    if 'label' not in test_df.columns:
        raise ValueError("Test dataset must include a 'label' column (0=normal, 1=anomaly)")
    
    # Ensure patient_id is int
    test_df['patient_id'] = test_df['patient_id'].astype(int)
    
    print("[INFO] Evaluating models...")
    report_lines = evaluate_models(test_df)
    
    # Save to reports.txt
    with open(REPORT_FILE, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"[INFO] Evaluation complete. Report saved to {REPORT_FILE}")
    print("\n".join(report_lines))

# -----------------------
# Run Script
# -----------------------
if __name__ == "__main__":
    main()
