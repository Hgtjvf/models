# ============================================================
# SMART PATIENT RISK BAND — ADVANCED MODEL EVALUATION REPORT
# Healthcare-focused evaluation
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# SECTION 1 — PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "smart_patient_risk_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "testing_dataset.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================
# SECTION 2 — LOAD MODEL & DATA
# ============================================================
print("📦 Loading model and dataset...")

data = joblib.load(MODEL_PATH)

risk_model = data["risk_model"]
label_encoder = data["label_encoder"]
FEATURES = data["features"]

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y_true = label_encoder.transform(df["risk_level"])
y_pred = risk_model.predict(X)

class_names = label_encoder.classes_

# ============================================================
# SECTION 3 — BASIC METRICS
# ============================================================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)

cm = confusion_matrix(y_true, y_pred)

# ============================================================
# SECTION 4 — HEALTHCARE CRITICAL METRICS
# ============================================================
# False Negatives per class
false_negatives = cm.sum(axis=1) - np.diag(cm)

# False Positive per class
false_positives = cm.sum(axis=0) - np.diag(cm)

# Support (how many samples per class)
support = cm.sum(axis=1)

# False Negative Rate (very important in healthcare)
fn_rate = false_negatives / support

# ============================================================
# SECTION 5 — SAVE IMPROVED REPORT
# ============================================================
report_path = os.path.join(REPORT_DIR, "evaluation_report.txt")

with open(report_path, "w") as f:
    f.write("SMART PATIENT RISK MODEL — HEALTHCARE EVALUATION REPORT\n")
    f.write("=" * 65 + "\n\n")

    f.write("🔹 OVERALL PERFORMANCE\n")
    f.write(f"Accuracy : {accuracy:.4f}\n\n")

    f.write("🔹 PER-CLASS PRECISION & RECALL\n")
    for i, name in enumerate(class_names):
        f.write(
            f"{name.upper():<10} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f}\n"
        )

    f.write("\n")

    f.write("🔹 CONFUSION MATRIX\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("🔹 FALSE NEGATIVES (Healthcare Critical)\n")
    for i, name in enumerate(class_names):
        f.write(
            f"{name.upper():<10} | FN: {false_negatives[i]} | FN Rate: {fn_rate[i]:.4f}\n"
        )

    f.write("\n")

    f.write("🔹 FALSE POSITIVES (False Alerts)\n")
    for i, name in enumerate(class_names):
        f.write(
            f"{name.upper():<10} | FP: {false_positives[i]}\n"
        )

    f.write("\n")

    f.write("🔹 DETAILED CLASSIFICATION REPORT\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names))

    f.write("\n")
    f.write("=" * 65 + "\n")
    f.write("Healthcare Note:\n")
    f.write(
        "Low False Negatives for HIGH risk class is the most critical metric.\n"
        "If FN for HIGH risk is large, the model is unsafe for deployment.\n"
    )

print("\n✅ Advanced evaluation report generated.")
print("📁 Saved at:", report_path)
