from pathlib import Path
from joblib import load
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "testing_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
REPORT_PATH = BASE_DIR / "reports" / "model_evaluation.txt"

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# -------------------------------------------------
# Load trained model bundle
# -------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

bundle = load(MODEL_PATH)

model = bundle["model"]
label_encoder = bundle["label_encoder"]
features = bundle["features"]

# -------------------------------------------------
# Prepare X and y (EXACT training logic)
# -------------------------------------------------
X = df[features]
y = df["risk_level"]

y_encoded = label_encoder.transform(y)

# -------------------------------------------------
# Train-test split (evaluation only)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -------------------------------------------------
# Predict on test data
# -------------------------------------------------
y_pred = model.predict(X_test)

# -------------------------------------------------
# Metrics
# -------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

# -------------------------------------------------
# Save evaluation report
# -------------------------------------------------
os.makedirs(BASE_DIR / "reports", exist_ok=True)

with open(REPORT_PATH, "w") as f:
    f.write("Random Forest Health Risk Model – Evaluation Report\n")
    f.write("=================================================\n\n")

    f.write(f"Accuracy: {accuracy:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\n")

    f.write("Classification Report:\n")
    f.write(report)

# -------------------------------------------------
# Console output
# -------------------------------------------------
print("\n✅ Model Evaluation Complete")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
print(f"\n📄 Report saved at: {REPORT_PATH}")
