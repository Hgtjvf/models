import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix

# =========================
# Load the trained model and feature columns
# =========================
rf_model = joblib.load("baseline_risk_rf_model.joblib")
feature_columns = joblib.load("baseline_risk_model_columns.joblib")

# =========================
# Load the test dataset
# =========================
df_test = pd.read_csv("baseline_risk_test_dataset.csv")

# Separate features and true labels
X_test = df_test.drop(columns=["risk_level", "risk_probability", "confidence_score"], errors='ignore')
y_true = df_test["risk_level"]

# Ensure all training columns are present in test data
for col in feature_columns:
    if col not in X_test.columns:
        X_test[col] = 0

# Reorder columns to match training data
X_test = X_test[feature_columns]

# =========================
# Make predictions
# =========================
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)
confidence_scores = np.max(y_proba, axis=1)

# =========================
# Compute metrics
# =========================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

# False alerts: predicted risk_level >=1 but actual risk_level = 0
false_alerts = np.sum((y_pred >= 1) & (y_true == 0))

# =========================
# Feature importances (overall top 5)
# =========================
importances = rf_model.feature_importances_
top_features = pd.Series(importances, index=feature_columns).sort_values(ascending=False).head(5)

# =========================
# Write report to file
# =========================
with open("reports.txt", "w") as f:
    f.write("=== Baseline Risk Model Evaluation Report ===\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision (weighted): {precision:.4f}\n")
    f.write(f"Recall (weighted): {recall:.4f}\n")
    f.write(f"False Alerts: {false_alerts}\n\n")
    
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred, zero_division=0))
    f.write("\n\n")
    
    f.write("Top Contributing Features (overall):\n")
    for feature, importance in top_features.items():
        f.write(f"{feature}: {importance:.4f}\n")
    f.write("\n")
    
    # Example: show risk probability & confidence for first 5 samples
    f.write("Example Predictions (first 5 samples):\n")
    for i in range(min(5, len(y_pred))):
        f.write(
            f"Sample {i+1} | Predicted risk_level: {y_pred[i]} | "
            f"Confidence score: {confidence_scores[i]:.4f} | "
            f"Risk probabilities: {y_proba[i]}\n"
        )

print("Evaluation complete! Report saved as reports.txt")
