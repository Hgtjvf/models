import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ========================
# Load the CSV dataset
# ========================
csv_file = "baseline_risk_dataset.csv"
data = pd.read_csv(csv_file)

# ========================
# Define inputs and outputs
# ========================
X = data.drop(columns=["risk_level", "risk_probability", "confidence_score"])
y = data["risk_level"]

# Convert categorical column 'activity_level' to numeric
# Keep all categories to avoid missing/unseen features later
X = pd.get_dummies(X, columns=["activity_level"], drop_first=False)

# ========================
# Train-test split
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================
# Train Random Forest
# ========================
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight="balanced"  # automatically adjusts for class imbalance
)
rf_model.fit(X_train, y_train)

# ========================
# Evaluate model
# ========================
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ========================
# Save model as .joblib
# ========================
model_file = "baseline_risk_rf_model.joblib"
joblib.dump(rf_model, model_file)

print(f"Random Forest model saved as: {model_file}")

# ========================
# Save column names for later prediction
# ========================
columns_file = "baseline_risk_model_columns.joblib"
joblib.dump(X_train.columns.tolist(), columns_file)
print(f"Feature columns saved as: {columns_file}")
