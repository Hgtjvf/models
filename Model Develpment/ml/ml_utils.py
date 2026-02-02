from pathlib import Path
import numpy as np
import joblib
import pandas as pd

# -------------------------------------------------
# Load model ONCE
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model.joblib"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
label_encoder = bundle["label_encoder"]
FEATURES = bundle["features"]

# -------------------------------------------------
# Prediction function (Django-safe)
# -------------------------------------------------
def predict_risk(input_data: dict) -> str:
    """
    input_data: dict with keys = FEATURES
    """

    # Convert to DataFrame (avoids sklearn warnings)
    df = pd.DataFrame([input_data], columns=FEATURES)

    prediction_encoded = model.predict(df)[0]
    prediction_label = label_encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    return prediction_label
