import os
from joblib import load
from typing import Tuple

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_model", "model.joblib")

_model_obj = None

def load_model():
    global _model_obj
    if _model_obj is None:
        _model_obj = load(MODEL_PATH)
    return _model_obj

def predict_text(text: str) -> Tuple[str, float]:
    """
    Returns (label, probability_of_label) where label is "FAKE" or "REAL".
    """
    obj = load_model()
    pipe = obj["pipeline"]
    le = obj["label_encoder"]
    proba = pipe.predict_proba([text])[0]  # probabilities for each encoded label index
    import numpy as np
    idx = int(np.argmax(proba))
    label = le.inverse_transform([idx])[0]
    confidence = float(proba[idx])
    return label, confidence
