import os
from joblib import load
from typing import Tuple

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_model", "model.joblib")

_model = None

def load_model():
    global _model
    if _model is None:
        _model = load(MODEL_PATH)
    return _model

def predict_text(text: str) -> Tuple[str, float]:
    """
    Returns (label, probability_of_label) where label is "FAKE" or "REAL".
    """
    model = load_model()
    proba = model.predict_proba([text])[0]
    # classes_ ordering
    classes = model.classes_
    # choose predicted class
    import numpy as np
    idx = int(np.argmax(proba))
    label = classes[idx]
    confidence = float(proba[idx])
    return label, confidence
