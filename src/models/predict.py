from __future__ import annotations

import numpy as np


def predict_labels(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def predict_proba_if_available(model, X: np.ndarray):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None
