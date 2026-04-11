from __future__ import annotations

import numpy as np


def batched_predict_label(blackbox, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    outputs = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        outputs.append(blackbox.predict_label(X[start:end]))
    return np.concatenate(outputs, axis=0)


def batched_predict_proba(blackbox, X: np.ndarray, batch_size: int = 256):
    outputs = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        outputs.append(blackbox.predict_proba(X[start:end]))
    return np.concatenate(outputs, axis=0)
