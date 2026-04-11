from __future__ import annotations

import numpy as np


def compute_agreement(y_ref: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    assert len(y_ref) == len(y_pred)
    return {"target_agreement": float((y_ref == y_pred).mean())}
