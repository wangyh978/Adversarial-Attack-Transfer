from __future__ import annotations

import numpy as np


def sample_lambda(alpha: float, size: int) -> np.ndarray:
    lam = np.random.beta(alpha, alpha, size=size)
    return lam.astype(np.float32)


def mixup_features(X: np.ndarray, y: np.ndarray, alpha: float = 0.1):
    assert len(X) == len(y)
    assert alpha > 0

    n = len(X)
    indices = np.random.permutation(n)
    lam = sample_lambda(alpha, n).reshape(-1, 1)

    X_a = X
    X_b = X[indices]
    y_a = y
    y_b = y[indices]

    X_mix = lam * X_a + (1.0 - lam) * X_b
    return X_mix.astype(np.float32), y_a, y_b, lam.squeeze(-1)
