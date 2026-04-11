from __future__ import annotations

from itertools import product
from typing import Iterable
import pandas as pd


def build_surrogate_grid(
    dataset: str,
    target_model: str,
    seed_sizes: Iterable[int] = (500, 1000, 2000),
    alphas: Iterable[float] = (0.1, 0.2, 0.5),
    depths: Iterable[int] = (3, 5, 7),
) -> pd.DataFrame:
    rows = []
    for seed_size, alpha, depth in product(seed_sizes, alphas, depths):
        rows.append({
            "dataset": dataset,
            "target_model": target_model,
            "seed_size": seed_size,
            "alpha": alpha,
            "depth": depth,
        })
    return pd.DataFrame(rows)
