from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def load_seed_only_dataset(dataset: str, target_model: str, seed_size: int):
    path = Path("data/surrogate_train") / dataset / f"{target_model}_seed_{seed_size}_seed_only.parquet"
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if str(c).startswith("f_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["target_label"].to_numpy()
    return X, y, feature_cols


def load_mixup_dataset(dataset: str, target_model: str, seed_size: int, alpha: float):
    path = Path("data/surrogate_train") / dataset / f"{target_model}_seed_{seed_size}_alpha_{alpha}_mixup.parquet"
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if str(c).startswith("f_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_a = df["y_a"].to_numpy()
    y_b = df["y_b"].to_numpy()
    lam = df["lam"].to_numpy(dtype=np.float32)
    return X, y_a, y_b, lam, feature_cols
