from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _as_float_tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Per-sample L2 normalization used by FGM/PGD-like attacks.

    Keeps the original public helper expected by src.attacks.fgm and other
    attack modules.
    """
    flat = x.view(x.size(0), -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(eps)
    return (flat / norm).view_as(x)


def _write_feature_bounds(dataset: str, feature_min: np.ndarray, feature_max: np.ndarray) -> None:
    info_path = Path("artifacts/preprocessors") / f"{dataset}_feature_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)

    info = {}
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            try:
                info = json.load(f)
            except Exception:
                info = {}

    info["feature_min"] = feature_min.astype(float).tolist()
    info["feature_max"] = feature_max.astype(float).tolist()
    info["feature_bounds_source"] = f"data/{dataset}/processed/X_train.npy"
    info["feature_bounds_note"] = (
        "Bounds are computed from the exact model-input feature matrix. "
        "This avoids mixing raw feature scales with preprocessed model-input scales."
    )

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def load_feature_bounds(dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load valid feature bounds for adversarial attacks.

    The attack operates on model-input arrays X_train/X_test, so bounds should
    come from data/<dataset>/processed/X_train.npy whenever possible.
    """

    x_train_path = Path("data") / dataset / "processed" / "X_train.npy"
    if x_train_path.exists():
        x_train = np.load(x_train_path).astype(np.float32)
        if x_train.ndim != 2:
            raise ValueError(f"Expected 2D X_train array, got shape={x_train.shape}")
        feature_min = np.nanmin(x_train, axis=0)
        feature_max = np.nanmax(x_train, axis=0)
        _write_feature_bounds(dataset, feature_min, feature_max)
        return _as_float_tensor(feature_min), _as_float_tensor(feature_max)

    info_path = Path("artifacts/preprocessors") / f"{dataset}_feature_info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        if "feature_min" in info and "feature_max" in info:
            return _as_float_tensor(info["feature_min"]), _as_float_tensor(info["feature_max"])

    raise FileNotFoundError(
        f"Cannot find feature bounds for {dataset}. Expected {x_train_path} "
        f"or artifacts/preprocessors/{dataset}_feature_info.json with feature_min/feature_max."
    )


def clip_to_bounds(x: torch.Tensor, min_v: torch.Tensor, max_v: torch.Tensor) -> torch.Tensor:
    min_v = min_v.to(device=x.device, dtype=x.dtype)
    max_v = max_v.to(device=x.device, dtype=x.dtype)

    while min_v.dim() < x.dim():
        min_v = min_v.unsqueeze(0)
        max_v = max_v.unsqueeze(0)

    return torch.max(torch.min(x, max_v), min_v)
