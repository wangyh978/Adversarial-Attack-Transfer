from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import torch

from src.utils.io import load_json


def _sort_feature_cols(cols: list[str]) -> list[str]:
    def key_fn(name: str):
        try:
            return int(name.split("_")[1])
        except Exception:
            return name
    return sorted(cols, key=key_fn)


def load_feature_bounds(dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    info_path = Path("artifacts/preprocessors") / f"{dataset}_feature_info.json"
    info = load_json(info_path)

    # 已有缓存，直接返回
    if "feature_min" in info and "feature_max" in info:
        min_v = torch.tensor(info["feature_min"], dtype=torch.float32)
        max_v = torch.tensor(info["feature_max"], dtype=torch.float32)
        return min_v, max_v

    # 否则从训练特征现算
    train_path = Path("data") / dataset / "processed" / "train_features.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing feature bounds in {info_path}, and train feature file not found: {train_path}"
        )

    df = pd.read_parquet(train_path)

    # 优先使用 feature_info.json 里的特征顺序
    feature_cols = info.get("feature_names")
    if not feature_cols:
        feature_cols = [c for c in df.columns if c.startswith("f_")]
        feature_cols = _sort_feature_cols(feature_cols)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Feature columns missing in {train_path}: {missing_cols[:10]}"
        )

    feature_min = df[feature_cols].min(axis=0).astype(float).tolist()
    feature_max = df[feature_cols].max(axis=0).astype(float).tolist()

    # 写回缓存，后续攻击直接复用
    info["feature_min"] = feature_min
    info["feature_max"] = feature_max
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    min_v = torch.tensor(feature_min, dtype=torch.float32)
    max_v = torch.tensor(feature_max, dtype=torch.float32)
    return min_v, max_v


def clip_to_bounds(x: torch.Tensor, min_v: torch.Tensor, max_v: torch.Tensor) -> torch.Tensor:
    min_v = min_v.to(x.device)
    max_v = max_v.to(x.device)
    return torch.max(torch.min(x, max_v), min_v)


def l2_normalize(grad: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return grad / norm.view(-1, *([1] * (grad.dim() - 1)))
