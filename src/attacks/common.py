from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from src.utils.io import load_json


def load_feature_bounds(dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    info = load_json(Path("artifacts/preprocessors") / f"{dataset}_feature_info.json")
    min_v = torch.tensor(info["feature_min"], dtype=torch.float32)
    max_v = torch.tensor(info["feature_max"], dtype=torch.float32)
    return min_v, max_v


def clip_to_bounds(x: torch.Tensor, min_v: torch.Tensor, max_v: torch.Tensor) -> torch.Tensor:
    min_v = min_v.to(x.device)
    max_v = max_v.to(x.device)
    return torch.max(torch.min(x, max_v), min_v)


def l2_normalize(grad: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return grad / norm.view(-1, *([1] * (grad.dim() - 1)))
