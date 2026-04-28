from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from src.attacks.common import load_feature_bounds


class AttackBase(ABC):
    def __init__(self, dataset: str, batch_size: int = 2048):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self._cached_bounds: tuple[torch.Tensor, torch.Tensor] | None = None

    def _feature_bounds(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_bounds is None:
            self._cached_bounds = load_feature_bounds(self.dataset)

        min_v, max_v = self._cached_bounds
        return min_v.to(device=device, dtype=dtype), max_v.to(device=device, dtype=dtype)

    def metadata(self) -> dict:
        return {}

    def generate(self, model, X, y, **kwargs):
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        try:
            device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError("Attack model must expose parameters so the device can be inferred.") from exc
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y)

        outputs: list[np.ndarray] = []
        min_v, max_v = self._feature_bounds(device, torch.float32)

        model.eval()
        for start in range(0, len(X_np), self.batch_size):
            end = start + self.batch_size
            x_batch = torch.as_tensor(X_np[start:end], dtype=torch.float32, device=device)
            y_batch = torch.as_tensor(y_np[start:end], dtype=torch.long, device=device)
            x_adv = self._generate_batch(model, x_batch, y_batch, min_v, max_v, **kwargs)
            outputs.append(x_adv.detach().cpu().numpy().astype(np.float32, copy=False))

        if not outputs:
            return np.empty((0, 0), dtype=np.float32), self.metadata()

        return np.concatenate(outputs, axis=0), self.metadata()

    @abstractmethod
    def _generate_batch(
        self,
        model,
        x_orig: torch.Tensor,
        y_t: torch.Tensor,
        min_v: torch.Tensor,
        max_v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
