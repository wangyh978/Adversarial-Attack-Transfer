from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.attacks.base import AttackBase
from src.attacks.common import clip_to_bounds, l2_normalize


class FGMAttack(AttackBase):
    def __init__(self, dataset: str, epsilon: float = 0.5, batch_size: int = 2048):
        super().__init__(dataset=dataset, batch_size=batch_size)
        self.epsilon = float(epsilon)

    def _generate_batch(self, model, x_orig, y_t, min_v, max_v, **kwargs):
        x = x_orig.clone().detach().requires_grad_(True)
        logits = model(x)
        loss = F.cross_entropy(logits, y_t)
        grad = torch.autograd.grad(loss, x)[0]
        grad = l2_normalize(grad)
        x_adv = clip_to_bounds(x + self.epsilon * grad, min_v, max_v)
        return x_adv

    def metadata(self) -> dict:
        return {
            "attack_name": "fgm",
            "epsilon": self.epsilon,
            "batch_size": self.batch_size,
        }
