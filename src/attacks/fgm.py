from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.attacks.base import AttackBase
from src.attacks.common import load_feature_bounds, clip_to_bounds, l2_normalize


class FGMAttack(AttackBase):
    def __init__(self, dataset: str, epsilon: float = 0.5):
        self.dataset = dataset
        self.epsilon = epsilon

    def generate(self, model, X, y, **kwargs):
        device = next(model.parameters()).device
        x = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        logits = model(x)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()

        grad = l2_normalize(x.grad)
        min_v, max_v = load_feature_bounds(self.dataset)
        x_adv = clip_to_bounds(x + self.epsilon * grad, min_v, max_v)

        return x_adv.detach().cpu().numpy().astype(np.float32), {
            "attack_name": "fgm",
            "epsilon": self.epsilon,
        }
