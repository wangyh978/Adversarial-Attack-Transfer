from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.attacks.base import AttackBase
from src.attacks.common import load_feature_bounds, clip_to_bounds


class CWAttack(AttackBase):
    """
    表格特征场景下的简化版 untargeted C&W L2。
    这是工程近似实现，便于先跑通完整链路。
    """
    def __init__(self, dataset: str, c: float = 0.1, steps: int = 50, lr: float = 1e-2):
        self.dataset = dataset
        self.c = c
        self.steps = steps
        self.lr = lr

    def generate(self, model, X, y, **kwargs):
        device = next(model.parameters()).device
        x_orig = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        delta = torch.zeros_like(x_orig, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        min_v, max_v = load_feature_bounds(self.dataset)
        min_v = min_v.to(device)
        max_v = max_v.to(device)

        for _ in range(self.steps):
            x_adv = clip_to_bounds(x_orig + delta, min_v, max_v)
            logits = model(x_adv)

            real = logits.gather(1, y_t.view(-1, 1)).squeeze(1)
            tmp = logits.clone()
            tmp[torch.arange(len(y_t)), y_t] = -1e9
            other = tmp.max(dim=1).values

            f_term = torch.relu(real - other)
            l2_term = torch.sum((x_adv - x_orig) ** 2, dim=1)
            loss = (l2_term + self.c * f_term).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x_adv = clip_to_bounds(x_orig + delta.detach(), min_v, max_v)
        return x_adv.cpu().numpy().astype(np.float32), {
            "attack_name": "cw",
            "c": self.c,
            "steps": self.steps,
            "lr": self.lr,
        }
