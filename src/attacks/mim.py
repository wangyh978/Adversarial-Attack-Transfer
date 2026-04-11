from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.attacks.base import AttackBase
from src.attacks.common import load_feature_bounds, clip_to_bounds


class MIMAttack(AttackBase):
    def __init__(self, dataset: str, epsilon: float = 0.5, steps: int = 10, step_size: float = 0.1, decay: float = 1.0):
        self.dataset = dataset
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.decay = decay

    def generate(self, model, X, y, **kwargs):
        device = next(model.parameters()).device
        x_orig = torch.tensor(X, dtype=torch.float32, device=device)
        x_adv = x_orig.clone().detach()
        y_t = torch.tensor(y, dtype=torch.long, device=device)
        momentum = torch.zeros_like(x_adv)

        min_v, max_v = load_feature_bounds(self.dataset)
        min_v = min_v.to(device)
        max_v = max_v.to(device)

        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()

            grad = x_adv.grad
            grad_norm = grad / (torch.mean(torch.abs(grad), dim=1, keepdim=True) + 1e-12)
            momentum = self.decay * momentum + grad_norm
            x_adv = x_adv.detach() + self.step_size * torch.sign(momentum)

            delta = torch.clamp(x_adv - x_orig, min=-self.epsilon, max=self.epsilon)
            x_adv = clip_to_bounds(x_orig + delta, min_v, max_v).detach()

        return x_adv.cpu().numpy().astype(np.float32), {
            "attack_name": "mim",
            "epsilon": self.epsilon,
            "steps": self.steps,
            "step_size": self.step_size,
            "decay": self.decay,
        }
