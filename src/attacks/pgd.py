from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.attacks.base import AttackBase
from src.attacks.common import clip_to_bounds, l2_normalize, project_l2_ball, random_l2_noise_like


class PGDAttack(AttackBase):
    def __init__(
        self,
        dataset: str,
        epsilon: float = 0.5,
        steps: int = 10,
        step_size: float = 0.1,
        random_start: bool = False,
        seed: int = 2026,
        batch_size: int = 2048,
    ):
        super().__init__(dataset=dataset, batch_size=batch_size)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.random_start = bool(random_start)
        self.seed = int(seed)

    def _generate_batch(self, model, x_orig, y_t, min_v, max_v, **kwargs):
        x_adv = x_orig.clone().detach()

        if self.random_start and self.epsilon > 0:
            generator = torch.Generator(device=x_orig.device)
            generator.manual_seed(self.seed)
            noise = random_l2_noise_like(x_orig, self.epsilon, generator=generator)
            x_adv = clip_to_bounds(x_orig + noise, min_v, max_v).detach()

        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y_t)
            grad = torch.autograd.grad(loss, x_adv)[0]
            grad = l2_normalize(grad)
            x_adv = x_adv.detach() + self.step_size * grad

            delta = project_l2_ball(x_adv - x_orig, self.epsilon)
            x_adv = clip_to_bounds(x_orig + delta, min_v, max_v).detach()

        return x_adv

    def metadata(self) -> dict:
        return {
            "attack_name": "pgd",
            "epsilon": self.epsilon,
            "steps": self.steps,
            "step_size": self.step_size,
            "random_start": self.random_start,
            "seed": self.seed,
            "batch_size": self.batch_size,
        }
