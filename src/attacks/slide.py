from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.attacks.base import AttackBase
from src.attacks.common import clip_to_bounds, project_l2_ball, random_l2_noise_like


def _masked_l2_normalize(grad: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Apply a sparse mask and L2-normalize each sample's update direction."""
    g = grad * mask
    flat = g.view(g.size(0), -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(eps)
    return (flat / norm).view_as(g)


class SLIDEAttack(AttackBase):
    """
    SLIDE for tabular NIDS features.

    The previous repository version implemented SLIDE as:
        from src.attacks.pgd import PGDAttack as SLIDEAttack
    which makes PGD and SLIDE identical.

    This implementation makes SLIDE a distinct sparse iterative attack for
    tabular/network-flow features:

    - at each step, select the top-k features by absolute gradient per sample;
    - update only those salient features;
    - use an L2-normalized sparse direction;
    - project the full perturbation back to an L2 ball;
    - finally clip to the valid feature bounds.

    It is intentionally conservative for traffic features: perturbations are
    sparse and bounded, rather than dense PGD-style changes on all features.
    """

    def __init__(
        self,
        dataset: str,
        epsilon: float = 0.5,
        steps: int = 10,
        step_size: float = 0.1,
        topk_ratio: float = 0.25,
        random_start: bool = True,
        seed: int = 2026,
        batch_size: int = 2048,
    ):
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be in (0, 1].")
        super().__init__(dataset=dataset, batch_size=batch_size)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.topk_ratio = float(topk_ratio)
        self.random_start = bool(random_start)
        self.seed = int(seed)

    def _topk_mask(self, grad: torch.Tensor) -> torch.Tensor:
        n_features = grad.size(1)
        k = max(1, int(round(n_features * self.topk_ratio)))
        idx = torch.topk(torch.abs(grad), k=k, dim=1).indices
        mask = torch.zeros_like(grad)
        mask.scatter_(1, idx, 1.0)
        return mask

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
            mask = self._topk_mask(grad)
            direction = _masked_l2_normalize(grad, mask)

            x_next = x_adv.detach() + self.step_size * direction
            delta = project_l2_ball(x_next - x_orig, self.epsilon)
            x_adv = clip_to_bounds(x_orig + delta, min_v, max_v).detach()

        return x_adv

    def metadata(self) -> dict:
        return {
            "attack_name": "slide",
            "epsilon": self.epsilon,
            "steps": self.steps,
            "step_size": self.step_size,
            "topk_ratio": self.topk_ratio,
            "random_start": self.random_start,
            "seed": self.seed,
            "implementation_note": "Sparse L2 iterative tabular SLIDE; not a PGD alias.",
            "batch_size": self.batch_size,
        }
