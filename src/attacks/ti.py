from __future__ import annotations

import torch
import torch.nn.functional as F

from src.attacks.base import AttackBase
from src.attacks.common import (
    clip_to_bounds,
    mean_abs_normalize,
    project_linf_ball,
    random_linf_noise_like,
    smooth_1d,
)


class TIAttack(AttackBase):
    def __init__(
        self,
        dataset: str,
        epsilon: float = 0.5,
        steps: int = 12,
        step_size: float = 0.08,
        decay: float = 1.0,
        kernel_size: int = 5,
        kernel_sigma: float = 1.0,
        random_start: bool = False,
        multi_scale: bool = True,
        seed: int = 2026,
        batch_size: int = 2048,
    ):
        super().__init__(dataset=dataset, batch_size=batch_size)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.decay = float(decay)
        self.kernel_size = int(kernel_size)
        self.kernel_sigma = float(kernel_sigma)
        self.random_start = bool(random_start)
        self.multi_scale = bool(multi_scale)
        self.seed = int(seed)

    def _smooth_grad(self, grad: torch.Tensor) -> torch.Tensor:
        smoothed = smooth_1d(grad, kernel_size=self.kernel_size, sigma=self.kernel_sigma)
        if not self.multi_scale:
            return smoothed

        kernels = []
        if self.kernel_size > 3:
            kernels.append((max(3, self.kernel_size - 2) | 1, max(0.6, self.kernel_sigma * 0.75)))
        kernels.append((self.kernel_size, self.kernel_sigma))
        kernels.append((self.kernel_size + 2, self.kernel_sigma * 1.25))

        blended = torch.zeros_like(grad)
        for size, sigma in kernels:
            blended = blended + smooth_1d(grad, kernel_size=size, sigma=sigma)
        return blended / len(kernels)

    def _generate_batch(self, model, x_orig, y_t, min_v, max_v, **kwargs):
        x_adv = x_orig.clone().detach()
        momentum = torch.zeros_like(x_adv)

        if self.random_start and self.epsilon > 0:
            generator = torch.Generator(device=x_orig.device)
            generator.manual_seed(self.seed)
            noise = random_linf_noise_like(x_orig, self.epsilon, generator=generator)
            x_adv = clip_to_bounds(x_orig + noise, min_v, max_v).detach()

        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y_t)
            grad = torch.autograd.grad(loss, x_adv)[0]
            grad = self._smooth_grad(grad)
            grad = mean_abs_normalize(grad)
            momentum = self.decay * momentum + grad

            x_adv = x_adv.detach() + self.step_size * torch.sign(momentum)
            delta = project_linf_ball(x_adv - x_orig, self.epsilon)
            x_adv = clip_to_bounds(x_orig + delta, min_v, max_v).detach()

        return x_adv

    def metadata(self) -> dict:
        return {
            "attack_name": "ti",
            "epsilon": self.epsilon,
            "steps": self.steps,
            "step_size": self.step_size,
            "decay": self.decay,
            "kernel_size": self.kernel_size,
            "kernel_sigma": self.kernel_sigma,
            "random_start": self.random_start,
            "multi_scale": self.multi_scale,
            "seed": self.seed,
            "batch_size": self.batch_size,
        }
