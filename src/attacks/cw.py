from __future__ import annotations

import torch

from src.attacks.base import AttackBase
from src.attacks.common import clip_to_bounds


class CWAttack(AttackBase):
    """
    Untargeted C&W-style L2 attack for tabular features.

    This version keeps feature values inside the observed model-input bounds,
    tracks the best successful adversarial sample for every item in the batch,
    and performs a small binary search over the confidence constant.
    """

    def __init__(
        self,
        dataset: str,
        c: float = 0.01,
        steps: int = 20,
        lr: float = 5e-3,
        confidence: float = 0.0,
        binary_search_steps: int = 3,
        abort_early: bool = True,
        batch_size: int = 1024,
    ):
        super().__init__(dataset=dataset, batch_size=batch_size)
        self.c = float(c)
        self.steps = int(steps)
        self.lr = float(lr)
        self.confidence = float(confidence)
        self.binary_search_steps = int(binary_search_steps)
        self.abort_early = bool(abort_early)

    def _objective(self, logits: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        real = logits.gather(1, y_t.view(-1, 1)).squeeze(1)
        mask = torch.nn.functional.one_hot(y_t, num_classes=logits.size(1)).bool()
        other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
        return torch.clamp(real - other + self.confidence, min=0.0)

    def _generate_batch(self, model, x_orig, y_t, min_v, max_v, **kwargs):
        batch_size = x_orig.size(0)
        best_adv = x_orig.clone()
        best_l2 = torch.full((batch_size,), float("inf"), device=x_orig.device, dtype=x_orig.dtype)
        found_adv = torch.zeros(batch_size, dtype=torch.bool, device=x_orig.device)

        lower = torch.zeros(batch_size, dtype=x_orig.dtype, device=x_orig.device)
        upper = torch.full((batch_size,), 1e10, dtype=x_orig.dtype, device=x_orig.device)
        const = torch.full((batch_size,), self.c, dtype=x_orig.dtype, device=x_orig.device)

        for _ in range(max(1, self.binary_search_steps)):
            delta = torch.zeros_like(x_orig, requires_grad=True)
            optimizer = torch.optim.Adam([delta], lr=self.lr)

            local_best_adv = x_orig.clone()
            local_best_l2 = torch.full_like(best_l2, float("inf"))
            local_found = torch.zeros_like(found_adv)
            previous_loss: float | None = None

            for step in range(self.steps):
                x_adv = clip_to_bounds(x_orig + delta, min_v, max_v)
                logits = model(x_adv)
                l2 = torch.sum((x_adv - x_orig) ** 2, dim=1)
                f_term = self._objective(logits, y_t)
                loss = (l2 + const * f_term).sum()

                optimizer.zero_grad()
                grad = torch.autograd.grad(loss, delta)[0]
                delta.grad = grad
                optimizer.step()

                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    success = preds != y_t
                    improved = success & (l2 < local_best_l2)
                    local_best_l2 = torch.where(improved, l2, local_best_l2)
                    local_best_adv = torch.where(improved.unsqueeze(1), x_adv.detach(), local_best_adv)
                    local_found = local_found | success

                if self.abort_early and step % max(1, self.steps // 10) == 0:
                    loss_value = float(loss.item())
                    if previous_loss is not None and abs(previous_loss - loss_value) <= max(1e-6, previous_loss * 1e-4):
                        break
                    previous_loss = loss_value

            improved_global = local_found & (local_best_l2 < best_l2)
            best_l2 = torch.where(improved_global, local_best_l2, best_l2)
            best_adv = torch.where(improved_global.unsqueeze(1), local_best_adv, best_adv)
            found_adv = found_adv | local_found

            upper = torch.where(local_found, torch.minimum(upper, const), upper)
            lower = torch.where(~local_found, torch.maximum(lower, const), lower)
            const = torch.where(
                upper < 1e9,
                (lower + upper) / 2.0,
                const * 10.0,
            )

        return torch.where(found_adv.unsqueeze(1), best_adv, x_orig)

    def metadata(self) -> dict:
        return {
            "attack_name": "cw",
            "c": self.c,
            "steps": self.steps,
            "lr": self.lr,
            "confidence": self.confidence,
            "binary_search_steps": self.binary_search_steps,
            "abort_early": self.abort_early,
            "batch_size": self.batch_size,
        }
