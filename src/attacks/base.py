from __future__ import annotations


class AttackBase:
    def generate(self, model, X, y, **kwargs):
        raise NotImplementedError
