from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys


@dataclass
class SurrogateJob:
    dataset: str
    target_model: str
    seed_size: int
    alpha: float
    depth: int

    def train(self) -> None:
        subprocess.run(
            [
                sys.executable, 
                "-m",
                "src.models.train_surrogate_mlp",
                "--dataset", self.dataset,
                "--target_model", self.target_model,
                "--seed_size", str(self.seed_size),
                "--alpha", str(self.alpha),
                "--depth", str(self.depth),
            ],
            check=True,
        )

    def evaluate(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.evaluation.evaluate_surrogate",
                "--dataset", self.dataset,
                "--target_model", self.target_model,
                "--seed_size", str(self.seed_size),
                "--alpha", str(self.alpha),
                "--depth", str(self.depth),
            ],
            check=True,
        )
