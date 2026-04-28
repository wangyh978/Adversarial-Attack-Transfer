from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path


@dataclass(frozen=True)
class SurrogateConfig:
    dataset: str
    target_model: str
    seed_size: int
    alpha: float
    depth: int
    model_path: str | None = None
    source: str | None = None


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def surrogate_model_path(dataset: str, target_model: str, seed_size: int, alpha: float, depth: int) -> Path:
    return Path("artifacts/models") / f"surrogate_{dataset}_{target_model}_seed{seed_size}_a{alpha}_d{depth}.pt"


def infer_best_surrogate_config(dataset: str, target_model: str) -> SurrogateConfig:
    candidate_jsons = [
        Path("artifacts/metadata") / f"best_surrogate_sweep_{dataset}_{target_model}.json",
        Path("artifacts/metadata") / f"best_surrogate_{dataset}_{target_model}.json",
    ]

    for path in candidate_jsons:
        data = _read_json(path)
        if not data:
            continue

        config = SurrogateConfig(
            dataset=dataset,
            target_model=target_model,
            seed_size=int(data["seed_size"]),
            alpha=float(data["alpha"]),
            depth=int(data["depth"]),
            model_path=data.get("model_path"),
            source=str(path),
        )
        return config

    candidates = sorted(Path("artifacts/models").glob(f"surrogate_{dataset}_{target_model}_seed*_a*_d*.pt"))
    if not candidates:
        raise FileNotFoundError("No surrogate model file and no best surrogate config found.")

    preferred = None
    for p in candidates:
        if "_seed1000_a0.1_d3.pt" in p.name:
            preferred = p
            break
    if preferred is None:
        preferred = candidates[-1]

    match = re.search(r"seed(\d+)_a([0-9.]+)_d(\d+)\.pt$", preferred.name)
    if not match:
        raise ValueError(f"Cannot parse surrogate config from filename: {preferred.name}")

    return SurrogateConfig(
        dataset=dataset,
        target_model=target_model,
        seed_size=int(match.group(1)),
        alpha=float(match.group(2)),
        depth=int(match.group(3)),
        model_path=str(preferred),
        source="fallback_from_surrogate_checkpoint_filename",
    )


def resolve_surrogate_config(
    dataset: str,
    target_model: str,
    *,
    seed_size: int | None = None,
    alpha: float | None = None,
    depth: int | None = None,
) -> SurrogateConfig:
    best = infer_best_surrogate_config(dataset, target_model)
    final = SurrogateConfig(
        dataset=dataset,
        target_model=target_model,
        seed_size=int(seed_size if seed_size is not None else best.seed_size),
        alpha=float(alpha if alpha is not None else best.alpha),
        depth=int(depth if depth is not None else best.depth),
        model_path=best.model_path,
        source=best.source,
    )
    return final


def adversarial_stem(attack: str, target_model: str, seed_size: int, alpha: float, depth: int) -> str:
    return f"{attack}_{target_model}_seed{seed_size}_a{alpha}_d{depth}"


def adversarial_dir(dataset: str, run_tag: str | None = None) -> Path:
    base = Path("data/adversarial") / dataset
    return base if not run_tag else base / "tagged" / run_tag


def transfer_results_dir(run_tag: str | None = None) -> Path:
    base = Path("results/tables")
    return base if not run_tag else base / "tagged" / run_tag
