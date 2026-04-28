from __future__ import annotations

from copy import deepcopy

from src.attacks.cw import CWAttack
from src.attacks.fgm import FGMAttack
from src.attacks.mim import MIMAttack
from src.attacks.pgd import PGDAttack
from src.attacks.slide import SLIDEAttack
from src.attacks.ti import TIAttack


ATTACKS = {
    "fgm": FGMAttack,
    "pgd": PGDAttack,
    "mim": MIMAttack,
    "ti": TIAttack,
    "cw": CWAttack,
    "slide": SLIDEAttack,
}

SUPPORTED_ATTACKS = tuple(sorted(ATTACKS))
DEFAULT_ATTACK_SEQUENCE = ["fgm", "pgd", "slide"]


DEFAULT_ATTACK_PROFILES: dict[str, dict] = {
    "fgm": {
        "epsilon": 0.5,
        "batch_size": 4096,
    },
    "pgd": {
        "epsilon": 0.5,
        "steps": 10,
        "step_size": 0.1,
        "random_start": False,
        "seed": 2026,
        "batch_size": 2048,
    },
    "mim": {
        "epsilon": 0.5,
        "steps": 10,
        "step_size": 0.1,
        "decay": 1.0,
        "random_start": False,
        "seed": 2026,
        "batch_size": 2048,
    },
    "ti": {
        "epsilon": 0.5,
        "steps": 12,
        "step_size": 0.08,
        "decay": 1.0,
        "kernel_size": 5,
        "kernel_sigma": 1.0,
        "random_start": False,
        "multi_scale": True,
        "seed": 2026,
        "batch_size": 2048,
    },
    "cw": {
        "c": 0.01,
        "steps": 20,
        "lr": 5e-3,
        "confidence": 0.0,
        "binary_search_steps": 3,
        "abort_early": True,
        "batch_size": 1024,
    },
    "slide": {
        "epsilon": 0.5,
        "steps": 10,
        "step_size": 0.1,
        "topk_ratio": 0.25,
        "random_start": True,
        "seed": 2026,
        "batch_size": 2048,
    },
}


DATASET_ATTACK_OVERRIDES: dict[tuple[str, str], dict] = {
    ("cw", "nsl_kdd"): {
        "c": 0.02,
        "steps": 24,
        "lr": 5e-3,
        "binary_search_steps": 4,
    },
    ("cw", "unsw_nb15"): {
        "steps": 24,
        "lr": 4e-3,
        "batch_size": 512,
    },
    ("ti", "nsl_kdd"): {
        "epsilon": 0.6,
        "steps": 14,
        "step_size": 0.07,
        "kernel_size": 7,
        "kernel_sigma": 1.2,
    },
    ("mim", "unsw_nb15"): {
        "steps": 12,
        "step_size": 0.08,
    },
    ("ti", "unsw_nb15"): {
        "steps": 14,
        "step_size": 0.07,
        "kernel_size": 7,
        "kernel_sigma": 1.2,
    },
}


CLI_ATTACK_FIELDS = (
    "epsilon",
    "steps",
    "step_size",
    "decay",
    "random_start",
    "attack_seed",
    "topk_ratio",
    "c_const",
    "confidence",
    "attack_lr",
    "binary_search_steps",
    "kernel_size",
    "kernel_sigma",
    "attack_batch_size",
)


def default_attack_kwargs(name: str, dataset: str) -> dict:
    if name not in ATTACKS:
        raise ValueError(f"Unsupported attack: {name}")

    kwargs = deepcopy(DEFAULT_ATTACK_PROFILES[name])
    kwargs.update(DATASET_ATTACK_OVERRIDES.get((name, dataset), {}))
    return kwargs


def build_attack(name: str, dataset: str, **overrides):
    kwargs = default_attack_kwargs(name, dataset)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return ATTACKS[name](dataset=dataset, **kwargs)


def attack_overrides_from_args(args) -> dict:
    mapping = {
        "epsilon": "epsilon",
        "steps": "steps",
        "step_size": "step_size",
        "decay": "decay",
        "random_start": "random_start",
        "attack_seed": "seed",
        "topk_ratio": "topk_ratio",
        "c_const": "c",
        "confidence": "confidence",
        "attack_lr": "lr",
        "binary_search_steps": "binary_search_steps",
        "kernel_size": "kernel_size",
        "kernel_sigma": "kernel_sigma",
        "attack_batch_size": "batch_size",
    }
    overrides = {}
    for arg_name, kw_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        overrides[kw_name] = value
    return overrides
