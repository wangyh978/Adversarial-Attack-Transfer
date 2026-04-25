from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch

from src.models.mlp_surrogate import MLPSurrogate
from src.utils.io import load_json

from src.attacks.fgm import FGMAttack
from src.attacks.pgd import PGDAttack
from src.attacks.mim import MIMAttack
from src.attacks.ti import TIAttack
from src.attacks.cw import CWAttack
from src.attacks.slide import SLIDEAttack


ATTACKS = {
    "fgm": FGMAttack,
    "pgd": PGDAttack,
    "mim": MIMAttack,
    "ti": TIAttack,
    "cw": CWAttack,
    "slide": SLIDEAttack,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--attack", required=True, choices=sorted(ATTACKS))
    parser.add_argument("--seed_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    return parser.parse_args()


def infer_best_config(dataset: str, target_model: str) -> dict:
    best_json = Path("artifacts/metadata") / f"best_surrogate_{dataset}_{target_model}.json"
    if best_json.exists():
        with open(best_json, "r", encoding="utf-8") as f:
            return json.load(f)

    candidates = sorted(
        Path("artifacts/models").glob(f"surrogate_{dataset}_{target_model}_seed*_a*_d*.pt")
    )
    if not candidates:
        raise FileNotFoundError("No surrogate model file and no best surrogate config found.")

    preferred = None
    for p in candidates:
        if "_seed1000_a0.1_d3.pt" in p.name:
            preferred = p
            break
    if preferred is None:
        preferred = candidates[-1]

    m = re.search(r"seed(\d+)_a([0-9.]+)_d(\d+)\.pt$", preferred.name)
    if not m:
        raise ValueError(f"Cannot parse surrogate config from filename: {preferred.name}")

    return {
        "dataset": dataset,
        "target_model": target_model,
        "seed_size": int(m.group(1)),
        "alpha": float(m.group(2)),
        "depth": int(m.group(3)),
        "model_path": str(preferred),
    }


def build_attack(name: str, dataset: str):
    if name == "fgm":
        return FGMAttack(dataset, epsilon=0.5)
    if name == "pgd":
        return PGDAttack(dataset, epsilon=0.5, steps=10, step_size=0.1)
    if name == "mim":
        return MIMAttack(dataset, epsilon=0.5, steps=10, step_size=0.1, decay=1.0)
    if name == "ti":
        return TIAttack(dataset, epsilon=0.5, steps=10, step_size=0.1)
    if name == "cw":
        return CWAttack(dataset, c=0.1, steps=30, lr=1e-2)
    if name == "slide":
        return SLIDEAttack(
            dataset,
            epsilon=0.5,
            steps=10,
            step_size=0.1,
            topk_ratio=0.25,
            random_start=True,
            seed=2026,
        )
    raise ValueError(name)


def main() -> None:
    args = parse_args()
    best = infer_best_config(args.dataset, args.target_model)

    seed_size = int(args.seed_size or best["seed_size"])
    alpha = float(args.alpha or best["alpha"])
    depth = int(args.depth or best["depth"])

    processed_dir = Path("data") / args.dataset / "processed"
    X_test = np.load(processed_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(processed_dir / "y_test.npy")

    feature_info = load_json(Path("artifacts/preprocessors") / f"{args.dataset}_feature_info.json")
    num_classes = int(feature_info["num_classes"])

    model = MLPSurrogate(input_dim=X_test.shape[1], num_classes=num_classes, depth=depth)
    model_path = (
        Path("artifacts/models")
        / f"surrogate_{args.dataset}_{args.target_model}_seed{seed_size}_a{alpha}_d{depth}.pt"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Surrogate checkpoint not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    attack = build_attack(args.attack, args.dataset)
    X_adv, meta = attack.generate(model, X_test, y_test)
    X_adv = X_adv.astype(np.float32, copy=False)

    if X_adv.shape != X_test.shape:
        raise ValueError(f"Adversarial shape mismatch: X_adv={X_adv.shape}, X_test={X_test.shape}")

    out_dir = Path("data/adversarial") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.attack}_{args.target_model}_seed{seed_size}_a{alpha}_d{depth}.parquet"

    # Store adversarial and its exact paired clean sample in the same file.
    # This prevents later perturbation metrics from comparing against a stale or
    # differently ordered X_test.npy.
    adv_cols = [f"f_{i}" for i in range(X_adv.shape[1])]
    clean_cols = [f"orig_f_{i}" for i in range(X_test.shape[1])]

    df_adv = pd.DataFrame(X_adv, columns=adv_cols)
    df_orig = pd.DataFrame(X_test, columns=clean_cols)
    df = pd.concat([df_adv, df_orig], axis=1)
    df["label_true"] = y_test
    df["sample_id"] = np.arange(len(y_test), dtype=np.int64)
    df.to_parquet(out_path, index=False)

    diff = X_adv - X_test
    l2 = np.linalg.norm(diff, ord=2, axis=1)
    linf = np.max(np.abs(diff), axis=1)

    meta.update(
        {
            "dataset": args.dataset,
            "target_model": args.target_model,
            "seed_size": seed_size,
            "alpha": alpha,
            "depth": depth,
            "surrogate_model_path": str(model_path),
            "output_path": str(out_path),
            "paired_clean_features_saved": True,
            "num_features": int(X_test.shape[1]),
            "mean_l2_perturbation_pre_eval": float(np.mean(l2)),
            "max_l2_perturbation_pre_eval": float(np.max(l2)),
            "mean_linf_perturbation_pre_eval": float(np.mean(linf)),
            "max_linf_perturbation_pre_eval": float(np.max(linf)),
        }
    )
    meta_path = out_dir / f"{args.attack}_{args.target_model}_seed{seed_size}_a{alpha}_d{depth}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved:", out_path)
    print("saved:", meta_path)
    print(
        "perturbation check:",
        {
            "mean_l2": meta["mean_l2_perturbation_pre_eval"],
            "max_l2": meta["max_l2_perturbation_pre_eval"],
            "mean_linf": meta["mean_linf_perturbation_pre_eval"],
            "max_linf": meta["max_linf_perturbation_pre_eval"],
        },
    )


if __name__ == "__main__":
    main()
