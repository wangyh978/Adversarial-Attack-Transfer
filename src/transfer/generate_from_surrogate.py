from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.attacks.registry import (
    SUPPORTED_ATTACKS,
    attack_overrides_from_args,
    build_attack,
    default_attack_kwargs,
)
from src.models.mlp_surrogate import MLPSurrogate
from src.transfer.experiment import (
    adversarial_dir,
    adversarial_stem,
    resolve_surrogate_config,
    surrogate_model_path,
)
from src.utils.io import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--attack", required=True, choices=SUPPORTED_ATTACKS)
    parser.add_argument("--seed_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=2026)

    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--step_size", type=float, default=None)
    parser.add_argument("--decay", type=float, default=None)
    parser.add_argument("--topk_ratio", type=float, default=None)
    parser.add_argument("--c_const", type=float, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--attack_lr", type=float, default=None)
    parser.add_argument("--binary_search_steps", type=int, default=None)
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--kernel_sigma", type=float, default=None)
    parser.add_argument("--attack_seed", type=int, default=None)
    parser.add_argument("--attack_batch_size", type=int, default=None)
    parser.add_argument("--random_start", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def sample_test_subset(
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    sample_size: int | None,
    sample_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_size is None or sample_size >= len(y_test):
        indices = np.arange(len(y_test), dtype=np.int64)
        return X_test, y_test, indices

    if sample_size < 1:
        raise ValueError("--sample_size must be >= 1 when provided.")

    indices = np.arange(len(y_test), dtype=np.int64)
    _, sampled_idx = train_test_split(
        indices,
        test_size=sample_size,
        stratify=y_test,
        random_state=sample_seed,
    )
    sampled_idx = np.sort(sampled_idx.astype(np.int64, copy=False))
    return X_test[sampled_idx], y_test[sampled_idx], sampled_idx


def main() -> None:
    args = parse_args()
    config = resolve_surrogate_config(
        args.dataset,
        args.target_model,
        seed_size=args.seed_size,
        alpha=args.alpha,
        depth=args.depth,
    )

    processed_dir = Path("data") / args.dataset / "processed"
    X_test = np.load(processed_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(processed_dir / "y_test.npy")
    X_test, y_test, sample_ids = sample_test_subset(
        X_test,
        y_test,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )

    feature_info = load_json(Path("artifacts/preprocessors") / f"{args.dataset}_feature_info.json")
    num_classes = int(feature_info["num_classes"])

    model = MLPSurrogate(input_dim=X_test.shape[1], num_classes=num_classes, depth=config.depth)
    model_path = surrogate_model_path(
        config.dataset,
        config.target_model,
        config.seed_size,
        config.alpha,
        config.depth,
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Surrogate checkpoint not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    attack_defaults = default_attack_kwargs(args.attack, args.dataset)
    attack_overrides = attack_overrides_from_args(args)
    attack = build_attack(args.attack, args.dataset, **attack_overrides)

    X_adv, meta = attack.generate(model, X_test, y_test)
    X_adv = X_adv.astype(np.float32, copy=False)

    if X_adv.shape != X_test.shape:
        raise ValueError(f"Adversarial shape mismatch: X_adv={X_adv.shape}, X_test={X_test.shape}")

    out_dir = adversarial_dir(args.dataset, args.run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{adversarial_stem(args.attack, args.target_model, config.seed_size, config.alpha, config.depth)}.parquet"

    adv_cols = [f"f_{i}" for i in range(X_adv.shape[1])]
    clean_cols = [f"orig_f_{i}" for i in range(X_test.shape[1])]

    df_adv = pd.DataFrame(X_adv, columns=adv_cols)
    df_orig = pd.DataFrame(X_test, columns=clean_cols)
    df = pd.concat([df_adv, df_orig], axis=1)
    df["label_true"] = y_test
    df["sample_id"] = sample_ids
    df.to_parquet(out_path, index=False)

    diff = X_adv - X_test
    l2 = np.linalg.norm(diff.reshape(diff.shape[0], -1), ord=2, axis=1)
    linf = np.max(np.abs(diff.reshape(diff.shape[0], -1)), axis=1)

    meta.update(
        {
            "dataset": args.dataset,
            "target_model": args.target_model,
            "seed_size": config.seed_size,
            "alpha": config.alpha,
            "depth": config.depth,
            "surrogate_model_path": str(model_path),
            "surrogate_config_source": config.source,
            "attack_defaults": attack_defaults,
            "attack_overrides": attack_overrides,
            "output_path": str(out_path),
            "run_tag": args.run_tag,
            "paired_clean_features_saved": True,
            "num_features": int(X_test.shape[1]),
            "sample_size_requested": args.sample_size,
            "sample_size_actual": int(len(y_test)),
            "sample_seed": int(args.sample_seed),
            "mean_l2_perturbation_pre_eval": float(np.mean(l2)),
            "max_l2_perturbation_pre_eval": float(np.max(l2)),
            "mean_linf_perturbation_pre_eval": float(np.mean(linf)),
            "max_linf_perturbation_pre_eval": float(np.max(linf)),
        }
    )
    meta_path = out_dir / f"{adversarial_stem(args.attack, args.target_model, config.seed_size, config.alpha, config.depth)}_meta.json"
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
