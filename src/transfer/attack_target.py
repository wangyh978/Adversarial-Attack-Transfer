from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from src.blackbox.query_api import BlackBoxModel
from src.evaluation.transfer_metrics import compute_transfer_metrics
from src.utils.io import save_json


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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--attack", required=True, choices=["fgm", "pgd", "mim", "ti", "cw", "slide"])
    parser.add_argument("--seed_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    return parser.parse_args()


def _feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    return sorted(cols, key=lambda x: int(str(x).split("_")[-1]))


def main() -> None:
    args = parse_args()
    best = infer_best_config(args.dataset, args.target_model)

    seed_size = int(args.seed_size or best["seed_size"])
    alpha = float(args.alpha or best["alpha"])
    depth = int(args.depth or best["depth"])

    adv_path = (
        Path("data/adversarial")
        / args.dataset
        / f"{args.attack}_{args.target_model}_seed{seed_size}_a{alpha}_d{depth}.parquet"
    )
    if not adv_path.exists():
        raise FileNotFoundError(f"Adversarial file not found: {adv_path}")

    adv_df = pd.read_parquet(adv_path)

    feature_cols = _feature_columns(adv_df, "f_")
    X_adv = adv_df[feature_cols].to_numpy(dtype=np.float32)
    y_true = adv_df["label_true"].to_numpy()

    orig_cols = _feature_columns(adv_df, "orig_f_")
    if orig_cols:
        X_clean = adv_df[orig_cols].to_numpy(dtype=np.float32)
        clean_source = "paired_orig_features_in_adversarial_file"
    else:
        clean_dir = Path("data") / args.dataset / "processed"
        X_clean = np.load(clean_dir / "X_test.npy").astype(np.float32)
        clean_source = "processed_X_test_npy_fallback"

    if X_clean.shape != X_adv.shape:
        raise ValueError(
            f"Clean/adversarial shape mismatch: clean={X_clean.shape}, adv={X_adv.shape}. "
            "Regenerate adversarial samples with the patched generate_from_surrogate.py."
        )

    blackbox = BlackBoxModel(args.dataset, args.target_model)
    y_clean_pred = blackbox.predict_label(X_clean.copy())
    y_adv_pred = blackbox.predict_label(X_adv.copy())

    diff = X_adv - X_clean
    l2 = np.linalg.norm(diff.reshape(diff.shape[0], -1), ord=2, axis=1)
    linf = np.max(np.abs(diff.reshape(diff.shape[0], -1)), axis=1)

    clean_correct = y_clean_pred == y_true
    adv_wrong = y_adv_pred != y_true

    result_df = pd.DataFrame(
        {
            "sample_id": adv_df["sample_id"].to_numpy() if "sample_id" in adv_df.columns else np.arange(len(y_true)),
            "attack_name": args.attack,
            "target_model": args.target_model,
            "label_true": y_true,
            "pred_clean_target": y_clean_pred,
            "pred_adv_target": y_adv_pred,
            "clean_correct": clean_correct,
            "adv_wrong": adv_wrong,
            "is_transfer_success": clean_correct & adv_wrong,
            "is_transfer_success_legacy": adv_wrong,
            "l2_perturbation": l2,
            "linf_perturbation": linf,
        }
    )

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"transfer_{args.attack}_{args.dataset}_{args.target_model}.csv"
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    metrics = compute_transfer_metrics(result_df)
    metrics["dataset"] = args.dataset
    metrics["target_model"] = args.target_model
    metrics["attack"] = args.attack
    metrics["seed_size"] = seed_size
    metrics["alpha"] = alpha
    metrics["depth"] = depth
    metrics["clean_feature_source_for_perturbation"] = clean_source

    # Diagnostics for spotting rare outliers without hiding them.
    for q in [0.5, 0.9, 0.95, 0.99, 0.999]:
        metrics[f"l2_q{q}"] = float(np.quantile(l2, q))
        metrics[f"linf_q{q}"] = float(np.quantile(linf, q))

    metrics["num_linf_gt_1"] = int(np.sum(linf > 1.0))
    metrics["num_l2_gt_5"] = int(np.sum(l2 > 5.0))

    metrics_path = out_dir / f"transfer_{args.attack}_{args.dataset}_{args.target_model}_metrics.json"
    save_json(metrics, metrics_path)

    print(metrics)
    print("saved:", out_csv)
    print("saved:", metrics_path)


if __name__ == "__main__":
    main()
