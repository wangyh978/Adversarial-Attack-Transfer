from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.attacks.registry import SUPPORTED_ATTACKS
from src.blackbox.query_api import BlackBoxModel
from src.evaluation.transfer_metrics import compute_transfer_metrics
from src.transfer.experiment import (
    adversarial_dir,
    adversarial_stem,
    resolve_surrogate_config,
    transfer_results_dir,
)
from src.utils.io import save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nsl_kdd", "unsw_nb15"])
    parser.add_argument("--target_model", required=True, choices=["tabnet", "xgb", "gbdt"])
    parser.add_argument("--attack", required=True, choices=SUPPORTED_ATTACKS)
    parser.add_argument("--seed_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    return parser.parse_args()


def _feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    return sorted(cols, key=lambda x: int(str(x).split("_")[-1]))


def main() -> None:
    args = parse_args()
    config = resolve_surrogate_config(
        args.dataset,
        args.target_model,
        seed_size=args.seed_size,
        alpha=args.alpha,
        depth=args.depth,
    )

    adv_path = (
        adversarial_dir(args.dataset, args.run_tag)
        / f"{adversarial_stem(args.attack, args.target_model, config.seed_size, config.alpha, config.depth)}.parquet"
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
            "Regenerate adversarial samples with the current generate_from_surrogate.py."
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

    out_dir = transfer_results_dir(args.run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"transfer_{args.attack}_{args.dataset}_{args.target_model}.csv"
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    metrics = compute_transfer_metrics(result_df)
    metrics["dataset"] = args.dataset
    metrics["target_model"] = args.target_model
    metrics["attack"] = args.attack
    metrics["seed_size"] = config.seed_size
    metrics["alpha"] = config.alpha
    metrics["depth"] = config.depth
    metrics["run_tag"] = args.run_tag
    metrics["clean_feature_source_for_perturbation"] = clean_source

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
