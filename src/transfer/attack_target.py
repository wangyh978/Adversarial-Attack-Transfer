from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import re
import json

from src.blackbox.query_api import BlackBoxModel
from src.evaluation.transfer_metrics import compute_transfer_metrics
from src.utils.io import load_json, save_json


def infer_best_config(dataset: str, target_model: str) -> dict:
    # 1) 先尝试读取正式的 best 配置
    best_json = Path("artifacts/metadata") / f"best_surrogate_{dataset}_{target_model}.json"
    if best_json.exists():
        with open(best_json, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2) 如果没有，就从现有 surrogate 模型里挑一组
    candidates = sorted(
        Path("artifacts/models").glob(f"surrogate_{dataset}_{target_model}_seed*_a*_d*.pt")
    )
    if not candidates:
        raise FileNotFoundError("未找到 surrogate 模型文件，也未找到 best surrogate 配置")

    # 优先选你当前已经验证过效果最好的一组
    preferred = None
    for p in candidates:
        if "_seed500_a0.1_d7.pt" in p.name:
            preferred = p
            break
    if preferred is None:
        preferred = candidates[-1]

    m = re.search(r"seed(\d+)_a([0-9.]+)_d(\d+)\.pt$", preferred.name)
    if not m:
        raise ValueError(f"无法从文件名解析 surrogate 配置: {preferred.name}")

    return {
        "dataset": dataset,
        "target_model": target_model,
        "seed_size": int(m.group(1)),
        "alpha": float(m.group(2)),
        "depth": int(m.group(3)),
        "model_path": str(preferred)
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


def main() -> None:
    args = parse_args()
    best = infer_best_config(args.dataset, args.target_model)
    seed_size = int(args.seed_size or best["seed_size"])
    alpha = float(args.alpha or best["alpha"])
    depth = int(args.depth or best["depth"])

    adv_path = Path("data/adversarial") / args.dataset / f"{args.attack}_{args.target_model}_seed{seed_size}_a{alpha}_d{depth}.parquet"
    adv_df = pd.read_parquet(adv_path)

    feature_cols = [c for c in adv_df.columns if str(c).startswith("f_")]
    X_adv = adv_df[feature_cols].to_numpy(dtype=np.float32)
    y_true = adv_df["label_true"].to_numpy()

    clean_dir = Path("data") / args.dataset / "processed"
    X_clean = np.load(clean_dir / "X_test.npy").astype(np.float32)

    blackbox = BlackBoxModel(args.dataset, args.target_model)
    y_clean_pred = blackbox.predict_label(X_clean)
    y_adv_pred = blackbox.predict_label(X_adv)

    result_df = pd.DataFrame({
        "sample_id": np.arange(len(y_true)),
        "attack_name": args.attack,
        "target_model": args.target_model,
        "label_true": y_true,
        "pred_clean_target": y_clean_pred,
        "pred_adv_target": y_adv_pred,
    })
    result_df["is_transfer_success"] = result_df["pred_adv_target"] != result_df["label_true"]

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"transfer_{args.attack}_{args.dataset}_{args.target_model}.csv"
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    metrics = compute_transfer_metrics(result_df)
    save_json(metrics, out_dir / f"transfer_{args.attack}_{args.dataset}_{args.target_model}_metrics.json")

    print(metrics)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
